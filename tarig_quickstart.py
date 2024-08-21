import sys
sys.path.append("./")
import argparse
import open3d as o3d
import numpy as np
from datasets.joint_categories import categories, left_idx, right_idx
from utils.nm_mesh import NonManifoldMesh
from gen_dataset import get_geo_edges
from geometric_proc.common_ops import calc_surface_geodesic
from models.TARig import TARigNet
import torch.nn.functional as F
import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from post_process.skeleton_building import build_primary_skeleton
from utils.vis_utils import show_obj_skel_o3dmesh_multiroot, draw_shifted_pts_by_mesh
from utils.skin_utils import bind_skin
from utils.tree_utils import TreeNode
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_obj(mesh_v):
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


def create_single_data(mesh_filename, normalize_mesh=False):
    """
    create input data for the network. The data is wrapped by Data structure in pytorch-geometric library
    :param mesh_filaname: name of the input mesh
    :return: wrapped data, voxelized mesh, and geodesic distance matrix of all vertices
    """
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    pivot, scale = None, None
    if normalize_mesh:
        mesh_v, pivot, scale = normalize_obj(np.asarray(mesh.vertices))
        mesh_f = np.asarray(mesh.triangles)
        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh_v), triangles=o3d.utility.Vector3iVector(mesh_f))
        # o3d.io.write_triangle_mesh(mesh_filename.replace(".obj", "_normalized.obj"), mesh)
    nm_mesh = NonManifoldMesh(mesh)
    mesh.compute_vertex_normals()
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    # vertices
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = torch.from_numpy(v).float()

    # topology edges
    print("     gathering topological edges.")
    tpl_e = nm_mesh.get_edges().T
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = calc_surface_geodesic(mesh)

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))
    # geo_e = deepcopy(tpl_e)

    # batch
    batch = torch.zeros(len(v), dtype=torch.long)
    data = Data(x=v[:, 3:6], pos=v[:, 0:3], tpl_edge_index=tpl_e, geo_edge_index=geo_e, batch=batch)
    return data, nm_mesh, mesh, pivot, scale


def predict_template(args):
    model = TARigNet(n_joints=args.n_joints,
                     input_normal=args.input_normal,
                     dropout=0)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    print("load model successfully!")

    data, nm_mesh, mesh, pivot, scale = create_single_data(args.mesh_filename, args.normalize_mesh)
    data = data.to(device)
    kpts, heatmaps, skinning_weights, conflow = model(data, 1)

    res = {}
    joint_pred_i, primary_heatmap_pred_i, secondary_heatmap_pred_i, conflow_pred_i = \
        kpts[0], heatmaps[0][:, :args.n_joints], heatmaps[0][:, args.n_joints:], conflow
        
    primary_joints = joint_pred_i.data.to("cpu").numpy()
    if args.is_sym:
        primary_joints_sym = copy.deepcopy(primary_joints)
        to_plane_x = 0.5 * (primary_joints[left_idx, 0] - primary_joints[right_idx, 0])
        to_plane_yz = 0.5 * (primary_joints[left_idx, 1:] + primary_joints[right_idx, 1:])
        primary_joints_sym[left_idx, 0] = to_plane_x
        primary_joints_sym[right_idx, 0] = -to_plane_x
        primary_joints_sym[left_idx, 1:] = primary_joints_sym[right_idx, 1:] = to_plane_yz
        primary_joints = primary_joints_sym

    # res['joint_pred'] = joint_pred_i.data.to("cpu").numpy()
    res['joint_pred'] = primary_joints
    res['heatmap_pred'] = primary_heatmap_pred_i.data.to("cpu").numpy()
    res['secondary_heatmap_pred'] = secondary_heatmap_pred_i.data.to("cpu").numpy()
    skin_weights_pred_i = F.softmax(skinning_weights, dim=1)
    res['skin_weights_pred'] = skin_weights_pred_i.data.to("cpu").numpy()
    res['conflow_pred'] = conflow_pred_i.data.to("cpu").numpy()
      
    pred_primary_skel = build_primary_skeleton(mesh, res["joint_pred"])
    img_primary_skel = show_obj_skel_o3dmesh_multiroot(mesh, pred_primary_skel)
    if args.normalize_mesh:
        pred_primary_skel.normalize(scale, -pivot)
        pred_primary_skel.save(args.mesh_filename.replace(".obj", "_rig.txt"))
    weight_mat = res['skin_weights_pred']
    pred_skel = bind_skin(np.array(mesh.vertices), pred_primary_skel, weight_mat, pred_primary_skel.get_node_names())
    pred_skel.save(args.mesh_filename.replace(".obj", "_rig.txt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TARig")
    parser.add_argument("--checkpoint", default="C:/Project/Python/tarig/checkpoints/20240407_195613/model_best.pth.tar", type=str)
    parser.add_argument("--mesh_filename", default="C:/Project/Dataset/thesis/7540_flower/7540_flower.obj", type=str)
    parser.add_argument("--normalize_mesh", default=1, type=int, help="whether to normalize the mesh")
    # hyper-parameters
    parser.add_argument("--n_joints", default=21, type=int, help="number of joints")
    parser.add_argument('--input_normal', action='store_true')
    parser.add_argument('--is_sym', action='store_true')
    # parser.set_defaults(is_sym=True)

    args = parser.parse_args()
    predict_template(args)

