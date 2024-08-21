"""
jingma
"""
import numpy as np
from utils.rig_parser import Info, TreeNode

def transfer_to_other_mesh(mesh1, mesh2, rig1):
    """
    ToDO: change the code, the current implementation is really buggy which bind the close body parts to the same joints!!!
    convert the the rig of the original model to
    Just assign skinning weight based on nearest neighbor
    :param mesh1: original o3d.TriMesh
    :param mesh2: remeshed o3d.TriMesh
    :param rig1: predicted rig
    :return: predicted rig for original mesh
    """
    tranfer_rig = Info()

    vert_ori = np.asarray(mesh1.vertices)
    vert_remesh = np.asarray(mesh2.vertices)

    vertice_distance = np.sqrt(np.sum((vert_ori[np.newaxis, ...] - vert_remesh[:, np.newaxis, :]) ** 2, axis=2))
    vertice_raw_id = np.argmin(vertice_distance, axis=1)

    tranfer_rig.root = rig1.root
    tranfer_rig.joint_pos = rig1.joint_pos
    new_skin = []
    for v in range(len(vert_remesh)):
        skin_v = [v]
        v_nn = vertice_raw_id[v]
        skin_v += rig1.joint_skin[v_nn][1:]
        new_skin.append(skin_v)
    tranfer_rig.joint_skin = new_skin
    return tranfer_rig


def post_filter(skin_weights, topology_edge, num_ring=1):
    skin_weights_new = np.zeros_like(skin_weights)
    for v in range(len(skin_weights)):
        adj_verts_multi_ring = []
        current_seeds = [v]
        for r in range(num_ring):
            adj_verts = []
            for seed in current_seeds:
                adj_edges = topology_edge[:, np.argwhere(topology_edge == seed)[:, 1]]
                adj_verts_seed = list(set(adj_edges.flatten().tolist()))
                adj_verts_seed.remove(seed)
                adj_verts += adj_verts_seed
            adj_verts_multi_ring += adj_verts
            current_seeds = adj_verts
        adj_verts_multi_ring = list(set(adj_verts_multi_ring))
        if v in adj_verts_multi_ring:
            adj_verts_multi_ring.remove(v)
        skin_weights_neighbor = [skin_weights[int(i), :][np.newaxis, :] for i in adj_verts_multi_ring]
        skin_weights_neighbor = np.concatenate(skin_weights_neighbor, axis=0)
        #max_bone_id = np.argmax(skin_weights[v, :])
        #if np.sum(skin_weights_neighbor[:, max_bone_id]) < 0.17 * len(skin_weights_neighbor):
        #    skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
        #else:
        #    skin_weights_new[v, :] = skin_weights[v, :]
        skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)

    #skin_weights_new[skin_weights_new.sum(axis=1) == 0, :] = skin_weights[skin_weights_new.sum(axis=1) == 0, :]
    return skin_weights_new


def bind_skin(mesh_v, rig, weight_mat, joint_names):
    rig.joint_skin.clear()
    # joint_pos = np.concatenate([np.array(rig.joint_pos[name]).reshape((1, 3)) for name in joint_names], axis=0)
    # print("bind_skin rig.joint_dict\n", rig.joint_dict.keys())
    joint_pos = np.concatenate([np.array(rig.joint_dict[name].pos).reshape((1, 3)) for name in joint_names], axis=0)
    def nearest_joint(v_id):
        v_pos = mesh_v[v_id, :]
        dist = np.linalg.norm(joint_pos - v_pos, axis=1)
        return np.argmin(dist)

    for i in range(weight_mat.shape[0]):
        skin = [i]

        if np.all(weight_mat[i, :] < 1e-3): # no joint has influence for the vertex
            joint_id = nearest_joint(i)
            weight_mat[i, joint_id] = 1

        weight_mat[i, :] /= np.sum(weight_mat[i, :])
        for j in range(weight_mat.shape[1]):
            if weight_mat[i, j] >= 1e-3:
                skin.append(joint_names[j])
                skin.append(weight_mat[i, j])

        rig.joint_skin.append(skin)
    return rig


def retarget_rig(rig, verts, tgt_joint_names):
    """
    Parameters
    ----------
    rig: input_rig
    verts: input mesh vertices
    tgt_joint_names: target_joint_names to retarget the input_rig to

    Returns: retargeted rig
    -------

    """
    num_v = len(verts)
    rig_tmp = Info()
    src_joints = rig.get_nodes()
    src_joint_names = [joint.name for joint in src_joints]

    src_weight = rig.extract_weight_matrix(num_v)
    tgt_weight = np.zeros((num_v, len(tgt_joint_names)))
    tpl_dist_mat = rig.tpl_dist_matrix()
    tpl_dist_max = np.amax(tpl_dist_mat)
    geo_dist_mat = rig.geo_dist_matrix()
    tgt_joint_pos = np.concatenate([np.array(src_joints[src_joint_names.index(name)].pos).reshape((1,3)) for name in tgt_joint_names if name in src_joint_names], axis=0)

    def neighbors_in_tgt(neighbors):
        return [nei for nei in neighbors if src_joint_names[nei] in tgt_joint_names]

    def nearest_neighbor(i, neighbors):
        idx = np.argmin(geo_dist_mat[i, neighbors])
        return neighbors[idx]

    C = np.zeros((len(tgt_joint_names), 3))
    BE = []
    for j in range(len(src_joints)):
        # search upwards to parent
        joint = src_joints[j]
        while joint.parent is not None and joint.name not in tgt_joint_names:
            joint = joint.parent

        # filter out the root_joint that is not in tgt_joints: 14783, 14432, 7548, 18342, 9826
        # find nearest neighbor in the graph, and merge the weight to its nearest neighbor
        if joint.parent is None and joint.name not in tgt_joint_names: # root joint that is not in tgt joint_names
            # print('Alert!', 'None' if joint.parent is None else joint.parent.name, f"{src_joint_names[j]}---{joint.name}")
            thresh = 1
            neighbors = np.nonzero(tpl_dist_mat[j, :] == thresh)[0]
            neis = neighbors_in_tgt(neighbors)
            while len(neis) == 0 and thresh <=tpl_dist_max:
                thresh += 1
                neighbors = np.nonzero(tpl_dist_mat[j, :] == thresh)[0]
                neis = neighbors_in_tgt(neighbors)
            min_nei = nearest_neighbor(j, neis)
            joint = src_joints[min_nei]

        # joint must be in tgt_joints
        idx = tgt_joint_names.index(joint.name)
        tgt_weight[:, idx] += src_weight[:, j]

        joint_p = joint.parent
        while joint_p is not None and joint_p.name not in tgt_joint_names:
            joint_p = joint_p.parent

        bone_edge = [idx, -1 if joint_p is None else tgt_joint_names.index(joint_p.name)]
        if bone_edge not in BE:
            BE.append(bone_edge)
            C[idx, :] = joint.pos

    # build skeleton
    BE = np.asarray(BE, dtype=np.int)
    nodes = {tgt_joint_names[i]: None for i in range(len(tgt_joint_names))}
    for idx in range(len(tgt_joint_names)):
        tgt_joint_name = tgt_joint_names[idx]
        nodes[tgt_joint_name] = TreeNode(tgt_joint_name, C[idx, :])

    for i in range(len(BE)):
        c = BE[i, 0] # child
        p = BE[i, 1] # parent
        node = nodes[tgt_joint_names[c]]
        # print(BE[i], tgt_joint_names[p], tgt_joint_names[c])
        if p == -1:
            rig_tmp.roots.append(node)
        else:
            parent_name = tgt_joint_names[p]
            node.parent = nodes[parent_name]
            nodes[parent_name].children.append(node)
    rig_tmp.joint_dict = nodes
    rig_tmp.template2auxilary = rig.template2auxilary
    

    # attach weight
    rig_tmp = bind_skin(verts, rig_tmp, tgt_weight, tgt_joint_names)
    return rig_tmp


def compute_hierarchical_skinning_mat(mesh_v, rig_info, categories):
    src_joint_names = rig_info.get_node_names()
    tgt_joint_names = [joint_name for joint_name in src_joint_names if joint_name in categories]

    rig_tgt = retarget_rig(rig_info, mesh_v, tgt_joint_names)
    weight = rig_tgt.extract_weight_matrix(len(mesh_v))
    node_names = rig_tgt.get_node_names()
    print(f"tgt_node_names={node_names}")
    W_cat = np.zeros((len(mesh_v), len(categories)))
    for node_name in node_names:
        W_cat[:, categories.index(node_name)] += weight[:, node_names.index(node_name)]
    return W_cat
