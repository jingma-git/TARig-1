import os
import cv2
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from utils.line_mesh import LineMesh


def drawSphere(center, radius, color=[0.0,0.0,0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def drawSphereWireframe(center, radius, color=[0.0,0.0,0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)


    # Assuming you have a TriangleMesh object 'mesh_sphere'
    # You would need to manually extract its edges and create a LineSet

    edges = set()  # Use a set to avoid duplicate edges
    for triangle in mesh_sphere.triangles:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted([triangle[i], triangle[j]]))
                edges.add(edge)

    lines = list(edges)
    lines = [[line[0], line[1]] for line in lines]  # Format for LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(mesh_sphere.vertices)),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color(color)  # Set line color

    return line_set


def drawCone(bottom_center, top_position, color=[0.6, 0.6, 0.9]):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.007, height=np.linalg.norm(top_position - bottom_center)+1e-6)
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center)+1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4: # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    T = bottom_center + 5e-3 * line2
    #print(R)
    cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    return cone


def show_obj_skel(mesh_name, root):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    vis.add_geometry(drawSphere(root.pos, 0.01, color=[0.1, 0.1, 0.1]))
    this_level = root.children
    while this_level:
        next_level = []
        for p_node in this_level:
            vis.add_geometry(drawSphere(p_node.pos, 0.008, color=[1.0, 0.0, 0.0])) # [0.3, 0.1, 0.1]
            vis.add_geometry(drawCone(np.array(p_node.parent.pos), np.array(p_node.pos)))
            next_level+=p_node.children
        this_level = next_level

    #param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    #ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    #vis.update_geometry()
    #vis.poll_events()
    #vis.update_renderer()

    #param = ctr.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def draw_shifted_pts(mesh_name, pts, weights=None):
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    pred_joints = o3d.geometry.PointCloud()
    pred_joints.points = o3d.utility.Vector3dVector(pts)
    if weights is None:
        color_joints = [[1.0, 0.0, 0.0] for i in range(len(pts))]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('YlOrRd')
        #weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        #weights = 1 / (1 + np.exp(-weights))
        color_joints = cmap(weights.squeeze())
        color_joints = color_joints[:, :-1]
    pred_joints.colors = o3d.utility.Vector3dVector(color_joints)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    vis.add_geometry(pred_joints)

    param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    #vis.run()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    #param = ctr.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def draw_shifted_pts_by_mesh(mesh, pts, weights=None):
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    pred_joints = o3d.geometry.PointCloud()
    pred_joints.points = o3d.utility.Vector3dVector(pts)
    if weights is None:
        color_joints = [[1.0, 0.0, 0.0] for i in range(len(pts))]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('turbo') # YlOrRd
        #weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        #weights = 1 / (1 + np.exp(-weights))
        color_joints = cmap(weights.squeeze())
        color_joints = color_joints[:, :-1]
    pred_joints.colors = o3d.utility.Vector3dVector(color_joints)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    vis.add_geometry(pred_joints)

    # param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    # ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    # vis.update_geometry()
    # vis.poll_events()
    # vis.update_renderer()

    #param = ctr.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def draw_joints(mesh_name, pts):
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    for joint_pos in pts:
        vis.add_geometry(drawSphere(joint_pos, 0.006, color=[1.0, 0.0, 0.0]))

    param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    #vis.run()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def volume_to_cubes(volume, threshold=0, dim=[1., 1., 1.]):
    #o = np.array([-dim[0]/2., -dim[1]/2., -dim[2]/2.])
    o = np.array([0, 0, 0])
    step = np.array([dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2]])
    points = []
    lines = []
    for x in range(1, volume.shape[0]-1):
        for y in range(1, volume.shape[1]-1):
            for z in range(1, volume.shape[2]-1):
                pos = o + np.array([x, y, z]) * step
                if volume[x, y, z] > threshold:
                    vidx = len(points)
                    POS = pos + step*0.95
                    xx = pos[0]
                    yy = pos[1]
                    zz = pos[2]
                    XX = POS[0]
                    YY = POS[1]
                    ZZ = POS[2]

                    points.append(np.array([xx, yy, zz])[np.newaxis, :])
                    points.append(np.array([xx, YY, zz])[np.newaxis, :])
                    points.append(np.array([XX, YY, zz])[np.newaxis, :])
                    points.append(np.array([XX, yy, zz])[np.newaxis, :])
                    points.append(np.array([xx, yy, ZZ])[np.newaxis, :])
                    points.append(np.array([xx, YY, ZZ])[np.newaxis, :])
                    points.append(np.array([XX, YY, ZZ])[np.newaxis, :])
                    points.append(np.array([XX, yy, ZZ])[np.newaxis, :])

                    lines.append(np.array([vidx + 1, vidx + 2]))
                    lines.append(np.array([vidx + 2, vidx + 6]))
                    lines.append(np.array([vidx + 6, vidx + 5]))
                    lines.append(np.array([vidx + 1, vidx + 5]))

                    lines.append(np.array([vidx + 1, vidx + 1]))
                    lines.append(np.array([vidx + 3, vidx + 3]))
                    lines.append(np.array([vidx + 7, vidx + 7]))
                    lines.append(np.array([vidx + 5, vidx + 5]))

                    lines.append(np.array([vidx + 0, vidx + 3]))
                    lines.append(np.array([vidx + 0, vidx + 4]))
                    lines.append(np.array([vidx + 4, vidx + 7]))
                    lines.append(np.array([vidx + 7, vidx + 3]))

    return points, lines


def show_mesh_vox(mesh_filename, vox):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vox_pts, vox_lines = volume_to_cubes(vox.data)
    vox_pts = np.concatenate(vox_pts, axis=0)
    line_set_vox = o3d.geometry.LineSet()
    line_set_vox.points = o3d.utility.Vector3dVector(vox_pts+np.array(vox.translate)[np.newaxis, :])
    line_set_vox.lines = o3d.utility.Vector2iVector(vox_lines)
    colors = [[0.0, 0.0, 1.0] for i in range(len(vox_lines))]
    line_set_vox.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set_vox)

    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    vis.run()
    vis.destroy_window()

    return



def show_obj_skel_o3dmesh(mesh, root):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector(
        [[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))]
    )
    vis.add_geometry(mesh_ls)

    # vis.add_geometry(drawSphere(root.pos, 0.01, color=[0.1, 0.1, 0.1]))
    vis.add_geometry(drawSphere(root.pos, 0.008, color=[65. / 255, 105. / 255, 225. / 255]))
    this_level = root.children
    while this_level:
        next_level = []
        for p_node in this_level:
            # vis.add_geometry(
            #     drawSphere(p_node.pos, 0.008, color=[1.0, 0.0, 0.0])
            # )  # [0.3, 0.1, 0.1]
            # vis.add_geometry(
            #     drawCone(np.array(p_node.parent.pos), np.array(p_node.pos))
            # )

            vis.add_geometry(
                drawSphere(p_node.pos, 0.008, color=[65. / 255, 105. / 255, 225. / 255])
            )
            vis.add_geometry(
                drawCone(np.array(p_node.parent.pos), np.array(p_node.pos), color=[0, 0, 1])
            )
            next_level += p_node.children
        this_level = next_level


    vis.run()

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image



def show_boneflow(mesh, flow, root=None):
    # show mesh
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector(
        [[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))]
    )

    mesh_v = np.asarray(mesh.vertices)
    num_v = len(mesh_v)
    to_v = mesh_v + flow * 0.01
    # show flow
    dpts = np.concatenate((mesh_v, to_v), axis=0)
    dlines = o3d.geometry.LineSet()
    dlines.points = o3d.utility.Vector3dVector(dpts)
    dlines.lines = o3d.utility.Vector2iVector(
        [
            [i, i+num_v]
            for i in range(num_v)
        ]
    )
    colors = [[0.0, 0.0, 1.0] for i in range(num_v)]
    dlines.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(dlines)
    vis.add_geometry(mesh_ls)

    if root is not None:
        vis.add_geometry(drawSphere(root.pos, 0.008, color=[65. / 255, 105. / 255, 225. / 255]))
        this_level = root.children
        while this_level:
            next_level = []
            for p_node in this_level:
                # vis.add_geometry(
                #     drawSphere(p_node.pos, 0.008, color=[1.0, 0.0, 0.0])
                # )  # [0.3, 0.1, 0.1]
                # vis.add_geometry(
                #     drawCone(np.array(p_node.parent.pos), np.array(p_node.pos))
                # )

                vis.add_geometry(
                    drawSphere(p_node.pos, 0.008, color=[65. / 255, 105. / 255, 225. / 255])
                )
                vis.add_geometry(
                    drawCone(np.array(p_node.parent.pos), np.array(p_node.pos), color=[0, 0, 1])
                )
                next_level += p_node.children
            this_level = next_level

    vis.run()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image



def show_obj_skel_o3dmesh_multiroot(mesh, info, boneflow=None, response_verts=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # ctr = vis.get_view_control()

    # draw mesh
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector(
        [[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))]
    )
    vis.add_geometry(mesh_ls)

    # draw skeleton
    for root in info.roots:
        vis.add_geometry(drawSphere(root.pos, 0.008, color=[65. / 255, 105. / 255, 225. / 255]))
        this_level = root.children
        while len(this_level)>0:
            next_level = []
            for p_node in this_level:
                # vis.add_geometry(
                #     drawSphere(p_node.pos, 0.008, color=[1.0, 0.0, 0.0])
                # )  # [0.3, 0.1, 0.1]
                # vis.add_geometry(
                #     drawCone(np.array(p_node.parent.pos), np.array(p_node.pos))
                # )

                vis.add_geometry(
                    drawSphere(p_node.pos, 0.008, color=[65. / 255, 105. / 255, 225. / 255])
                )
                vis.add_geometry(
                    drawCone(np.array(p_node.parent.pos), np.array(p_node.pos), color=[0, 0, 1])
                )
                next_level += p_node.children
            this_level = next_level
        

    # draw boneflow
    if boneflow is not None:
        if response_verts is not None:
            mesh_v = np.asarray(mesh.vertices)[response_verts, :]
            num_v = len(mesh_v)
            to_v = mesh_v + boneflow[response_verts, :] * 0.01

            dpts = np.concatenate((mesh_v, to_v), axis=0)
            dlines = o3d.geometry.LineSet()
            dlines.points = o3d.utility.Vector3dVector(dpts)
            dlines.lines = o3d.utility.Vector2iVector(
                [
                    [i, i+len(response_verts)]
                    for i in range(len(response_verts))
                ]
            )
            colors = [[0.0, 0.0, 1.0] for i in range(len(response_verts))]
            dlines.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(dlines)            

        else:
            mesh_v = np.asarray(mesh.vertices)
            num_v = len(mesh_v)
            to_v = mesh_v + boneflow * 0.01
            # show flow
            dpts = np.concatenate((mesh_v, to_v), axis=0)
            dlines = o3d.geometry.LineSet()
            dlines.points = o3d.utility.Vector3dVector(dpts)
            dlines.lines = o3d.utility.Vector2iVector(
                [
                    [i, i+num_v]
                    for i in range(num_v)
                ]
            )
            colors = [[0.0, 0.0, 1.0] for i in range(num_v)]
            dlines.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(dlines)            

    vis.run()

    image = None
    # image = vis.capture_screen_float_buffer()
    # vis.destroy_window()
    # image = np.asarray(image) * 255
    # image = image.astype(np.uint8)
    return image



def show_obj_skel_o3dmesh_with_joint_featuresize(mesh, root, fs):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector(
        [[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))]
    )
    vis.add_geometry(mesh_ls)

    color = np.random.rand(3)
    vis.add_geometry(drawSphere(root.pos, 0.008, color=color))
    if fs[root.name] > 1e-4:
        vis.add_geometry(drawSphereWireframe(root.pos, fs[root.name], color=color))
    this_level = root.children
    while this_level:
        next_level = []
        for p_node in this_level:
            color = np.random.rand(3)
            vis.add_geometry(
                drawSphere(p_node.pos, 0.008, color=color)
            )
            if fs[p_node.name] > 1e-4:
                vis.add_geometry(
                    drawSphereWireframe(p_node.pos, fs[p_node.name], color=color)
                )
            vis.add_geometry(
                drawCone(np.array(p_node.parent.pos), np.array(p_node.pos), color=[0, 0, 1])
            )
            next_level += p_node.children
        this_level = next_level


    vis.run()

    # image = vis.capture_screen_float_buffer()
    # vis.destroy_window()
    # image = np.asarray(image) * 255
    # image = image.astype(np.uint8)
    # return image



def show_o3dmesh(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    # draw mesh
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector(
        [[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))]
    )
    vis.add_geometry(mesh_ls)

    vis.run()

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def show_o3dmesh_kept_edges(mesh, keep):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    line_set = o3d.geometry.LineSet()
    line_set.points = mesh.vertices
    line_set.lines = o3d.utility.Vector2iVector(keep)
    line_set.colors = o3d.utility.Vector3dVector(
        [[1.0, 0, 0] for i in range(len(line_set.lines))]
    )

    vis.add_geometry(line_set)
    vis.run()

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def show_chains_with_skel(mesh, paths, skel=None, show_end=True, len_thresh=math.inf):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    points = np.asarray(mesh.vertices)
    for path in paths:
        # if len(path) < 5:
        #     continue
        chain = points[path]
        # if chain_length(chain) < len_thresh:
        #     continue

        points_path = [points[path[i], :] for i in range(len(path))]
        lines_path = [[i, i+1] for i in range(0, len(path)-1)]
        colors_path = [[1, 0, 0] for i in range(len(lines_path))]
        line_mesh = LineMesh(points_path, lines_path, colors_path, radius=0.002)
        line_mesh = line_mesh.cylinder_segments
        for line in line_mesh:
            vis.add_geometry(line)

        if show_end:
            # path start
            vis.add_geometry(
                drawSphere(points[path[0], :], 0.005, color=[1, 0, 0]) # red
            )
            # path end
            vis.add_geometry(
                drawSphere(points[path[-1], :], 0.005, color=[0, 0, 1]) # blue
            )

    # drawSkel(vis, skel.root)
    vis.run()
    vis.destroy_window()