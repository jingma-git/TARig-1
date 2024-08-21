from utils.rig_parser import Info, TreeNode
from datasets.joint_categories import categories, kpts_parent_id

def build_primary_skeleton(mesh, primary_joints):
    all_joints = []
    for i in range(len(primary_joints)):
        joint = TreeNode(categories[i], primary_joints[i, :], parent=None)
        all_joints.append(joint)

    for i in range(len(primary_joints)):
        pid = kpts_parent_id[i]
        if pid >= 0:
            all_joints[i].parent = all_joints[pid]

    info = Info()
    info.joint_dict = {joint.name:joint for joint in all_joints}
    info.create_connections()
    info.roots = [all_joints[0]]
    return info

