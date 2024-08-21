"""
jingma: multi-roots skeleton
"""

import numpy as np
from utils.tree_utils import TreeNode


class Info:
    """
    Wrap class for rig information
    """
    def __init__(self, filename=None):
        self.joint_dict = {} # key: joint_name, val: TreeNode
        self.joint_skin = [] # skinning weights
        self.roots = [] # support for multi roots skeleton
        if filename is not None:
            self.load(filename)
        self.create_template2auxilary()

    def load(self, filename):
        with open(filename, 'r') as f_txt:
            lines = f_txt.readlines()
        for line in lines:
            word = line.split()
            if word[0] == 'joints':
                joint_name = word[1].strip()
                joint_pos = [float(word[2]), float(word[3]), float(word[4])]
                self.joint_dict[joint_name] = TreeNode(joint_name, joint_pos, None)
            elif word[0] == 'skin':
                skin_item = word[1:]
                self.joint_skin.append(skin_item)
        
        for line in lines:
            word = line.split()
            if word[0] == 'hier':
                parent_name, joint_name = word[1].strip(), word[2].strip()
                parent = self.joint_dict[parent_name]
                child = self.joint_dict[joint_name]
                child.parent = parent
                parent.children.append(child)
        
        for joint_name, joint in self.joint_dict.items():
            if joint.parent is None:
                self.roots.append(joint)

    def save(self, filename):
        with open(filename, 'w') as file_info:
            for joint_name, joint in self.joint_dict.items():
                file_info.write(
                    'joints {0} {1:.8f} {2:.8f} {3:.8f}\n'.format(joint_name, joint.pos[0], joint.pos[1], joint.pos[2]))
            
            for joint in self.roots:
                file_info.write(
                    'root {0}\n'.format(joint.name))
                
            for skw in self.joint_skin:
                cur_line = 'skin {0} '.format(skw[0])
                for cur_j in range(1, len(skw), 2):
                    cur_line += '{0} {1:.4f} '.format(skw[cur_j], float(skw[cur_j+1]))
                cur_line += '\n'
                file_info.write(cur_line)

            # for root in self.roots:
            #     this_level = root.children
            #     while this_level:
            #         next_level = []
            #         for p_node in this_level:
            #             file_info.write('hier {0} {1}\n'.format(p_node.parent.name, p_node.name))
            #             next_level += p_node.children
            #         this_level = next_level
            for joint_name, joint in self.joint_dict.items():
                if joint.parent is not None:
                    file_info.write("hier {0} {1}\n".format(joint.parent.name, joint.name))
                
    def get_nodes(self):
        return [node for node_name, node in self.joint_dict.items()]
    
    def get_node_names(self):
        return [node_name for node_name, node in self.joint_dict.items()]

    def create_connections(self):
        """
        attach children after all nodes' parent are set
        """
        for name, node in self.joint_dict.items():
            node.children.clear()

        for name, node in self.joint_dict.items():
            if node.parent is not None:
                node.parent.children.append(node)

    def create_template2auxilary(self):
        from datasets.joint_categories import categories
        self.template2auxilary = {} # primary_node_name, [secondary_node_name_0, ..., secondary_node_name_n]
        nodes = self.get_nodes()
        for node in nodes:
            if node.name not in categories:
                cur = node.parent
                while cur:  # find the first parent that is in the category by tracing the skeleton  hierarchy
                    if cur.name in categories:
                        break
                    cur = cur.parent

                if cur:
                    if self.template2auxilary.get(cur.name) is None:
                        self.template2auxilary[cur.name] = [node.name]
                    else:
                        self.template2auxilary[cur.name].append(node.name)

    def directional_adj_matrix(self):
        nodes = self.get_nodes()
        node_names = [node.name for node in nodes]
        num_joint = len(nodes)
        mat = np.zeros((num_joint, num_joint))
        for node in nodes:
            if node.parent:
                p_idx = node_names.index(node.parent.name)
                c_idx = node_names.index(node.name)
                mat[c_idx, p_idx] = 1
        return mat

    def sub_adj_matrix(self, joint_names):
        node_names = self.get_node_names()
        idxs = [node_names.index(j_name) for j_name in joint_names]
        M = self.directional_adj_matrix()
        m = len(idxs)
        subM = np.zeros((m, m))
        for i in range(m):
            I = idxs[i]
            for j in range(m):
                J = idxs[j]
                subM[i, j] = M[I, J]
        return subM
    
    def extract_weight_matrix(self, num_v):
        joint_names = self.get_node_names()
        weight = np.zeros((num_v, len(joint_names)))

        for idx in range(len(self.joint_skin)):
            i = int(self.joint_skin[idx][0])
            skin = self.joint_skin[idx]
            for b in range(1, len(skin), 2):
                joint_name = skin[b]
                skin_val = float(skin[b + 1])
                j = joint_names.index(joint_name)
                weight[i, j] = skin_val
        return weight
    
    def tpl_dist_matrix(self):
        parents = self.get_parent_ids()
        n_joint = len(parents)
        dist_mat = np.empty((n_joint, n_joint), dtype=np.int)
        dist_mat[:, :] = 100000
        for i, p in enumerate(parents):
            dist_mat[i, i] = 0
            if i != 0:
                dist_mat[i, p] = dist_mat[p, i] = 1

        """
        Floyd's algorithm
        """
        for k in range(n_joint):
            for i in range(n_joint):
                for j in range(n_joint):
                    dist_mat[i, j] = min(dist_mat[i, j], dist_mat[i, k] + dist_mat[k, j])

        return dist_mat

    def geo_dist_matrix(self):
        joint_pos = self.get_joint_pos()
        dist_mat = np.sqrt(np.sum((joint_pos[np.newaxis, ...] - joint_pos[:, np.newaxis, :])**2, axis=2))
        return dist_mat
    
    def get_parent_ids(self):
        nodes = self.get_nodes()
        node_names = [node.name for node in nodes]
        parents = []
        for node in nodes:
            if node.parent is None:
                parents.append(-1)
            else:
                parents.append(node_names.index(node.parent.name))
        parents = np.asarray(parents, dtype=np.int32)
        return parents
    
    def get_joint_pos(self):
        nodes = self.get_nodes()
        joint_pos = np.zeros((len(nodes), 3))
        for i, node in enumerate(nodes):
            joint_pos[i, :] = node.pos
        return joint_pos
    
    def get_jointpos_with_name(self):
        nodes = self.get_nodes()
        node_names = []
        joint_pos = np.zeros((len(nodes), 3))
        for i, node in enumerate(nodes):
            joint_pos[i, :] = node.pos
            node_names.append(node.name)
        return joint_pos, node_names

    def get_bones_by_names(self, names):
        nodes = self.get_nodes()
        B = []
        for node in nodes:
            if node.parent is not None:
                if node.parent.name in names and node.name in names:
                    p = node.parent.pos
                    q = node.pos
                    B.append([p[0], p[1], p[2], q[0], q[1], q[2]])
        B = np.asarray(B)
        return B
    
    def get_joint_by_names(self, names):
        nodes = self.get_nodes()
        node_names = [node.name for node in nodes]
        res = []
        for i in range(len(names)):
            res.append(nodes[node_names.index(names[i])])
        return res
    
    def normalize(self, scale, trans):
        for joint_name, joint in self.joint_dict.items():
            joint.pos = np.array(joint.pos)
            joint.pos /= scale
            joint.pos -= trans


def compute_bone_pts(rig_info, density=10):
    bone_pts = []
    nodes = rig_info.get_nodes()
    avg_dist = 0
    num_bones = 0

    for node in nodes:
        if node.parent:
            src_pos = np.array(node.parent.pos)
            tgt_pos = np.array(node.pos)
            avg_dist += np.linalg.norm(src_pos - tgt_pos)
            num_bones += 1

    avg_dist /= num_bones
    step = avg_dist / density
    for node in nodes:
        if node.parent:
            src_pos = np.array(node.parent.pos)
            tgt_pos = np.array(node.pos)
            bone_len = np.linalg.norm(src_pos - tgt_pos)
            num_segs = int(bone_len / step)

            bone_pts.append(src_pos[np.newaxis, :])
            if num_segs > 0:
                dir = tgt_pos - src_pos
                for i in range(1, num_segs):
                    t = 1.0 / num_segs
                    pos = src_pos + t * i * dir
                    bone_pts.append(pos[np.newaxis, :])

            if len(node.children) == 0:
                bone_pts.append(tgt_pos[np.newaxis, :])
    bone_pts = np.concatenate(bone_pts, axis=0)
    return bone_pts


def compute_bone_pts_from_bones(bones, density=10):
    bone_pts = []
    avg_dist = 0
    num_bones = bones.shape[0]

    for i in range(len(bones)):
        src_pos = bones[i, 0:3]
        tgt_pos = bones[i, 3:]
        avg_dist += np.linalg.norm(src_pos - tgt_pos)

    avg_dist /= num_bones
    step = avg_dist / density
    for i in range(len(bones)):
        src_pos = bones[i, 0:3]
        tgt_pos = bones[i, 3:]
        bone_len = np.linalg.norm(src_pos - tgt_pos)
        num_segs = int(bone_len / step)

        bone_pts.append(src_pos[np.newaxis, :])
        if num_segs > 0:
            dir = tgt_pos - src_pos
            for i in range(1, num_segs):
                t = 1.0 / num_segs
                pos = src_pos + t * i * dir
                bone_pts.append(pos[np.newaxis, :])

        bone_pts.append(tgt_pos[np.newaxis, :])

    bone_pts = np.concatenate(bone_pts, axis=0)
    return bone_pts