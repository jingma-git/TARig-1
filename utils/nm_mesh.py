import numpy as np
from utils.graph_utils import dijkstra_graph
"""
jingma
"""

class NonManifoldMesh:
    def __init__(self, o3d_mesh):
        o3d_mesh.compute_triangle_normals()
        self.mesh = o3d_mesh
        self.verts = np.asarray(o3d_mesh.vertices)
        self.tris = np.asarray(o3d_mesh.triangles)
        self.tri_normals = np.asarray(o3d_mesh.triangle_normals)
        self.edges = None

        self.v_faces = None
        self.v_edges = None
        self.edge_faces = {}
        self.edges, self.edge_faces, self.v_faces, self.v_edges, self.v_verts = self.build_topology(self.verts, self.tris)
        self.geo_dist, self.predecessors = None, None
        # print("done!")


    @staticmethod
    def build_topology(verts, tris):
        v_faces = [[] for i in range(len(verts))]
        v_edges = [[] for i in range(len(verts))]
        v_verts = [[] for i in range(len(verts))]

        edge_dict = {}
        for f_id in range(len(tris)):
            tri = tris[f_id]
            for i in range(len(tri)):
                j = (i+1) % 3
                u = min(tri[i], tri[j])
                v = max(tri[i], tri[j])
                edge = (u, v)
                if edge_dict.get(edge) == None:
                    edge_dict[edge] = [f_id]
                else:
                    edge_dict[edge].append(f_id)

                if f_id not in v_faces[u]:
                    v_faces[u].append(f_id)
                if f_id not in v_faces[v]:
                    v_faces[v].append(f_id)
                if edge not in v_edges[u]:
                    v_edges[u].append(edge)
                if edge not in v_edges[v]:
                    v_edges[v].append(edge)
                if v not in v_verts[u]:
                    v_verts[u].append(v)
                if u not in v_verts[v]:
                    v_verts[v].append(u)

        return edge_dict.keys(), edge_dict, v_faces, v_edges, v_verts


    def compute_geo_dist_among_verts(self):
        # geodesic distance
        [self.geo_dist, self.predecessors] = dijkstra_graph(self.verts, list(self.edges))

    def get_edges(self):
        edges = [list(edge) for edge in list(self.edges)]
        return np.asarray(edges)

    def get_sharp_edges(self, degree):
        res = []
        for edge, e_faces in self.edge_faces.items():
            if len(e_faces) == 2:
                i, j = e_faces[0], e_faces[1]
                nor_i, nor_j = self.tri_normals[i], self.tri_normals[j]
                dot_ij = np.clip(np.dot(nor_i, nor_j), -1, 1)
                angle = np.arccos(dot_ij)
                if angle > degree:
                    res.append(edge)
        return res


    def get_boundary_edges(self):
        res = []
        for edge, e_faces in self.edge_faces.items():
            if len(e_faces) == 1:
                res.append(edge)
        return res


    def create_dynamic_bones_from_src(self, v_id, flow, max_bones=50):
        dynamic_nodes = [] # joints/verts on dynamic bones
        v = v_id
        count = 0
        while True:
            dynamic_nodes.append(v)
            if count > max_bones:
                break

            min_u, min_angle = -1, 99999
            for u in self.v_verts[v]:
                dir_vu = self.verts[u, :] - self.verts[v, :]
                len_vu = np.linalg.norm(dir_vu)
                flow_u = flow[u, :]
                len_flow = np.linalg.norm(flow_u)
                if len_vu > 1e-8 and len_flow > 1e-8:
                    dir_vu = dir_vu / len_vu
                    flow_u = flow_u / len_flow
                    dot = np.clip(np.dot(dir_vu, flow_u), -1, 1)
                    angle = np.arccos(dot)
                    if angle < min_angle:
                        min_angle = angle
                        min_u = u


            if min_angle < np.pi / 3:
                v = min_u
            else:
                break
            count += 1
        return dynamic_nodes


    def create_dynamic_bones(self, start_vs, flow, max_bones=50):
        chains = []
        for v in start_vs:
            chain = self.create_dynamic_bones_from_src(v, flow)
            chains.append(chain)
        return chains


    def average_edge_length(self):
        tot_len = 0
        edges = list(self.edges)

        for i in range(len(edges)):
            u, v = edges[i]
            tot_len += np.sqrt(np.sum((self.verts[u, :] - self.verts[v, :]) ** 2))

        tot_len /= len(self.edges)
        return tot_len


    def extract_chains(self, chains, conflow):
        """
        assume each vertex correspond to one chain
        """
        flag = np.ones(len(self.verts), dtype=np.bool)

        for i in range(len(chains)):
            if len(chains[i]) <= 1:
                flag[i] = False
                continue

            if flag[i]==False:
                continue
            chain = chains[i]
            chain_set = set(chain)

            for v in chain[1:]:
                chain_v_set = set(chains[v])
                if chain_v_set <= chain_set:
                    flag[v] = False

        res_chains = []
        for i in range(len(chains)):
            if flag[i]:
                res_chains.append(chains[i])
        return res_chains

    def min_angle(self, chain):
        min = np.pi
        for i in range(1, len(chain) - 1):
            e10 = self.verts[chain[i - 1], :] - self.verts[chain[i], :]
            e10 /= np.linalg.norm(e10)
            e12 = self.verts[chain[i + 1],:] - self.verts[chain[i], :]
            e12 /= np.linalg.norm(e12)
            angle = np.arccos(np.dot(e10, e12))
            if angle < min:
                min = angle
        return min

    def extract_chains_return_idxs(self, chains, conflow):
        """
        assume each vertex correspond to one chain
        """
        flag = np.ones(len(self.verts), dtype=np.bool)
        # omit overlapping chains
        for i in range(len(chains)):
            if len(chains[i]) <= 1:
                flag[i] = False
                continue

            if flag[i]==False:
                continue
            chain = chains[i]
            chain_set = set(chain)

            for v in chain[1:]:
                chain_v_set = set(chains[v])
                if chain_v_set <= chain_set:
                    flag[v] = False
        # omit chain that has sharp angle
        chain_idxs = np.nonzero(flag)[0]
        # angle_thresh = np.pi * 2/3
        # for idx in chain_idxs:
        #     angle = self.min_angle(chains[idx])
        #     if len(chains[idx]) >= 3 and angle < angle_thresh:
        #         flag[idx] = False
        # chain_idxs = np.nonzero(flag)[0]
        return chain_idxs

    def average_edge_length_in_region(self, verts):
        edges = {}
        for v in verts:
            for edge in self.v_edges[v]:
                edges[edge] = 1
        edges = list(edges.keys())

        tot_len = 0
        for i in range(len(edges)):
            u, v = edges[i]
            tot_len += np.sqrt(np.sum((self.verts[u, :] - self.verts[v, :]) ** 2))
        tot_len /= len(edges)
        return tot_len
    def compute_chain_length(self, chain):
        tot_len = 0

        for i in range(0, len(chain)-1):
            tot_len += np.sqrt(np.sum((self.verts[chain[i+1]] - self.verts[chain[i]]) ** 2))

        return tot_len

    def compute_rigid_binding_by_geodesic_distance(self, joint_pos):
        verts = self.verts
        N = len(verts)
        num_joints = len(joint_pos)
        joint2verts_dist = np.linalg.norm(joint_pos[:,None,:] - verts[None, :, :], axis=2)
        sorted_idxs = np.argsort(joint2verts_dist, axis=1)
        joint_vertices = [sorted_idxs[j, 0:6]for j in range(num_joints)]

        if self.geo_dist is None and self.predecessors is None:
            self.compute_geo_dist_among_verts()

        dist = self.geo_dist
        rigid_bind = np.zeros(N, np.int)
        for i in range(N):
            j_tpl, min_geo = -1, np.inf
            for j in range(num_joints):
                for att_v in joint_vertices[j]:
                    if dist[i, att_v]!=-9999 and dist[i, att_v] < min_geo:
                        min_geo = dist[i, att_v]
                        j_tpl = j
            min_j = j_tpl
            if j_tpl == -1:
                min_j = np.argmin(joint2verts_dist[:, i])
            rigid_bind[i] = min_j
        return rigid_bind

    def compute_connection_cost(self, nodes_pos, nodes_vs, boneflow):
        """
        boneflow: normalized boneflow direction, ToDO...
        """
        N = len(nodes_pos)
        angle = np.ones((N, N)) * np.inf
        dist = np.ones((N, N)) * np.inf
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pij = nodes_pos[j, :] - nodes_pos[i, :]
                pij_len = np.linalg.norm(pij)
                dist[i, j] = pij_len
                # pij /= pij_len
                vs_i = nodes_vs[i]
                vs_j = nodes_vs[j]
                if len(vs_i) == 0:
                    continue

                flow_diff = 0
                for v in vs_i:
                    flow_diff += (1 - np.dot(pij, boneflow[v, :]))
                flow_diff /= len(vs_i)
                angle[i, j] = flow_diff

                # if flow_diff < 0.5:
                #     angle[i, j] = flow_diff
                #
                # connected_in_tpl = False
                # min_geo_dist = np.inf
                # for vi in vs_i:
                #     for vj in vs_j:
                #         if self.geo_dist[vi, vj]!=-9999 and self.geo_dist[vi, vj] < min_geo_dist:
                #             connected_in_tpl = True
                #             min_geo_dist = self.geo_dist[vi, vj]

                # if connected_in_tpl:
                #     dist[i, j] = min_geo_dist
                # else:
                #     dist[i, j] = pij_len

        return angle, dist

if __name__ == "__main__":
    import open3d as o3d
    import os

    data_dir = "F:/Dataset/AutoRS"
    model_id = 14765
    mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, "obj_remesh/{}.obj".format(model_id)))
    nm_mesh = NonManifoldMesh(mesh)
    sharp_edges = nm_mesh.get_sharp_edges(degree=np.pi * 2 / 3)
    print("sharp_edges", len(sharp_edges))