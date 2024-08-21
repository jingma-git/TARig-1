"""
jingma
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

def dijkstra_graph(verts, edges):
    N = len(verts)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)
    edges = list(edges)
    for i, j in edges:
        d = np.linalg.norm(verts[i, :] - verts[j, :])
        conn_matrix[i, j] = d
        conn_matrix[j, i] = d
    [geo_dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                                  return_predecessors=True, unweighted=False)
    return geo_dist, predecessors


def dijkstra_skel(joints):
    N = len(joints)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for i in range(len(joints)):
        joint = joints[i]
        if joint.parent is not None:
            j = joints.index(joint.parent)
            conn_matrix[i, j] = 1
            conn_matrix[j, i] = 1

    [geo_dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                                  return_predecessors=True, unweighted=False)
    return geo_dist