"""Mesh boundary and breadth-first ring distances."""
from collections import Counter, deque
import numpy as np

def boundary_edges(faces):
    counts = Counter(
        tuple(sorted((int(first), int(second))))
        for face in np.asarray(faces, dtype=np.int64)
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    )
    return {edge for edge, count in counts.items() if count == 1}


def graph_distance_from_boundary(faces, boundary_vertices, vertex_count):
    adjacency = [set() for _ in range(vertex_count)]
    for face in np.asarray(faces, dtype=np.int64):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            adjacency[int(first)].add(int(second))
            adjacency[int(second)].add(int(first))
    sentinel = np.iinfo(np.int32).max
    distance = np.full(vertex_count, sentinel, dtype=np.int32)
    queue = deque()
    for vertex in boundary_vertices:
        distance[int(vertex)] = 0
        queue.append(int(vertex))
    while queue:
        first = queue.popleft()
        for second in adjacency[first]:
            if distance[second] > distance[first] + 1:
                distance[second] = distance[first] + 1
                queue.append(second)
    if np.any(distance == sentinel):
        raise RuntimeError("MANO topology contains vertices disconnected from wrist boundary")
    return distance
