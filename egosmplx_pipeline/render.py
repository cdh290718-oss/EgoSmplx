"""Clipped fisheye mesh overlays and Wavefront OBJ export."""
import numpy as np
import cv2

def read_obj(path):
    vertices, faces = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(v.split("/")[0]) - 1 for v in line.split()[1:4]])
    return np.asarray(vertices, np.float64), np.asarray(faces, np.int32)


def render(image, vertices, faces, camera, max_edge_fraction):
    uv = camera.world2camera(vertices)
    axes = np.asarray(getattr(camera, "camera_axis_transform", np.eye(3)), np.float64)
    camera_vertices = vertices @ axes.T
    valid = (
        np.isfinite(camera_vertices).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (camera_vertices[:, 2] > 1e-5)
    )
    height, width = image.shape[:2]
    max_edge = max(40.0, max_edge_fraction * np.hypot(width, height))
    overlay = image.copy()
    for tri in faces:
        if not valid[tri].all():
            continue
        points = uv[tri]
        if (
            points[:, 0].max() < 0 or points[:, 0].min() >= width
            or points[:, 1].max() < 0 or points[:, 1].min() >= height
        ):
            continue
        edges = points[[1, 2, 0]] - points
        if np.linalg.norm(edges, axis=1).max() > max_edge:
            continue
        cv2.polylines(
            overlay, [np.rint(points).astype(np.int32)], True,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return cv2.addWeighted(image, 0.22, overlay, 0.78, 0)

def write_obj(path, vertices, faces):
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
