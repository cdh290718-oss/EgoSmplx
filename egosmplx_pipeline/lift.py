"""Preserve MANO projection through calibrated rays and radial distance."""
import numpy as np

def native_projection(points_camera, focal, center):
    normalized = points_camera[:, :2] / points_camera[:, 2:3].clip(1e-8)
    return focal * normalized + np.asarray(center, dtype=np.float64)


def lift_mano_to_fisheye(camera, wilor, candidate, body_wrist, body_wrist_uv):
    vertices_camera_native = (
        np.asarray(wilor["vertices_3d_local"][candidate], dtype=np.float64)
        + np.asarray(wilor["cam_t"][candidate], dtype=np.float64)[None]
    )
    joints_camera_native = (
        np.asarray(wilor["joints_3d_local"][candidate], dtype=np.float64)
        + np.asarray(wilor["cam_t"][candidate], dtype=np.float64)[None]
    )
    focal = float(wilor['focal_length_px'])
    center = np.asarray(wilor['image_size_wh'], dtype=np.float64) / 2
    vertex_uv = native_projection(vertices_camera_native, focal, center)
    joint_uv = native_projection(joints_camera_native, focal, center)
    # Keep the body wrist fixed and translate the complete hand projection.
    # This residual can be large when the body estimate is inaccurate.
    uv_shift = np.asarray(body_wrist_uv, dtype=np.float64) - joint_uv[0]
    vertex_uv += uv_shift[None]
    joint_uv += uv_shift[None]

    native_root_depth = float(np.linalg.norm(joints_camera_native[0]))
    body_root_depth = float(np.linalg.norm(body_wrist))
    vertex_relative_depth = np.linalg.norm(vertices_camera_native, axis=1) - native_root_depth
    joint_relative_depth = np.linalg.norm(joints_camera_native, axis=1) - native_root_depth
    vertex_depth = np.clip(body_root_depth + vertex_relative_depth, 0.05, None)
    joint_depth = np.clip(body_root_depth + joint_relative_depth, 0.05, None)
    vertices_fisheye = camera.camera2world(vertex_uv, vertex_depth)
    joints_fisheye = camera.camera2world(joint_uv, joint_depth)
    return {
        "vertices": vertices_fisheye.astype(np.float32),
        "joints": joints_fisheye.astype(np.float32),
        "vertex_uv": vertex_uv.astype(np.float32),
        "joint_uv": joint_uv.astype(np.float32),
        "uv_shift": uv_shift.astype(np.float32),
        "body_wrist_depth": body_root_depth,
        "native_wilor_root_depth": native_root_depth,
    }
