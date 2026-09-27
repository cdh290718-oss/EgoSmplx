"""Joint mappings, bounded pose increments, and rigid rotation math. No annotation reader."""
import numpy as np
import torch

JOINT_MAPPING = {
    "nose": "Nose",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
}


BODY_POSE_NAMES = (
    "L_Hip", "R_Hip", "Spine_1", "L_Knee", "R_Knee", "Spine_2",
    "L_Ankle", "R_Ankle", "Spine_3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)


MAX_DELTA_DEG = {
    "L_Hip": 18.0, "R_Hip": 18.0,
    "Spine_1": 10.0, "Spine_2": 10.0, "Spine_3": 12.0,
    "L_Knee": 22.0, "R_Knee": 22.0,
    "L_Ankle": 14.0, "R_Ankle": 14.0,
    "Neck": 12.0, "Head": 12.0,
    "L_Collar": 15.0, "R_Collar": 15.0,
    "L_Shoulder": 28.0, "R_Shoulder": 28.0,
    "L_Elbow": 25.0, "R_Elbow": 25.0,
    "L_Wrist": 15.0, "R_Wrist": 15.0,
}


def save_obj(vertices, faces, path):
    with path.open("w", encoding="utf-8") as output:
        for vertex in vertices:
            output.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            output.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

def axis_angle_rotation(axis_angle):
    x, y, z = axis_angle.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    theta = torch.linalg.norm(axis_angle)
    first = torch.sinc(theta / np.pi)
    second = 0.5 * torch.sinc(theta / (2.0 * np.pi)) ** 2
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return eye + first * skew + second * (skew @ skew)
