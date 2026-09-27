#!/usr/bin/env python3
"""Run WiLoR hand detection/reconstruction on a configurable image manifest."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.camera import cam_crop_to_full
from wilor.utils.yolo_loader import load_yolo_detector


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--detector", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.20)
    ap.add_argument("--rescale-factor", type=float, default=2.0)
    return ap.parse_args()


def project(points, translation, focal_length, image_size_wh):
    width, height = map(float, image_size_wh)
    xyz = points + translation[None]
    uv = xyz[:, :2] / np.maximum(xyz[:, 2:3], 1e-6)
    uv *= float(focal_length)
    uv += np.asarray([width / 2.0, height / 2.0], np.float32)
    return uv


def write_obj(path, vertices, faces):
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for name in ("npz", "json", "mesh", "overlay"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.manifest.read_text())['frames']
    images = [Path(r['image']) for r in rows]
    if len({p.stem for p in images}) != len(images):
        raise ValueError('WiLoR input stems must be unique; runner stages images by frame id')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_wilor(str(args.checkpoint), str(args.config))
    detector = load_yolo_detector(str(args.detector))
    model = model.to(device).eval()
    detector = detector.to(device)
    faces = np.asarray(model.mano.faces, np.int32)
    report = {"method": "WiLoR detector + MANO regression", "frames": []}

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        result = detector(image, conf=args.confidence, verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        right = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
        frame = {"image": str(image_path), "detections": []}
        overlay = image.copy()
        focal_value = float(model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(image.shape[:2]))
        if len(boxes) == 0:
            np.savez_compressed(args.output_dir / 'npz' / f'{image_path.stem}_wilor.npz',
                                detector_confidence=np.empty(0,np.float32), is_right=np.empty(0,np.float32),
                                joints_2d=np.empty((0,21,2),np.float32), mano_faces=faces,
                                image_size_wh=np.array(image.shape[1::-1]), focal_length_px=np.array(focal_value),
                                image_sha256=np.array(__import__('hashlib').sha256(image_path.read_bytes()).hexdigest()))
            cv2.imwrite(str(args.output_dir / 'overlay' / f'{image_path.stem}_wilor.jpg'), overlay)
            report['frames'].append(frame)
            continue

        dataset = ViTDetDataset(model_cfg, image, boxes, right, rescale_factor=args.rescale_factor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        arrays = {key: [] for key in (
            "bbox_xyxy", "detector_confidence", "is_right", "global_orient_aa",
            "hand_pose_aa", "betas", "cam_t", "joints_3d_local", "joints_2d",
            "vertices_3d_local",
        )}
        offset = 0
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                output = model(batch)
            pred_mano = output["pred_mano_params"]
            pred_cam = output["pred_cam"].clone()
            pred_cam[:, 1] *= 2 * batch["right"] - 1
            image_size = batch["img_size"].float()
            focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * image_size.max()
            translations = cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                image_size, focal,
            ).detach().cpu().numpy()
            vertices_batch = output["pred_vertices"].detach().cpu().numpy()
            joints_batch = output["pred_keypoints_3d"].detach().cpu().numpy()
            hand_rot = pred_mano["hand_pose"].detach().cpu().numpy()
            global_rot = pred_mano["global_orient"].detach().cpu().numpy()
            betas = pred_mano["betas"].detach().cpu().numpy()
            right_batch = batch["right"].detach().cpu().numpy()
            size_batch = image_size.detach().cpu().numpy()
            for index in range(len(vertices_batch)):
                source_index = offset + index
                is_right = bool(right_batch[index] > 0.5)
                vertices = vertices_batch[index].copy()
                joints = joints_batch[index].copy()
                mirror = 1.0 if is_right else -1.0
                vertices[:, 0] *= mirror
                joints[:, 0] *= mirror
                uv = project(joints, translations[index], float(focal), size_batch[index])
                global_aa = Rotation.from_matrix(global_rot[index, 0]).as_rotvec().astype(np.float32)
                hand_aa = Rotation.from_matrix(hand_rot[index]).as_rotvec().astype(np.float32)
                arrays["bbox_xyxy"].append(boxes[source_index])
                arrays["detector_confidence"].append(scores[source_index])
                arrays["is_right"].append(float(is_right))
                arrays["global_orient_aa"].append(global_aa)
                arrays["hand_pose_aa"].append(hand_aa)
                arrays["betas"].append(betas[index])
                arrays["cam_t"].append(translations[index])
                arrays["joints_3d_local"].append(joints)
                arrays["joints_2d"].append(uv)
                arrays["vertices_3d_local"].append(vertices)
                rec = {
                    "index": source_index,
                    "is_right": is_right,
                    "confidence": float(scores[source_index]),
                    "bbox_xyxy": boxes[source_index].tolist(),
                    "joints_2d": uv.tolist(),
                    "global_orient_aa": global_aa.tolist(),
                    "hand_pose_aa": hand_aa.tolist(),
                    "betas": betas[index].tolist(),
                    "cam_t": translations[index].tolist(),
                }
                frame["detections"].append(rec)
                color = (0, 220, 255) if is_right else (255, 180, 0)
                p1 = tuple(np.rint(boxes[source_index, :2]).astype(int))
                p2 = tuple(np.rint(boxes[source_index, 2:]).astype(int))
                cv2.rectangle(overlay, p1, p2, color, 2)
                cv2.putText(overlay, f"{'R' if is_right else 'L'} {scores[source_index]:.2f}", p1,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                for point in uv:
                    if np.isfinite(point).all():
                        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, color, -1, cv2.LINE_AA)
                write_obj(
                    args.output_dir / "mesh" / f"{image_path.stem}_hand{source_index}_{'R' if is_right else 'L'}.obj",
                    vertices + translations[index][None], faces,
                )
            offset += len(vertices_batch)

        np.savez_compressed(
            args.output_dir / "npz" / f"{image_path.stem}_wilor.npz",
            **{key: np.asarray(value) for key, value in arrays.items()},
            mano_faces=faces,
            image_size_wh=np.array(image.shape[1::-1]), focal_length_px=np.array(focal_value),
            image_sha256=np.array(__import__('hashlib').sha256(image_path.read_bytes()).hexdigest()),
        )
        (args.output_dir / "json" / f"{image_path.stem}_wilor.json").write_text(
            json.dumps(frame, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        cv2.imwrite(str(args.output_dir / "overlay" / f"{image_path.stem}_wilor.jpg"), overlay)
        report["frames"].append(frame)
        print(image_path.name, len(frame["detections"]))

    (args.output_dir / "wilor_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
