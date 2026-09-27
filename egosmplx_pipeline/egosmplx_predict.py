#!/usr/bin/env python
"""Run person detection, EgoSMPLX inference, and fisheye projection on DownwardStereo extracted frames."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

REPO_ROOT = Path(os.environ["EGOSMPLX_REPO_ROOT"]).resolve()
from egosmplx_pipeline import camera as calibrated_camera
sys.modules["FishEyeCalibrated"] = calibrated_camera
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "main"))
sys.path.insert(0, str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "common"))

from config import cfg
from utils.inference_utils import process_mmdet_results, non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "DownwardStereo" / "left_frames_stride10")
    parser.add_argument("--image-glob", default="*.jpg")
    parser.add_argument("--fisheye-calibration", type=Path, default=REPO_ROOT / "DownwardStereo" / "fisheye.left.calibration.json")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "main" / "config" / "config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "output" / "train_5datasets_4gpu_resume_s7_to_s12_20260801_191616" / "model_dump" / "snapshot_11.pth.tar",
    )
    parser.add_argument(
        "--det-config",
        type=Path,
        default=REPO_ROOT / "pretrained_models" / "mmdet" / "mmdet_faster_rcnn_r50_fpn_coco.py",
    )
    parser.add_argument(
        "--det-checkpoint",
        type=Path,
        default=REPO_ROOT / "pretrained_models" / "mmdet" / "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--bbox-thr", type=float, default=50.0)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--multi-person", action="store_true")
    parser.add_argument("--no-detector", action="store_true", help="Run one full-image SMPL-X prediction per image without Faster R-CNN.")
    parser.add_argument("--save-mesh", action="store_true")
    parser.add_argument(
        "--save-wireframe-overlay",
        action="store_true",
        help="Also save a wireframe overlay from the same prediction and camera projection.",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show-verts", action="store_true")
    parser.add_argument("--show-bbox", action="store_true", default=True)
    parser.add_argument("--mesh-mode", choices=("solid", "wireframe", "points"), default="solid")
    parser.add_argument(
        "--smplx-input-mode",
        choices=(
            "full_image",
            "det_crop",
            "det_tangent",
            "det_undistort_contiguous",
        ),
        default="full_image",
        help=(
            "det_tangent uses the detected person box to place the 16x12 ViT "
            "tangent patches in original fisheye-camera coordinates; "
            "det_undistort_contiguous first undistorts the full frame, detects "
            "there, and samples one continuous 12-column x 16-row ViT crop."
        ),
    )
    parser.add_argument(
        "--det-tangent-expand",
        type=float,
        default=1.2,
        help="Isotropic expansion applied to the detector box in det_tangent mode.",
    )
    parser.add_argument(
        "--det-tangent-patch-size",
        type=float,
        default=0.1,
        help="Half-size of each tangent-plane patch in det_tangent mode.",
    )
    parser.add_argument(
        "--global-undistort-alpha",
        type=float,
        default=1.0,
        help=(
            "OpenCV optimal-new-camera alpha for det_undistort_contiguous: "
            "0 crops invalid borders, 1 retains the full calibrated field."
        ),
    )
    parser.add_argument(
        "--contiguous-roi-mode",
        choices=("stretch", "aspect_pad"),
        default="stretch",
        help=(
            "For det_undistort_contiguous, stretch the clipped expanded box "
            "to the 12:16 token canvas, or preserve 12:16 geometry with "
            "out-of-image padding."
        ),
    )
    parser.add_argument(
        "--input-crop-mode",
        choices=("none", "center_square", "training_horizontal"),
        default="none",
        help="Crop the network input without changing the original image used for fisheye mesh overlay.",
    )
    parser.add_argument(
        "--horizontal-crop-pixels",
        type=int,
        default=128,
        help="Pixels removed from both sides with --input-crop-mode training_horizontal.",
    )
    parser.add_argument("--contact-sheet-limit", type=int, default=40)
    parser.add_argument(
        "--refine-camera-from-positionnet",
        action="store_true",
        help=(
            "Keep the predicted SMPL-X local pose/shape fixed and refine a "
            "camera extrinsic rotation/translation against PositionNet upper-body joints."
        ),
    )
    parser.add_argument("--camera-fit-iters", type=int, default=300)
    parser.add_argument("--camera-fit-lr", type=float, default=0.03)
    parser.add_argument(
        "--camera-fit-mode",
        choices=("translation", "translation_global_orient", "extrinsics"),
        default="translation",
        help=(
            "With --refine-camera-from-positionnet, optimize tx,ty,tz only; "
            "SMPL-X global_orient plus translation; or the legacy camera-frame "
            "rotation plus translation."
        ),
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    return parser.parse_args()


def make_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "full" if args.max_images is None else f"{args.max_images}imgs"
    return REPO_ROOT / "output" / f"downwardstereo_left_stride10_detect_s11_{suffix}_{stamp}"


def setup_model(args, output_dir):
    os.environ.setdefault("EGOSMPLX_REPO_ROOT", str(REPO_ROOT))
    os.environ["EGOSMPLX_FISHEYE_CALIBRATION"] = str(args.fisheye_calibration.resolve())
    cfg.get_config_fromfile(str(args.config_path.resolve()))
    cfg.update_test_config(
        testset="EHF",
        agora_benchmark="na",
        shapy_eval_split=None,
        pretrained_model_path=str(args.checkpoint_path.resolve()),
        use_cache=False,
    )
    cfg.fisheye_camera_path = str(args.fisheye_calibration.resolve())
    cfg.fisheye_camera_paths = [str(args.fisheye_calibration.resolve())]
    cfg.update_config(args.num_gpus, str(output_dir))

    from base import Demoer

    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    return demoer


def load_runtime_utils():
    from FishEyeCalibrated import FishEyeCameraCalibrated
    from utils.human_models import smpl_x
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.vis import render_mesh_fisheye, save_obj

    return FishEyeCameraCalibrated, smpl_x, load_img, process_bbox, generate_patch_image, render_mesh_fisheye, save_obj


def filter_bboxes(raw_bboxes, score_thr, bbox_thr):
    kept = []
    for bbox in raw_bboxes:
        x1, y1, x2, y2, score = [float(v) for v in bbox[:5]]
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if score < score_thr:
            continue
        if w < bbox_thr or h < bbox_thr:
            continue
        kept.append(np.array([x1, y1, x2, y2, score], dtype=np.float32))
    return kept


def save_contact_sheet(image_paths, out_path, max_images=40, thumb_w=360):
    paths = image_paths[:max_images]
    if not paths:
        return
    thumbs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = max(1, int(h * thumb_w / max(1, w)))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    max_h = max(t.shape[0] for t in thumbs)
    canvas = np.full((rows * max_h, cols * thumb_w, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas[r * max_h : r * max_h + thumb.shape[0], c * thumb_w : c * thumb_w + thumb.shape[1]] = thumb
    cv2.imwrite(str(out_path), canvas)


def draw_no_detection(img_rgb, text="NO PERSON DETECTED"):
    img = img_rgb.copy()
    cv2.putText(img, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 40, 40), 3, cv2.LINE_AA)
    return img


def rgb_to_bgr_u8(img_rgb):
    img = np.clip(img_rgb, 0, 255).astype(np.uint8)
    return img[:, :, ::-1]


def expand_bbox_xyxy(box_xyxy, scale, image_w, image_h):
    """Expand a detector box about its center and clip it to the source image."""
    x1, y1, x2, y2 = [float(value) for value in box_xyxy[:4]]
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    width = max(2.0, abs(x2 - x1) * float(scale))
    height = max(2.0, abs(y2 - y1) * float(scale))
    x1 = np.clip(center_x - width * 0.5, 0.0, float(image_w - 1))
    x2 = np.clip(center_x + width * 0.5, 0.0, float(image_w - 1))
    y1 = np.clip(center_y - height * 0.5, 0.0, float(image_h - 1))
    y2 = np.clip(center_y + height * 0.5, 0.0, float(image_h - 1))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        raise ValueError(f"Expanded detector box is degenerate: {(x1, y1, x2, y2)}")
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def create_global_undistort_model(calibration_path, image_w, image_h, alpha):
    """Build one continuous Brown-Conrady rectification for the full frame."""
    calibration = json.loads(Path(calibration_path).read_text())
    required = (
        "sourceFocalLength",
        "sourcePrincipalPoint",
        "sourceRadialDistortion",
    )
    missing = [key for key in required if key not in calibration]
    if missing:
        raise KeyError(
            "Continuous global undistortion requires Brown-Conrady source "
            f"fields in {calibration_path}; missing {missing}"
        )
    calibration_size = tuple(int(value) for value in calibration.get("size", (image_w, image_h)))
    if calibration_size != (int(image_w), int(image_h)):
        raise ValueError(
            "Calibration/image size mismatch for global undistortion: "
            f"calibration={calibration_size}, image={(image_w, image_h)}"
        )
    focal = np.asarray(calibration["sourceFocalLength"], dtype=np.float64)
    principal = np.asarray(calibration["sourcePrincipalPoint"], dtype=np.float64)
    radial = list(calibration["sourceRadialDistortion"])
    tangential = list(calibration.get("sourceTangentialDistortion", (0.0, 0.0)))
    while len(radial) < 3:
        radial.append(0.0)
    source_k = np.asarray(
        [
            [focal[0], 0.0, principal[0]],
            [0.0, focal[1], principal[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [radial[0], radial[1], tangential[0], tangential[1], radial[2]],
        dtype=np.float64,
    )
    new_k, valid_roi = cv2.getOptimalNewCameraMatrix(
        source_k,
        distortion,
        (int(image_w), int(image_h)),
        float(alpha),
        (int(image_w), int(image_h)),
        centerPrincipalPoint=False,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_k,
        distortion,
        None,
        new_k,
        (int(image_w), int(image_h)),
        cv2.CV_32FC1,
    )
    return {
        "source_k": source_k,
        "distortion": distortion,
        "new_k": new_k,
        "valid_roi": tuple(int(value) for value in valid_roi),
        "map_x": map_x,
        "map_y": map_y,
        "alpha": float(alpha),
    }


def undistort_full_image(image_rgb, undistort_model):
    """Remap a complete fisheye frame into one continuous pinhole image."""
    return cv2.remap(
        image_rgb,
        undistort_model["map_x"],
        undistort_model["map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def undistorted_pixels_to_original(points_xy, undistort_model):
    """Map pixels from the continuous undistorted view to the raw frame."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    new_k = undistort_model["new_k"]
    normalized = np.column_stack(
        (
            (points[:, 0] - new_k[0, 2]) / new_k[0, 0],
            (points[:, 1] - new_k[1, 2]) / new_k[1, 1],
            np.ones((len(points),), dtype=np.float64),
        )
    )
    projected, _ = cv2.projectPoints(
        normalized,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        undistort_model["source_k"],
        undistort_model["distortion"],
    )
    return projected.reshape(-1, 2).astype(np.float32)


def expand_bbox_to_aspect(box_xyxy, scale, target_width_over_height):
    """Expand a box and pad it to one aspect ratio without clipping."""
    x1, y1, x2, y2 = [float(value) for value in box_xyxy[:4]]
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    width = max(2.0, abs(x2 - x1) * float(scale))
    height = max(2.0, abs(y2 - y1) * float(scale))
    if width / height > float(target_width_over_height):
        height = width / float(target_width_over_height)
    else:
        width = height * float(target_width_over_height)
    return np.asarray(
        [
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ],
        dtype=np.float32,
    )


def create_contiguous_bbox_pixel_grid(
    bbox_xyxy,
    patch_rows,
    patch_columns,
    patch_h,
    patch_w,
):
    """Split one continuous rectified ROI into row-major ViT patch grids."""
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    output_h = int(patch_rows) * int(patch_h)
    output_w = int(patch_columns) * int(patch_w)
    xs = np.linspace(x1, x2, output_w, dtype=np.float32)
    ys = np.linspace(y1, y2, output_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    full_grid = np.stack((grid_x, grid_y), axis=-1)
    return (
        full_grid.reshape(
            patch_rows,
            patch_h,
            patch_columns,
            patch_w,
            2,
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(patch_rows * patch_columns, patch_h, patch_w, 2)
        .astype(np.float32)
    )


def draw_contiguous_sampling_grid(image_rgb, grid_pixels, bbox_xyxy):
    """Draw the regular 12-column x 16-row crop on the undistorted image."""
    canvas = image_rgb.copy()
    for patch in grid_pixels:
        corners = np.asarray(
            [patch[0, 0], patch[0, -1], patch[-1, -1], patch[-1, 0]],
            dtype=np.float32,
        )
        cv2.polylines(
            canvas,
            [np.rint(corners).astype(np.int32)],
            isClosed=True,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    x1, y1, x2, y2 = np.rint(bbox_xyxy).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(
        canvas,
        "continuous undistorted ROI -> 12 cols x 16 rows",
        (max(10, x1), max(30, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def create_bbox_tangent_pixel_grid(
    sampler,
    fisheye_camera,
    bbox_xyxy,
    patch_rows=16,
    patch_columns=12,
    patch_size=0.1,
):
    """Create row-major tangent-patch samples in original fisheye pixels."""
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    horizontal = (np.arange(patch_columns, dtype=np.float32) + 0.5) / patch_columns
    vertical = (np.arange(patch_rows, dtype=np.float32) + 0.5) / patch_rows
    normalized_centers = np.dstack(np.meshgrid(horizontal, vertical)).reshape(-1, 2)
    centers_2d = normalized_centers.copy()
    centers_2d[:, 0] = x1 + centers_2d[:, 0] * (x2 - x1)
    centers_2d[:, 1] = y1 + centers_2d[:, 1] * (y2 - y1)
    centers_2d_up = centers_2d.copy()
    centers_2d_up[:, 1] -= (y2 - y1) / (2.0 * patch_rows)

    depth = np.ones((len(centers_2d),), dtype=np.float32)
    centers_3d = fisheye_camera.camera2world(centers_2d, depth)
    centers_3d_up = fisheye_camera.camera2world(centers_2d_up, depth)
    patch_h, patch_w = [
        int(value) for value in sampler.patches_2d_per_camera_pixels.shape[-3:-1]
    ]
    tangent_patches = [
        sampler._generate_patch_coordinates(
            center,
            center_up,
            (float(patch_size), float(patch_size)),
            (patch_h, patch_w),
            visualize=False,
        )
        for center, center_up in zip(centers_3d, centers_3d_up)
    ]
    grid = sampler._project_patches_to_original_image(
        tangent_patches,
        fisheye_camera=fisheye_camera,
        image_h=int(fisheye_camera.img_size[1]),
        image_w=int(fisheye_camera.img_size[0]),
        normalize=False,
    )
    return np.asarray(grid, dtype=np.float32)


def draw_tangent_sampling_grid(image_rgb, grid_pixels, bbox_xyxy):
    """Draw every actual tangent-patch footprint and its detected ROI."""
    canvas = image_rgb.copy()
    for patch in grid_pixels:
        corners = np.asarray(
            [patch[0, 0], patch[0, -1], patch[-1, -1], patch[-1, 0]],
            dtype=np.float32,
        )
        if np.isfinite(corners).all():
            cv2.polylines(
                canvas,
                [np.rint(corners).astype(np.int32)],
                True,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    x1, y1, x2, y2 = np.rint(bbox_xyxy).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(
        canvas,
        "det ROI -> 16x12 tangent patches",
        (max(10, x1), max(30, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_vit_patch_atlas(
    sampled_patches,
    out_path,
    rows=16,
    columns=12,
    scale=4,
    gap=2,
    grid_color_rgb=None,
):
    """Save the exact RGB patches returned by UndistortPatch before PatchEmbed."""
    patches = sampled_patches.detach().cpu().numpy()[0]
    if patches.shape[0] != rows * columns:
        raise ValueError(
            f"Expected {rows * columns} ViT patches, got {patches.shape[0]}"
        )
    patches = np.clip(patches.transpose(0, 2, 3, 1) * 255.0, 0, 255).astype(np.uint8)
    patch_h, patch_w = patches.shape[1:3]
    atlas_h = rows * patch_h + (rows - 1) * gap
    atlas_w = columns * patch_w + (columns - 1) * gap
    atlas = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)
    for patch_index, patch in enumerate(patches):
        row, column = divmod(patch_index, columns)
        top = row * (patch_h + gap)
        left = column * (patch_w + gap)
        atlas[top:top + patch_h, left:left + patch_w] = patch
    atlas = cv2.resize(
        atlas,
        (atlas.shape[1] * scale, atlas.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    if grid_color_rgb is not None:
        patch_h_scaled = patch_h * scale
        patch_w_scaled = patch_w * scale
        for row in range(1, rows):
            y = row * patch_h_scaled
            cv2.line(
                atlas,
                (0, y),
                (atlas.shape[1] - 1, y),
                grid_color_rgb,
                2,
                cv2.LINE_AA,
            )
        for column in range(1, columns):
            x = column * patch_w_scaled
            cv2.line(
                atlas,
                (x, 0),
                (x, atlas.shape[0] - 1),
                grid_color_rgb,
                2,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(out_path), rgb_to_bgr_u8(atlas))


POSITIONNET_BODY_EDGES = (
    (0, 1), (1, 3), (3, 5),
    (0, 2), (2, 4), (4, 6),
    (0, 7),
    (7, 8), (8, 10), (10, 12),
    (7, 9), (9, 11), (11, 13),
    (5, 14), (5, 15), (5, 16),
    (6, 17), (6, 18), (6, 19),
    (7, 24), (24, 22), (22, 20), (24, 23), (23, 21),
)


def positionnet_coords_to_sampling_grid(
    body_joint_img,
    sampling_grid_pixels,
    patch_rows=16,
    patch_columns=12,
):
    """Map PositionNet coordinates through the actual tangent-patch centers."""
    coords = np.asarray(body_joint_img, dtype=np.float32)
    grid = np.asarray(sampling_grid_pixels, dtype=np.float32)
    if grid.shape[0] != patch_rows * patch_columns:
        raise ValueError(
            f"Expected {patch_rows * patch_columns} sampling patches, got {grid.shape[0]}"
        )
    patch_h, patch_w = grid.shape[1:3]
    center_rows = slice((patch_h - 1) // 2, patch_h // 2 + 1)
    center_columns = slice((patch_w - 1) // 2, patch_w // 2 + 1)
    patch_centers = grid.reshape(
        patch_rows, patch_columns, patch_h, patch_w, 2
    )[:, :, center_rows, center_columns].mean(axis=(2, 3))

    x = np.clip(coords[:, 0], 0.0, float(patch_columns - 1))
    y = np.clip(coords[:, 1], 0.0, float(patch_rows - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, patch_columns - 1)
    y1 = np.minimum(y0 + 1, patch_rows - 1)
    wx = (x - x0).reshape(-1, 1)
    wy = (y - y0).reshape(-1, 1)
    pixels = (
        (1.0 - wx) * (1.0 - wy) * patch_centers[y0, x0]
        + wx * (1.0 - wy) * patch_centers[y0, x1]
        + (1.0 - wx) * wy * patch_centers[y1, x0]
        + wx * wy * patch_centers[y1, x1]
    )
    return pixels.astype(np.float32), patch_centers.astype(np.float32)


def draw_positionnet_result(image_rgb, body_joint_hm, body_joint_pixels, sampling_bbox_xyxy):
    """Overlay the depth-marginalized PositionNet heatmap and body skeleton."""
    canvas = image_rgb.copy()
    heatmap_2d = np.asarray(body_joint_hm, dtype=np.float32).sum(axis=(0, 1))
    heatmap_2d -= heatmap_2d.min()
    peak = float(heatmap_2d.max())
    if peak > 0:
        heatmap_2d /= peak
    heatmap_color_bgr = cv2.applyColorMap(
        np.rint(heatmap_2d * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap_color = heatmap_color_bgr[:, :, ::-1]

    image_h, image_w = canvas.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in sampling_bbox_xyxy]
    x1, x2 = np.clip([x1, x2], 0, image_w)
    y1, y2 = np.clip([y1, y2], 0, image_h)
    if x2 > x1 and y2 > y1:
        heatmap_roi = cv2.resize(
            heatmap_color,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_CUBIC,
        ).astype(canvas.dtype, copy=False)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(
            canvas[y1:y2, x1:x2],
            0.68,
            heatmap_roi,
            0.32,
            0.0,
        )

    for start, end in POSITIONNET_BODY_EDGES:
        points = body_joint_pixels[[start, end]]
        if np.isfinite(points).all():
            cv2.line(
                canvas,
                tuple(np.rint(points[0]).astype(int)),
                tuple(np.rint(points[1]).astype(int)),
                (255, 255, 0),
                4,
                cv2.LINE_AA,
            )
    for point in body_joint_pixels:
        if np.isfinite(point).all():
            cv2.circle(
                canvas,
                tuple(np.rint(point).astype(int)),
                6,
                (255, 0, 0),
                -1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        "PositionNet: 25 body joints (yellow) + heatmap",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_positionnet_smplx_2d_comparison(
    image_rgb,
    positionnet_pixels,
    smplx_pixels,
):
    """Draw all 25 corresponding body joints from PositionNet and SMPL-X."""
    canvas = image_rgb.copy()
    positionnet_pixels = np.asarray(positionnet_pixels, dtype=np.float32)
    smplx_pixels = np.asarray(smplx_pixels, dtype=np.float32)
    if positionnet_pixels.shape != (25, 2) or smplx_pixels.shape != (25, 2):
        raise ValueError(
            "PositionNet and SMPL-X body points must both have shape (25, 2)"
        )

    valid = np.isfinite(positionnet_pixels).all(axis=1) & np.isfinite(smplx_pixels).all(axis=1)
    valid &= (np.abs(positionnet_pixels) < 1e6).all(axis=1)
    valid &= (np.abs(smplx_pixels) < 1e6).all(axis=1)
    errors = np.linalg.norm(smplx_pixels - positionnet_pixels, axis=1)

    # Draw per-joint correspondence first so the two skeletons remain visible.
    for joint_index in np.flatnonzero(valid):
        positionnet_point = tuple(np.rint(positionnet_pixels[joint_index]).astype(int))
        smplx_point = tuple(np.rint(smplx_pixels[joint_index]).astype(int))
        cv2.line(
            canvas,
            positionnet_point,
            smplx_point,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for start, end in POSITIONNET_BODY_EDGES:
        if valid[start] and valid[end]:
            cv2.line(
                canvas,
                tuple(np.rint(positionnet_pixels[start]).astype(int)),
                tuple(np.rint(positionnet_pixels[end]).astype(int)),
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                tuple(np.rint(smplx_pixels[start]).astype(int)),
                tuple(np.rint(smplx_pixels[end]).astype(int)),
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

    for joint_index in np.flatnonzero(valid):
        positionnet_point = tuple(np.rint(positionnet_pixels[joint_index]).astype(int))
        smplx_point = tuple(np.rint(smplx_pixels[joint_index]).astype(int))
        cv2.circle(canvas, positionnet_point, 6, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, smplx_point, 7, (0, 255, 0), 2, cv2.LINE_AA)
        midpoint = tuple(
            np.rint((positionnet_pixels[joint_index] + smplx_pixels[joint_index]) * 0.5)
            .astype(int)
        )
        cv2.putText(
            canvas,
            str(joint_index),
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    selected_valid = valid[CAMERA_FIT_BODY_INDICES]
    selected_errors = errors[CAMERA_FIT_BODY_INDICES][selected_valid]
    all_mean_error = float(errors[valid].mean()) if valid.any() else float("nan")
    fit_mean_error = (
        float(selected_errors.mean()) if len(selected_errors) else float("nan")
    )
    labels = (
        ((255, 0, 0), "red: PositionNet 2D (filled)"),
        ((0, 255, 0), "green: SMPL-X projected 2D (ring)"),
        ((0, 255, 255), "cyan: corresponding joint"),
        ((255, 255, 255), f"mean all25={all_mean_error:.1f}px, fit12={fit_mean_error:.1f}px"),
    )
    for row, (color, label) in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (24, 38 + row * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


CAMERA_FIT_BODY_INDICES = np.asarray(
    [7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24], dtype=np.int64
)
CAMERA_FIT_BODY_EDGES = (
    (7, 8), (7, 9), (8, 10), (10, 12), (9, 11), (11, 13),
    (7, 20), (7, 21), (20, 22), (21, 23), (22, 24), (23, 24),
)


def heatmap_joints_to_original_pixels(joint_img, input_hw, input_offset):
    """Invert the normalization used by Model.get_coord for full-image input."""
    input_h, input_w = [float(v) for v in input_hw]
    offset_x, offset_y = [float(v) for v in input_offset]
    joint_px = np.asarray(joint_img, dtype=np.float64).copy()
    joint_px[:, 0] = joint_px[:, 0] / float(cfg.output_hm_shape[2]) * input_w + offset_x
    joint_px[:, 1] = joint_px[:, 1] / float(cfg.output_hm_shape[1]) * input_h + offset_y
    return joint_px


def axis_angle_to_matrix(axis_angle):
    """Differentiable Rodrigues transform for a single three-vector."""
    theta = torch.linalg.vector_norm(axis_angle).clamp_min(1e-8)
    axis = axis_angle / theta
    x, y, z = axis.unbind()
    zero = torch.zeros((), device=axis.device, dtype=axis.dtype)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero)
    ).reshape(3, 3)
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return identity + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


def compose_global_orient(delta_rotation, network_global_orient):
    """Left-compose a camera-frame delta into the SMPL-X root axis-angle."""
    delta_np = delta_rotation.detach().cpu().numpy().astype(np.float64)
    network_np = (
        network_global_orient.detach().cpu().numpy().reshape(3).astype(np.float64)
    )
    network_rotation, _ = cv2.Rodrigues(network_np)
    refined_rotation = delta_np @ network_rotation
    refined_axis_angle, _ = cv2.Rodrigues(refined_rotation)
    return torch.as_tensor(
        refined_axis_angle.reshape(3),
        device=network_global_orient.device,
        dtype=network_global_orient.dtype,
    )


def refine_camera_extrinsics(
    body_joints_cam,
    body_targets_px,
    initial_translation,
    fisheye_cam,
    iterations,
    learning_rate,
):
    """Fit a camera-frame delta R,t while local pose, shape, and expression stay fixed."""
    if iterations <= 0:
        raise ValueError("--camera-fit-iters must be positive")
    if learning_rate <= 0:
        raise ValueError("--camera-fit-lr must be positive")

    device = body_joints_cam.device
    dtype = body_joints_cam.dtype
    indices = torch.as_tensor(CAMERA_FIT_BODY_INDICES, device=device, dtype=torch.long)
    joints = body_joints_cam.index_select(0, indices).detach()
    targets = torch.as_tensor(
        body_targets_px[CAMERA_FIT_BODY_INDICES], device=device, dtype=dtype
    )
    initial = initial_translation.detach().to(device=device, dtype=dtype)

    identity = torch.eye(3, device=device, dtype=dtype)

    def project(rotation, translation):
        transformed = joints @ rotation.t() + translation[None, :]
        return fisheye_cam.world2camera_pytorch(transformed)

    with torch.no_grad():
        initial_projection = project(identity, initial)
        initial_error = torch.linalg.vector_norm(
            initial_projection - targets, dim=1
        ).mean()

    initial_z = max(0.1, float(initial[2].item()))
    depth_starts = sorted(set([initial_z, 0.5, 0.75, 1.0, 1.5, 2.0]))
    best = None
    for depth_start in depth_starts:
        parameter = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                float(initial[0]),
                float(initial[1]),
                float(np.log(depth_start)),
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([parameter], lr=learning_rate)
        for _ in range(iterations):
            rotation = axis_angle_to_matrix(parameter[:3])
            translation = torch.stack(
                (parameter[3], parameter[4], torch.exp(parameter[5]))
            )
            projection = project(rotation, translation)
            # Charbonnier-style robust pixel loss. Dividing by 100 keeps the
            # Adam step scale stable while preserving the minimizer.
            residual = (projection - targets) / 100.0
            loss = torch.sqrt(residual.square() + 1e-4).mean()
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                rotation_norm = torch.linalg.vector_norm(parameter[:3])
                if rotation_norm > np.pi:
                    parameter[:3].mul_(float(np.pi / rotation_norm))
                parameter[3].clamp_(-3.0, 3.0)
                parameter[4].clamp_(-3.0, 3.0)
                parameter[5].clamp_(float(np.log(0.1)), float(np.log(5.0)))

        with torch.no_grad():
            rotation = axis_angle_to_matrix(parameter[:3])
            translation = torch.stack(
                (parameter[3], parameter[4], torch.exp(parameter[5]))
            )
            projection = project(rotation, translation)
            error = torch.linalg.vector_norm(projection - targets, dim=1).mean()
            candidate = (
                float(error),
                rotation.detach().clone(),
                translation.detach().clone(),
                projection.detach().clone(),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise RuntimeError("Camera translation refinement produced no finite solution")
    return {
        "rotation": best[1],
        "translation": best[2],
        "initial_projection": initial_projection.detach(),
        "refined_projection": best[3],
        "initial_mean_error_px": float(initial_error),
        "refined_mean_error_px": float(best[0]),
        "joint_indices": CAMERA_FIT_BODY_INDICES.copy(),
    }


def refine_camera_translation(
    body_joints_local,
    body_targets_px,
    initial_translation,
    fisheye_cam,
    iterations,
    learning_rate,
):
    """Fit tx, ty, and positive tz while keeping pose and rotation fixed."""
    if iterations <= 0:
        raise ValueError("--camera-fit-iters must be positive")
    if learning_rate <= 0:
        raise ValueError("--camera-fit-lr must be positive")

    device = body_joints_local.device
    dtype = body_joints_local.dtype
    indices = torch.as_tensor(CAMERA_FIT_BODY_INDICES, device=device, dtype=torch.long)
    joints = body_joints_local.index_select(0, indices).detach()
    targets = torch.as_tensor(
        body_targets_px[CAMERA_FIT_BODY_INDICES], device=device, dtype=dtype
    )
    initial = initial_translation.detach().to(device=device, dtype=dtype)
    identity = torch.eye(3, device=device, dtype=dtype)

    def project(translation):
        return fisheye_cam.world2camera_pytorch(joints + translation[None, :])

    with torch.no_grad():
        initial_projection = project(initial)
        initial_error = torch.linalg.vector_norm(
            initial_projection - targets, dim=1
        ).mean()

    initial_z = max(0.1, float(initial[2].item()))
    depth_starts = sorted(set([initial_z, 0.5, 0.75, 1.0, 1.5, 2.0]))
    best = None
    for depth_start in depth_starts:
        parameter = torch.tensor(
            [
                float(initial[0]),
                float(initial[1]),
                float(np.log(depth_start)),
            ],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([parameter], lr=learning_rate)
        for _ in range(iterations):
            translation = torch.stack(
                (parameter[0], parameter[1], torch.exp(parameter[2]))
            )
            projection = project(translation)
            residual = (projection - targets) / 100.0
            loss = torch.sqrt(residual.square() + 1e-4).mean()
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                parameter[0].clamp_(-3.0, 3.0)
                parameter[1].clamp_(-3.0, 3.0)
                parameter[2].clamp_(float(np.log(0.1)), float(np.log(5.0)))

        with torch.no_grad():
            translation = torch.stack(
                (parameter[0], parameter[1], torch.exp(parameter[2]))
            )
            projection = project(translation)
            error = torch.linalg.vector_norm(projection - targets, dim=1).mean()
            candidate = (
                float(error),
                translation.detach().clone(),
                projection.detach().clone(),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise RuntimeError("Camera translation refinement produced no finite solution")
    return {
        "rotation": identity,
        "translation": best[1],
        "initial_projection": initial_projection.detach(),
        "refined_projection": best[2],
        "initial_mean_error_px": float(initial_error),
        "refined_mean_error_px": float(best[0]),
        "joint_indices": CAMERA_FIT_BODY_INDICES.copy(),
    }


def draw_camera_fit_debug(
    image_rgb,
    target_body_px,
    initial_body_px,
    refined_body_px,
    initial_error,
    refined_error,
):
    canvas = image_rgb.copy()
    series = (
        (target_body_px, (255, 0, 0), "PositionNet"),
        (initial_body_px, (0, 0, 255), "network cam"),
        (refined_body_px, (0, 255, 0), "refined cam"),
    )
    for points, color, _ in series:
        for start, end in CAMERA_FIT_BODY_EDGES:
            if np.isfinite(points[[start, end]]).all():
                cv2.line(
                    canvas,
                    tuple(np.rint(points[start]).astype(int)),
                    tuple(np.rint(points[end]).astype(int)),
                    color,
                    4,
                    cv2.LINE_AA,
                )
        for index in CAMERA_FIT_BODY_INDICES:
            if np.isfinite(points[index]).all():
                cv2.circle(
                    canvas,
                    tuple(np.rint(points[index]).astype(int)),
                    8,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
    labels = (
        ((255, 0, 0), "red: PositionNet target"),
        ((0, 0, 255), f"blue: network cam ({initial_error:.1f}px)"),
        ((0, 255, 0), f"green: refined cam ({refined_error:.1f}px)"),
    )
    for row, (color, label) in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (30, 45 + row * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            3,
            cv2.LINE_AA,
        )
    return canvas


def main():
    args = parse_args()
    paths_to_check = [args.image_dir, args.fisheye_calibration, args.config_path, args.checkpoint_path]
    if not args.no_detector:
        paths_to_check.extend([args.det_config, args.det_checkpoint])
    for path in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.no_detector and args.smplx_input_mode != "full_image":
        raise ValueError("--no-detector requires --smplx-input-mode full_image")
    if (
        args.refine_camera_from_positionnet
        and args.smplx_input_mode
        not in ("full_image", "det_tangent", "det_undistort_contiguous")
    ):
        raise ValueError(
            "Camera refinement requires --smplx-input-mode full_image, "
            "det_tangent, or det_undistort_contiguous"
        )
    if args.horizontal_crop_pixels < 0:
        raise ValueError("--horizontal-crop-pixels must be non-negative")
    if args.det_tangent_expand <= 0:
        raise ValueError("--det-tangent-expand must be positive")
    if args.det_tangent_patch_size <= 0:
        raise ValueError("--det-tangent-patch-size must be positive")
    if not 0.0 <= args.global_undistort_alpha <= 1.0:
        raise ValueError("--global-undistort-alpha must be in [0, 1]")
    if (
        args.smplx_input_mode in ("det_tangent", "det_undistort_contiguous")
        and args.input_crop_mode != "none"
    ):
        raise ValueError(
            f"{args.smplx_input_mode} requires --input-crop-mode none"
        )


    image_paths = sorted(args.image_dir.glob(args.image_glob))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir} with glob {args.image_glob}")

    output_dir = make_output_dir(args)
    overlay_dir = output_dir / "img"
    wireframe_dir = output_dir / "wireframe"
    smplx_dir = output_dir / "smplx"
    meta_dir = output_dir / "meta"
    det_dir = output_dir / "detections"
    debug_dir = output_dir / "projection_debug"
    mesh_dir = output_dir / "mesh"
    vit_patch_dir = output_dir / "vit_patches"
    positionnet_dir = output_dir / "positionnet"
    undistorted_dir = output_dir / "undistorted"
    undistorted_detection_dir = output_dir / "undistorted_detections"
    joint_compare_dir = output_dir / "joint_compare"
    mesh_joint_compare_dir = output_dir / "mesh_joint_compare"
    for path in (overlay_dir, smplx_dir, meta_dir, det_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.smplx_input_mode in ("det_tangent", "det_undistort_contiguous"):
        vit_patch_dir.mkdir(parents=True, exist_ok=True)
        positionnet_dir.mkdir(parents=True, exist_ok=True)
    if args.smplx_input_mode == "det_undistort_contiguous":
        undistorted_dir.mkdir(parents=True, exist_ok=True)
        undistorted_detection_dir.mkdir(parents=True, exist_ok=True)
    joint_compare_dir.mkdir(parents=True, exist_ok=True)
    mesh_joint_compare_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mesh:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    if args.save_wireframe_overlay:
        wireframe_dir.mkdir(parents=True, exist_ok=True)

    demoer = setup_model(args, output_dir)
    FishEyeCameraCalibrated, smpl_x, load_img, process_bbox, generate_patch_image, render_mesh_fisheye, save_obj = load_runtime_utils()
    fisheye_cam = FishEyeCameraCalibrated(str(args.fisheye_calibration.resolve()))
    positionnet_capture = {}
    if args.smplx_input_mode in ("det_tangent", "det_undistort_contiguous"):
        model_core = demoer.model.module if hasattr(demoer.model, "module") else demoer.model

        def capture_positionnet_output(_module, _inputs, outputs):
            positionnet_capture["body_joint_hm"] = outputs[0].detach().cpu()
            positionnet_capture["body_joint_img"] = outputs[1].detach().cpu()

        model_core.body_position_net.register_forward_hook(capture_positionnet_output)
    detector = None
    if not args.no_detector:
        detector = init_detector(str(args.det_config.resolve()), str(args.det_checkpoint.resolve()), device="cuda:0")
    transform = transforms.ToTensor()

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read {image_paths[0]}")
    img_h0, img_w0 = first.shape[:2]
    undistort_model = None
    if args.smplx_input_mode == "det_undistort_contiguous":
        undistort_model = create_global_undistort_model(
            args.fisheye_calibration,
            img_w0,
            img_h0,
            args.global_undistort_alpha,
        )
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_dir / "detected_mesh_overlay_stride10.mp4"), fourcc, 10.0, (img_w0, img_h0))

    saved_images = []
    saved_wireframe_images = []
    saved_sampling_grid_images = []
    saved_patch_atlas_images = []
    saved_positionnet_images = []
    saved_camera_fit_images = []
    saved_joint_compare_images = []
    saved_mesh_joint_compare_images = []
    saved_undistorted_images = []
    saved_undistorted_detection_images = []
    saved_contiguous_patch_grid_images = []
    records = []
    projection_consistency_errors = []
    total_persons = 0
    no_detection = 0
    start_time = time.time()

    progress_desc = "Full-image project" if args.no_detector else "Detect + project"
    with tqdm(image_paths, desc=progress_desc, mininterval=10.0) as pbar:
        for image_path in pbar:
            original_img = load_img(str(image_path))
            vis_img = original_img.copy()
            wireframe_img = original_img.copy() if args.save_wireframe_overlay else None
            img_h, img_w = original_img.shape[:2]
            detector_img = original_img
            if args.smplx_input_mode == "det_undistort_contiguous":
                if (img_w, img_h) != (img_w0, img_h0):
                    raise ValueError(
                        "All globally undistorted frames must share one size; "
                        f"expected {(img_w0, img_h0)}, got {(img_w, img_h)}"
                    )
                detector_img = undistort_full_image(original_img, undistort_model)
                undistorted_out = undistorted_dir / f"{image_path.stem}_undistorted.jpg"
                cv2.imwrite(str(undistorted_out), rgb_to_bgr_u8(detector_img))
                saved_undistorted_images.append(undistorted_out)

            if args.no_detector:
                raw_boxes = []
                det_boxes = [np.array([0.0, 0.0, float(img_w - 1), float(img_h - 1), 1.0], dtype=np.float32)]
            else:
                if args.smplx_input_mode == "det_undistort_contiguous":
                    detector_input = np.ascontiguousarray(rgb_to_bgr_u8(detector_img))
                else:
                    detector_input = str(image_path)
                mmdet_results = inference_detector(detector, detector_input)
                raw_boxes = process_mmdet_results(mmdet_results, cat_id=0, multi_person=True)[0]
                filtered = filter_bboxes(raw_boxes, args.score_thr, args.bbox_thr)
                if args.multi_person:
                    det_boxes = non_max_suppression(filtered, args.iou_thr) if filtered else []
                else:
                    det_boxes = filtered[:1]

            if args.smplx_input_mode == "det_undistort_contiguous":
                detector_vis = detector_img.copy()
                for selected_box in det_boxes:
                    dx1, dy1, dx2, dy2, dscore = [
                        float(value) for value in selected_box[:5]
                    ]
                    cv2.rectangle(
                        detector_vis,
                        (int(round(dx1)), int(round(dy1))),
                        (int(round(dx2)), int(round(dy2))),
                        (255, 0, 0),
                        3,
                    )
                    cv2.putText(
                        detector_vis,
                        f"person {dscore:.2f}",
                        (int(round(dx1)), max(25, int(round(dy1)) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                undistorted_detection_out = (
                    undistorted_detection_dir
                    / f"{image_path.stem}_undistorted_detection.jpg"
                )
                cv2.imwrite(
                    str(undistorted_detection_out),
                    rgb_to_bgr_u8(detector_vis),
                )
                saved_undistorted_detection_images.append(
                    undistorted_detection_out
                )

            det_record = {
                "detector_enabled": not args.no_detector,
                "image_path": str(image_path.resolve()),
                "raw_person_detections": int(len(raw_boxes)),
                "score_thr": float(args.score_thr),
                "bbox_thr": float(args.bbox_thr),
                "detector_coordinate_space": (
                    "global_undistorted"
                    if args.smplx_input_mode == "det_undistort_contiguous"
                    else "original_fisheye"
                ),
                "selected_bboxes_xyxy_score": [np.asarray(b).astype(float).tolist() for b in det_boxes],
            }
            (det_dir / f"{image_path.stem}_detections.json").write_text(json.dumps(det_record, indent=2) + "\n")

            if not det_boxes:
                no_detection += 1
                vis_img = draw_no_detection(vis_img)
                out_img = overlay_dir / f"{image_path.stem}_det_mesh_overlay.jpg"
                cv2.imwrite(str(out_img), rgb_to_bgr_u8(vis_img))
                saved_images.append(out_img)
                if wireframe_img is not None:
                    wireframe_out = wireframe_dir / f"{image_path.stem}_det_mesh_wireframe.jpg"
                    cv2.imwrite(str(wireframe_out), rgb_to_bgr_u8(wireframe_img))
                    saved_wireframe_images.append(wireframe_out)
                if video_writer is not None:
                    video_writer.write(rgb_to_bgr_u8(vis_img))
                records.append({**det_record, "persons_projected": 0})
                continue

            persons_projected = 0
            for bbox_id, det_box in enumerate(det_boxes):
                x1, y1, x2, y2, score = [float(v) for v in det_box[:5]]
                mmdet_box_xywh = np.array([x1, y1, abs(x2 - x1), abs(y2 - y1)], dtype=np.float32)
                sampling_bbox_xyxy = None
                sampling_grid_pixels = None
                if args.smplx_input_mode == "det_tangent":
                    sampling_bbox_xyxy = expand_bbox_xyxy(
                        det_box,
                        args.det_tangent_expand,
                        img_w,
                        img_h,
                    )
                    sx1, sy1, sx2, sy2 = sampling_bbox_xyxy
                    bbox = np.asarray([sx1, sy1, sx2 - sx1, sy2 - sy1], dtype=np.float32)
                elif args.smplx_input_mode == "det_undistort_contiguous":
                    if args.contiguous_roi_mode == "stretch":
                        sampling_bbox_xyxy = expand_bbox_xyxy(
                            det_box,
                            args.det_tangent_expand,
                            img_w,
                            img_h,
                        )
                    else:
                        sampling_bbox_xyxy = expand_bbox_to_aspect(
                            det_box,
                            args.det_tangent_expand,
                            target_width_over_height=12.0 / 16.0,
                        )
                    sx1, sy1, sx2, sy2 = sampling_bbox_xyxy
                    bbox = np.asarray(
                        [sx1, sy1, sx2 - sx1, sy2 - sy1],
                        dtype=np.float32,
                    )
                else:
                    bbox = process_bbox(mmdet_box_xywh, img_w, img_h)
                if bbox is None:
                    continue

                if args.smplx_input_mode == "det_crop":
                    img_crop, img2bb_trans, bb2img_trans = generate_patch_image(
                        original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
                    )
                    img = transform(img_crop.astype(np.float32)) / 255.0
                    img = img.cuda()[None, :, :, :]
                    meta_info = {}
                    # FishEyeCameraCalibrated.world2camera already returns
                    # original fisheye-image pixel coordinates. Applying
                    # bb2img_trans here double-applies the crop inverse and
                    # shifts/scales the mesh overlay.
                    render_proj_trans = None
                else:
                    img2bb_trans = np.eye(2, 3, dtype=np.float32)
                    bb2img_trans = np.eye(2, 3, dtype=np.float32)
                    input_img = (
                        detector_img
                        if args.smplx_input_mode == "det_undistort_contiguous"
                        else original_img
                    )
                    input_offset = np.array([0.0, 0.0], dtype=np.float32)
                    input_crop_xyxy = [0, 0, int(img_w), int(img_h)]
                    if args.input_crop_mode == "center_square":
                        side = min(img_h, img_w)
                        crop_left = int(round((img_w - side) / 2.0))
                        crop_top = int(round((img_h - side) / 2.0))
                        input_img = original_img[crop_top:crop_top + side, crop_left:crop_left + side, :]
                        input_offset = np.array([crop_left, crop_top], dtype=np.float32)
                        input_crop_xyxy = [crop_left, crop_top, crop_left + side, crop_top + side]
                    elif args.input_crop_mode == "training_horizontal":
                        crop_left = int(args.horizontal_crop_pixels)
                        crop_right = int(args.horizontal_crop_pixels)
                        if crop_left + crop_right >= img_w:
                            raise ValueError(
                                "Horizontal crop removes the entire image: "
                                f"width={img_w}, left={crop_left}, right={crop_right}"
                            )
                        input_img = original_img[:, crop_left:img_w - crop_right, :]
                        input_offset = np.array([crop_left, 0.0], dtype=np.float32)
                        input_crop_xyxy = [crop_left, 0, img_w - crop_right, img_h]
                    input_h, input_w = input_img.shape[:2]
                    img = transform(input_img.astype(np.float32)) / 255.0
                    img = img.cuda()[None, :, :, :]
                    meta_info = {
                        "camera_id": torch.zeros((1,), dtype=torch.long, device=img.device),
                        "input_img_shape": torch.tensor([[input_h, input_w]], dtype=torch.float32, device=img.device),
                        "input_coord_offset": torch.from_numpy(input_offset[None]).to(device=img.device, dtype=torch.float32),
                    }
                    render_proj_trans = None
                    if args.smplx_input_mode in (
                        "det_tangent",
                        "det_undistort_contiguous",
                    ):
                        model_core = demoer.model.module if hasattr(demoer.model, "module") else demoer.model
                        sampler = model_core.encoder.fisheye2sphere
                        patch_rows = int(sampler.patch_num_vertical)
                        patch_columns = int(sampler.selected_patch_indices.numel() // patch_rows)
                        if patch_rows * patch_columns != 192:
                            raise ValueError(
                                "Detection-aligned sampling expects the checkpoint's "
                                "192-token 16-row x 12-column ViT layout; "
                                f"got rows={patch_rows}, columns={patch_columns}"
                            )
                        if args.smplx_input_mode == "det_tangent":
                            sampling_grid_pixels = create_bbox_tangent_pixel_grid(
                                sampler,
                                fisheye_cam,
                                sampling_bbox_xyxy,
                                patch_rows=patch_rows,
                                patch_columns=patch_columns,
                                patch_size=args.det_tangent_patch_size,
                            )
                        else:
                            patch_h, patch_w = [
                                int(value)
                                for value in sampler.patches_2d_per_camera_pixels.shape[-3:-1]
                            ]
                            sampling_grid_pixels = create_contiguous_bbox_pixel_grid(
                                sampling_bbox_xyxy,
                                patch_rows=patch_rows,
                                patch_columns=patch_columns,
                                patch_h=patch_h,
                                patch_w=patch_w,
                            )
                        sampler.patches_2d_per_camera_pixels = torch.from_numpy(
                            sampling_grid_pixels[None]
                        ).to(device=img.device, dtype=img.dtype)
                        sampler.selected_patch_indices = torch.arange(
                            sampling_grid_pixels.shape[0],
                            device=img.device,
                            dtype=torch.long,
                        )
                        with torch.no_grad():
                            sampled_vit_patches = sampler(
                                img,
                                camera_ids=meta_info["camera_id"],
                                input_img_shapes=meta_info["input_img_shape"],
                                input_coord_offsets=meta_info["input_coord_offset"],
                            )
                        patch_prefix = f"{image_path.stem}_{bbox_id:02d}"
                        np.save(
                            vit_patch_dir / f"{patch_prefix}_sampling_grid_pixels.npy",
                            sampling_grid_pixels,
                        )
                        if args.smplx_input_mode == "det_tangent":
                            sampling_overlay = draw_tangent_sampling_grid(
                                original_img,
                                sampling_grid_pixels,
                                sampling_bbox_xyxy,
                            )
                        else:
                            sampling_overlay = draw_contiguous_sampling_grid(
                                detector_img,
                                sampling_grid_pixels,
                                sampling_bbox_xyxy,
                            )
                        sampling_grid_out = vit_patch_dir / f"{patch_prefix}_sampling_grid.jpg"
                        patch_atlas_out = vit_patch_dir / f"{patch_prefix}_patch_atlas.jpg"
                        cv2.imwrite(str(sampling_grid_out), rgb_to_bgr_u8(sampling_overlay))
                        if args.smplx_input_mode == "det_tangent":
                            save_vit_patch_atlas(
                                sampled_vit_patches,
                                patch_atlas_out,
                                rows=patch_rows,
                                columns=patch_columns,
                            )
                        else:
                            save_vit_patch_atlas(
                                sampled_vit_patches,
                                patch_atlas_out,
                                rows=patch_rows,
                                columns=patch_columns,
                                gap=0,
                            )
                            patch_grid_out = (
                                vit_patch_dir
                                / f"{patch_prefix}_patch_atlas_grid.jpg"
                            )
                            save_vit_patch_atlas(
                                sampled_vit_patches,
                                patch_grid_out,
                                rows=patch_rows,
                                columns=patch_columns,
                                gap=0,
                                grid_color_rgb=(255, 0, 0),
                            )
                            saved_contiguous_patch_grid_images.append(
                                patch_grid_out
                            )
                        saved_sampling_grid_images.append(sampling_grid_out)
                        saved_patch_atlas_images.append(patch_atlas_out)
                positionnet_capture.clear()
                with torch.no_grad():
                    out = demoer.model({"img": img, "img_ori": img}, {}, meta_info, "test")
                body_joint_pixels_undistorted = None
                if args.smplx_input_mode in (
                    "det_tangent",
                    "det_undistort_contiguous",
                ):
                    if not {"body_joint_hm", "body_joint_img"}.issubset(positionnet_capture):
                        raise RuntimeError("PositionNet forward hook did not capture its outputs")
                    body_joint_hm = positionnet_capture["body_joint_hm"].numpy()[0]
                    body_joint_img = positionnet_capture["body_joint_img"].numpy()[0]
                    body_joint_pixels_sampling, patch_center_grid_pixels = (
                        positionnet_coords_to_sampling_grid(
                            body_joint_img,
                            sampling_grid_pixels,
                            patch_rows=patch_rows,
                            patch_columns=patch_columns,
                        )
                    )
                    if args.smplx_input_mode == "det_undistort_contiguous":
                        body_joint_pixels_undistorted = body_joint_pixels_sampling
                        body_joint_pixels = undistorted_pixels_to_original(
                            body_joint_pixels_undistorted,
                            undistort_model,
                        )
                        positionnet_canvas = detector_img
                        positionnet_canvas_points = body_joint_pixels_undistorted
                    else:
                        body_joint_pixels = body_joint_pixels_sampling
                        positionnet_canvas = original_img
                        positionnet_canvas_points = body_joint_pixels
                    positionnet_prefix = f"{image_path.stem}_{bbox_id:02d}"
                    positionnet_payload = {
                        "body_joint_hm": body_joint_hm.astype(np.float32),
                        "body_joint_img": body_joint_img.astype(np.float32),
                        "body_joint_xy_original_from_grid": body_joint_pixels.astype(np.float32),
                        "patch_center_grid_pixels": patch_center_grid_pixels.astype(np.float32),
                        "body_joint_names": np.asarray(smpl_x.pos_joints_name[:25]),
                        "sampling_bbox_xyxy": sampling_bbox_xyxy.astype(np.float32),
                        "output_hm_shape": np.asarray(cfg.output_hm_shape, dtype=np.int32),
                    }
                    if body_joint_pixels_undistorted is not None:
                        positionnet_payload[
                            "body_joint_xy_global_undistorted"
                        ] = body_joint_pixels_undistorted.astype(np.float32)
                        positionnet_payload[
                            "global_undistorted_new_k"
                        ] = undistort_model["new_k"].astype(np.float64)
                    np.savez_compressed(
                        positionnet_dir / f"{positionnet_prefix}_positionnet.npz",
                        **positionnet_payload,
                    )
                    positionnet_vis = draw_positionnet_result(
                        positionnet_canvas,
                        body_joint_hm,
                        positionnet_canvas_points,
                        sampling_bbox_xyxy,
                    )
                    positionnet_vis_out = positionnet_dir / f"{positionnet_prefix}_positionnet.jpg"
                    cv2.imwrite(str(positionnet_vis_out), rgb_to_bgr_u8(positionnet_vis))
                    saved_positionnet_images.append(positionnet_vis_out)
                mesh = out["smplx_mesh_cam"].detach().cpu().numpy()[0]
                network_translation = out["cam_trans"].detach()[0]
                render_translation = network_translation
                network_global_orient = out["smplx_root_pose"].detach()[0]
                render_global_orient = network_global_orient.clone()
                identity_rotation = torch.eye(
                    3,
                    device=network_global_orient.device,
                    dtype=network_global_orient.dtype,
                )
                global_orient_delta_matrix = identity_rotation.clone()
                camera_rotation_delta = identity_rotation.clone()
                global_orient_delta_degrees = 0.0
                global_orient_mesh_consistency_max_error_m = None
                global_orient_stage_mean_error_px = None
                body_targets_px = None
                body_projection_initial_px = None
                body_projection_refined_px = None
                body_projection_model_px = None
                projection_consistency_max_error_px = None
                camera_fit = None

                if args.smplx_input_mode in (
                    "full_image",
                    "det_tangent",
                    "det_undistort_contiguous",
                ):
                    # smplx_joint_cam_pred is root-relative and is intended for
                    # 3D evaluation. Rebuild the absolute local joints from the
                    # predicted SMPL-X parameters before adding cam_trans; these
                    # are the joints used by get_coord to create joint_proj.
                    zero_eye_pose = torch.zeros_like(out["smplx_jaw_pose"])
                    with torch.no_grad():
                        model_core = demoer.model.module if hasattr(demoer.model, "module") else demoer.model
                        reconstructed_smplx = model_core.smplx_layer(
                            betas=out["smplx_shape"],
                            body_pose=out["smplx_body_pose"],
                            global_orient=out["smplx_root_pose"],
                            right_hand_pose=out["smplx_rhand_pose"],
                            left_hand_pose=out["smplx_lhand_pose"],
                            jaw_pose=out["smplx_jaw_pose"],
                            leye_pose=zero_eye_pose,
                            reye_pose=zero_eye_pose,
                            expression=out["smplx_expr"],
                        )
                    body_joints_local = reconstructed_smplx.joints[0, smpl_x.joint_idx[:25], :]
                    body_joints_local_render = body_joints_local
                    if args.smplx_input_mode in (
                        "det_tangent",
                        "det_undistort_contiguous",
                    ):
                        body_targets_px = body_joint_pixels
                    else:
                        joint_img = out["joint_img"].detach().cpu().numpy()[0]
                        body_targets_px = heatmap_joints_to_original_pixels(
                            joint_img[:25, :2], (input_h, input_w), input_offset
                        )
                    with torch.no_grad():
                        body_projection_initial = fisheye_cam.world2camera_pytorch(
                            body_joints_local + network_translation[None, :]
                        )
                    body_projection_initial_px = body_projection_initial.cpu().numpy()
                    body_projection_refined_px = body_projection_initial_px.copy()
                    body_projection_model_px = heatmap_joints_to_original_pixels(
                        out["smplx_joint_proj"].detach().cpu().numpy()[0, :25, :2],
                        (input_h, input_w),
                        input_offset,
                    )
                    projection_consistency_max_error_px = float(
                        np.linalg.norm(
                            body_projection_model_px - body_projection_initial_px,
                            axis=1,
                        ).max()
                    )
                    projection_consistency_errors.append(projection_consistency_max_error_px)

                    if args.refine_camera_from_positionnet:
                        camera_fit_fn = (
                            refine_camera_translation
                            if args.camera_fit_mode == "translation"
                            else refine_camera_extrinsics
                        )
                        camera_fit = camera_fit_fn(
                            body_joints_local,
                            body_targets_px,
                            network_translation,
                            fisheye_cam,
                            args.camera_fit_iters,
                            args.camera_fit_lr,
                        )
                        render_rotation = camera_fit["rotation"]
                        render_translation = camera_fit["translation"]
                        if args.camera_fit_mode == "translation_global_orient":
                            global_orient_stage_mean_error_px = camera_fit[
                                "refined_mean_error_px"
                            ]
                            global_orient_delta_matrix = render_rotation
                            render_global_orient = compose_global_orient(
                                render_rotation,
                                network_global_orient,
                            )
                            cosine = ((torch.trace(render_rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
                            global_orient_delta_degrees = float(
                                torch.rad2deg(torch.acos(cosine)).item()
                            )
                            with torch.no_grad():
                                refined_smplx = model_core.smplx_layer(
                                    betas=out["smplx_shape"],
                                    body_pose=out["smplx_body_pose"],
                                    global_orient=render_global_orient[None, :],
                                    right_hand_pose=out["smplx_rhand_pose"],
                                    left_hand_pose=out["smplx_lhand_pose"],
                                    jaw_pose=out["smplx_jaw_pose"],
                                    leye_pose=zero_eye_pose,
                                    reye_pose=zero_eye_pose,
                                    expression=out["smplx_expr"],
                                )
                            body_joints_local_render = refined_smplx.joints[
                                0, smpl_x.joint_idx[:25], :
                            ]
                            translation_refit = refine_camera_translation(
                                body_joints_local_render,
                                body_targets_px,
                                render_translation,
                                fisheye_cam,
                                args.camera_fit_iters,
                                args.camera_fit_lr,
                            )
                            render_translation = translation_refit["translation"]
                            camera_fit["translation"] = render_translation
                            camera_fit["refined_projection"] = translation_refit[
                                "refined_projection"
                            ]
                            camera_fit["refined_mean_error_px"] = translation_refit[
                                "refined_mean_error_px"
                            ]
                            mesh = (
                                refined_smplx.vertices[0]
                                + render_translation[None, :]
                            ).detach().cpu().numpy()
                            mesh_from_refined_params = mesh.copy()
                            global_orient_mesh_consistency_max_error_m = float(
                                np.linalg.norm(
                                    mesh - mesh_from_refined_params,
                                    axis=1,
                                ).max()
                            )
                        elif args.camera_fit_mode == "extrinsics":
                            camera_rotation_delta = render_rotation
                            body_joints_local_render = (
                                body_joints_local @ render_rotation.t()
                            )
                        else:
                            body_joints_local_render = body_joints_local
                        with torch.no_grad():
                            body_projection_refined = fisheye_cam.world2camera_pytorch(
                                body_joints_local_render
                                + render_translation[None, :]
                            )
                        body_projection_refined_px = body_projection_refined.cpu().numpy()
                        if args.camera_fit_mode != "translation_global_orient":
                            network_translation_np = network_translation.cpu().numpy()
                            render_translation_np = render_translation.cpu().numpy()
                            mesh = (
                                (mesh - network_translation_np[None, :])
                                @ render_rotation.cpu().numpy().T
                                + render_translation_np[None, :]
                            )
                        debug_image = draw_camera_fit_debug(
                            original_img,
                            body_targets_px,
                            body_projection_initial_px,
                            body_projection_refined_px,
                            camera_fit["initial_mean_error_px"],
                            camera_fit["refined_mean_error_px"],
                        )
                        camera_fit_out = debug_dir / f"{image_path.stem}_{bbox_id:02d}_camera_fit.jpg"
                        cv2.imwrite(str(camera_fit_out), rgb_to_bgr_u8(debug_image))
                        saved_camera_fit_images.append(camera_fit_out)

                if body_targets_px is not None:
                    joint_compare_image = draw_positionnet_smplx_2d_comparison(
                        original_img,
                        body_targets_px,
                        body_projection_refined_px,
                    )
                    joint_compare_out = (
                        joint_compare_dir
                        / f"{image_path.stem}_{bbox_id:02d}_positionnet_vs_smplx.jpg"
                    )
                    cv2.imwrite(
                        str(joint_compare_out),
                        rgb_to_bgr_u8(joint_compare_image),
                    )
                    saved_joint_compare_images.append(joint_compare_out)

                smplx_pred = {
                    "global_orient": render_global_orient.reshape(-1, 3).detach().cpu().numpy(),
                    "global_orient_network": network_global_orient.reshape(-1, 3).detach().cpu().numpy(),
                    "global_orient_delta_matrix": global_orient_delta_matrix.reshape(1, 3, 3).detach().cpu().numpy(),
                    "body_pose": out["smplx_body_pose"].reshape(-1, 3).detach().cpu().numpy(),
                    "left_hand_pose": out["smplx_lhand_pose"].reshape(-1, 3).detach().cpu().numpy(),
                    "right_hand_pose": out["smplx_rhand_pose"].reshape(-1, 3).detach().cpu().numpy(),
                    "jaw_pose": out["smplx_jaw_pose"].reshape(-1, 3).detach().cpu().numpy(),
                    "leye_pose": np.zeros((1, 3), dtype=np.float32),
                    "reye_pose": np.zeros((1, 3), dtype=np.float32),
                    "betas": out["smplx_shape"].reshape(-1, 10).detach().cpu().numpy(),
                    "expression": out["smplx_expr"].reshape(-1, 10).detach().cpu().numpy(),
                    "transl": render_translation.reshape(-1, 3).detach().cpu().numpy(),
                    "transl_network": network_translation.reshape(-1, 3).detach().cpu().numpy(),
                    "camera_rotation_delta": camera_rotation_delta.reshape(1, 3, 3).detach().cpu().numpy(),
                    "det_bbox_xyxy_score": np.asarray(det_box).reshape(1, 5).astype(np.float32),
                    "smplx_joint_cam_root_relative": out["smplx_joint_cam_pred"][0].detach().cpu().numpy().astype(np.float32),
                    "smplx_mesh_cam": mesh.astype(np.float32),
                }
                if body_targets_px is not None:
                    body_joints_camera_network = body_joints_local + network_translation[None, :]
                    if camera_fit is not None:
                        body_joints_camera_render = (
                            body_joints_local_render + render_translation[None, :]
                        )
                    else:
                        body_joints_camera_render = body_joints_camera_network
                    smplx_pred["smplx_body_joints_3d_local"] = body_joints_local.detach().cpu().numpy().astype(np.float32)
                    if args.camera_fit_mode == "translation_global_orient" and camera_fit is not None:
                        smplx_pred["smplx_body_joints_3d_local_refined_global_orient"] = (
                            body_joints_local_render
                        ).detach().cpu().numpy().astype(np.float32)
                    smplx_pred["smplx_body_joints_3d_network_cam"] = body_joints_camera_network.detach().cpu().numpy().astype(np.float32)
                    smplx_pred["smplx_body_joints_3d_render_cam"] = body_joints_camera_render.detach().cpu().numpy().astype(np.float32)
                    smplx_pred["positionnet_body_joints_2d"] = body_targets_px.astype(np.float32)
                    if body_joint_pixels_undistorted is not None:
                        smplx_pred[
                            "positionnet_body_joints_2d_global_undistorted"
                        ] = body_joint_pixels_undistorted.astype(np.float32)
                    smplx_pred["smplx_body_joints_2d_network_cam"] = body_projection_initial_px.astype(np.float32)
                    smplx_pred["smplx_body_joints_2d_render_cam"] = body_projection_refined_px.astype(np.float32)
                    smplx_pred["smplx_body_joints_2d_model_output"] = body_projection_model_px.astype(np.float32)
                    smplx_pred["projection_consistency_max_error_px"] = np.asarray(
                        projection_consistency_max_error_px, dtype=np.float32
                    )
                np.savez(smplx_dir / f"{image_path.stem}_{bbox_id:02d}_smplx.npz", **smplx_pred)
                if args.save_mesh:
                    save_obj(mesh, smpl_x.face, str(mesh_dir / f"{image_path.stem}_{bbox_id:02d}.obj"))

                vis_img = render_mesh_fisheye(
                    vis_img,
                    mesh,
                    smpl_x.face,
                    {"focal": [1.0, 1.0], "princpt": [0.0, 0.0]},
                    fisheye_cam,
                    mesh_as_vertices=args.show_verts,
                    mode=args.mesh_mode,
                    proj_trans=render_proj_trans,
                )
                if wireframe_img is not None:
                    wireframe_img = render_mesh_fisheye(
                        wireframe_img,
                        mesh,
                        smpl_x.face,
                        {"focal": [1.0, 1.0], "princpt": [0.0, 0.0]},
                        fisheye_cam,
                        mesh_as_vertices=False,
                        mode="wireframe",
                        proj_trans=render_proj_trans,
                    )
                    if body_targets_px is not None:
                        mesh_joint_compare_image = draw_positionnet_smplx_2d_comparison(
                            wireframe_img,
                            body_targets_px,
                            body_projection_refined_px,
                        )
                        mesh_joint_compare_out = (
                            mesh_joint_compare_dir
                            / f"{image_path.stem}_{bbox_id:02d}_mesh_positionnet_smplx.jpg"
                        )
                        cv2.imwrite(
                            str(mesh_joint_compare_out),
                            rgb_to_bgr_u8(mesh_joint_compare_image),
                        )
                        saved_mesh_joint_compare_images.append(mesh_joint_compare_out)
                if args.show_bbox and not args.no_detector:
                    if args.smplx_input_mode == "det_undistort_contiguous":
                        det_corners_undistorted = np.asarray(
                            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            dtype=np.float32,
                        )
                        det_corners_original = undistorted_pixels_to_original(
                            det_corners_undistorted,
                            undistort_model,
                        )
                        det_corners_int = np.rint(
                            det_corners_original
                        ).astype(np.int32)
                        cv2.polylines(
                            vis_img,
                            [det_corners_int],
                            isClosed=True,
                            color=(255, 0, 0),
                            thickness=3,
                            lineType=cv2.LINE_AA,
                        )
                        label_point = tuple(det_corners_int[0])
                    else:
                        pt1 = (int(round(x1)), int(round(y1)))
                        pt2 = (int(round(x2)), int(round(y2)))
                        cv2.rectangle(vis_img, pt1, pt2, (255, 0, 0), 3)
                        label_point = pt1
                    cv2.putText(
                        vis_img,
                        f"person {score:.2f}",
                        (label_point[0], max(25, label_point[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                meta = {
                    "image_path": str(image_path.resolve()),
                    "bbox_id": int(bbox_id),
                    "det_bbox_xyxy_score": np.asarray(det_box).astype(float).tolist(),
                    "bbox_processed_xywh": bbox.tolist(),
                    "img2bb_trans": img2bb_trans.tolist(),
                    "bb2img_trans": bb2img_trans.tolist(),
                    "fisheye_calibration": str(args.fisheye_calibration.resolve()),
                    "checkpoint_path": str(args.checkpoint_path.resolve()),
                    "config_path": str(args.config_path.resolve()),
                    "smplx_input_mode": args.smplx_input_mode,
                    "detector_coordinate_space": (
                        "global_undistorted"
                        if args.smplx_input_mode == "det_undistort_contiguous"
                        else "original_fisheye"
                    ),
                    "input_crop_mode": args.input_crop_mode,
                    "input_crop_xyxy": input_crop_xyxy if args.smplx_input_mode != "det_crop" else None,
                    "input_coord_offset": input_offset.astype(float).tolist() if args.smplx_input_mode != "det_crop" else [0.0, 0.0],
                    "det_tangent_sampling_bbox_xyxy": (
                        sampling_bbox_xyxy.astype(float).tolist()
                        if sampling_bbox_xyxy is not None
                        else None
                    ),
                    "sampling_bbox_coordinate_space": (
                        "global_undistorted"
                        if args.smplx_input_mode == "det_undistort_contiguous"
                        else (
                            "original_fisheye"
                            if sampling_bbox_xyxy is not None
                            else None
                        )
                    ),
                    "det_tangent_patch_grid_shape": (
                        list(sampling_grid_pixels.shape)
                        if sampling_grid_pixels is not None
                        else None
                    ),
                    "det_tangent_patch_size": (
                        float(args.det_tangent_patch_size)
                        if args.smplx_input_mode == "det_tangent"
                        else None
                    ),
                    "positionnet_2d_mapping": (
                        "bilinear_tangent_patch_centers"
                        if args.smplx_input_mode == "det_tangent"
                        else (
                            "continuous_undistorted_grid_then_brown_conrady_to_original"
                            if args.smplx_input_mode == "det_undistort_contiguous"
                            else "heatmap_to_full_image"
                        )
                    ),
                    "global_undistort_alpha": (
                        float(args.global_undistort_alpha)
                        if args.smplx_input_mode == "det_undistort_contiguous"
                        else None
                    ),
                    "contiguous_roi_mode": (
                        args.contiguous_roi_mode
                        if args.smplx_input_mode == "det_undistort_contiguous"
                        else None
                    ),
                    "global_undistorted_new_k": (
                        undistort_model["new_k"].astype(float).tolist()
                        if args.smplx_input_mode == "det_undistort_contiguous"
                        else None
                    ),
                    "camera_translation_network": network_translation.detach().cpu().numpy().astype(float).tolist(),
                    "camera_translation_render": render_translation.detach().cpu().numpy().astype(float).tolist(),
                    "global_orient_network": network_global_orient.detach().cpu().numpy().astype(float).tolist(),
                    "global_orient_render": render_global_orient.detach().cpu().numpy().astype(float).tolist(),
                    "global_orient_delta_matrix": global_orient_delta_matrix.detach().cpu().numpy().astype(float).tolist(),
                    "global_orient_delta_degrees": float(global_orient_delta_degrees),
                    "global_orient_mesh_consistency_max_error_m": (
                        global_orient_mesh_consistency_max_error_m
                    ),
                    "global_orient_stage_mean_error_px": (
                        global_orient_stage_mean_error_px
                    ),
                    "camera_rotation_delta": camera_rotation_delta.detach().cpu().numpy().astype(float).tolist(),
                    "camera_refined_from_positionnet": bool(camera_fit is not None),
                    "camera_fit_mode": args.camera_fit_mode if camera_fit is not None else None,
                    "projection_consistency_max_error_px": projection_consistency_max_error_px,
                }
                if camera_fit is not None:
                    meta["camera_fit_joint_indices"] = camera_fit["joint_indices"].astype(int).tolist()
                    meta["camera_fit_initial_mean_error_px"] = camera_fit["initial_mean_error_px"]
                    meta["camera_fit_refined_mean_error_px"] = camera_fit["refined_mean_error_px"]
                (meta_dir / f"{image_path.stem}_{bbox_id:02d}.json").write_text(json.dumps(meta, indent=2) + "\n")
                persons_projected += 1
                total_persons += 1
                torch.cuda.empty_cache()

            out_img = overlay_dir / f"{image_path.stem}_det_mesh_overlay.jpg"
            cv2.imwrite(str(out_img), rgb_to_bgr_u8(vis_img))
            saved_images.append(out_img)
            if wireframe_img is not None:
                wireframe_out = wireframe_dir / f"{image_path.stem}_det_mesh_wireframe.jpg"
                cv2.imwrite(str(wireframe_out), rgb_to_bgr_u8(wireframe_img))
                saved_wireframe_images.append(wireframe_out)
            if video_writer is not None:
                video_writer.write(rgb_to_bgr_u8(vis_img))
            records.append({**det_record, "persons_projected": int(persons_projected)})

    if video_writer is not None:
        video_writer.release()
    save_contact_sheet(saved_images, output_dir / "contact_sheet.jpg", max_images=args.contact_sheet_limit)
    if saved_wireframe_images:
        save_contact_sheet(
            saved_wireframe_images,
            output_dir / "contact_sheet_wireframe.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_sampling_grid_images:
        save_contact_sheet(
            saved_sampling_grid_images,
            output_dir / "contact_sheet_vit_sampling_grid.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_patch_atlas_images:
        save_contact_sheet(
            saved_patch_atlas_images,
            output_dir / "contact_sheet_vit_patch_atlas.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_contiguous_patch_grid_images:
        save_contact_sheet(
            saved_contiguous_patch_grid_images,
            output_dir / "contact_sheet_vit_patch_atlas_grid.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_undistorted_images:
        save_contact_sheet(
            saved_undistorted_images,
            output_dir / "contact_sheet_global_undistorted.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_undistorted_detection_images:
        save_contact_sheet(
            saved_undistorted_detection_images,
            output_dir / "contact_sheet_global_undistorted_detection.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_positionnet_images:
        save_contact_sheet(
            saved_positionnet_images,
            output_dir / "contact_sheet_positionnet.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_camera_fit_images:
        save_contact_sheet(
            saved_camera_fit_images,
            output_dir / "contact_sheet_camera_fit.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_joint_compare_images:
        save_contact_sheet(
            saved_joint_compare_images,
            output_dir / "contact_sheet_positionnet_vs_smplx_2d.jpg",
            max_images=args.contact_sheet_limit,
        )
    if saved_mesh_joint_compare_images:
        save_contact_sheet(
            saved_mesh_joint_compare_images,
            output_dir / "contact_sheet_mesh_positionnet_smplx_2d.jpg",
            max_images=args.contact_sheet_limit,
        )

    summary = {
        "image_dir": str(args.image_dir.resolve()),
        "image_glob": args.image_glob,
        "processed_images": int(len(image_paths)),
        "images_without_detection": int(no_detection),
        "projected_persons": int(total_persons),
        "multi_person": bool(args.multi_person),
        "detector_enabled": not args.no_detector,
        "score_thr": float(args.score_thr),
        "bbox_thr": float(args.bbox_thr),
        "smplx_input_mode": args.smplx_input_mode,
        "det_tangent_expand": float(args.det_tangent_expand),
        "det_tangent_patch_size": float(args.det_tangent_patch_size),
        "global_undistort_alpha": float(args.global_undistort_alpha),
        "contiguous_roi_mode": (
            args.contiguous_roi_mode
            if args.smplx_input_mode == "det_undistort_contiguous"
            else None
        ),
        "global_undistorted_source_k": (
            undistort_model["source_k"].astype(float).tolist()
            if undistort_model is not None
            else None
        ),
        "global_undistorted_new_k": (
            undistort_model["new_k"].astype(float).tolist()
            if undistort_model is not None
            else None
        ),
        "global_undistorted_distortion": (
            undistort_model["distortion"].astype(float).tolist()
            if undistort_model is not None
            else None
        ),
        "global_undistorted_valid_roi": (
            list(undistort_model["valid_roi"])
            if undistort_model is not None
            else None
        ),
        "input_crop_mode": args.input_crop_mode,
        "horizontal_crop_pixels": int(args.horizontal_crop_pixels),
        "camera_refined_from_positionnet": bool(args.refine_camera_from_positionnet),
        "camera_fit_mode": args.camera_fit_mode,
        "camera_fit_iters": int(args.camera_fit_iters),
        "camera_fit_lr": float(args.camera_fit_lr),
        "save_wireframe_overlay": bool(args.save_wireframe_overlay),
        "projection_consistency_max_error_px": (
            float(max(projection_consistency_errors))
            if projection_consistency_errors
            else None
        ),
        "output_dir": str(output_dir.resolve()),
        "fisheye_calibration": str(args.fisheye_calibration.resolve()),
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "det_config": str(args.det_config.resolve()),
        "det_checkpoint": str(args.det_checkpoint.resolve()),
        "elapsed_sec": float(time.time() - start_time),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
