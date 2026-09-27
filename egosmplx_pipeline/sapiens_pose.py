#!/usr/bin/env python3
"""Single-person Sapiens2 inference using the installed official pipeline.

An explicit box or the full image is used; no manual keypoint annotations are
read. Output keypoints are in original image pixels, with uncalibrated scores.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True, help='Sapiens2 checkout')
    p.add_argument('--dependencies', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--size', choices=['1b', '5b'], default='1b')
    p.add_argument('--bbox', nargs=4, type=float, help='Original image xyxy; default full image')
    a = p.parse_args()
    root = a.repo_root.resolve()
    repo = root
    sys.path.insert(0, str(repo))
    if a.dependencies:
        sys.path.insert(0, str(a.dependencies))
    import cv2
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from sapiens.engine.config import Config
    from sapiens.engine.datasets import Compose
    from sapiens.pose.datasets import parse_pose_metainfo
    from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap
    from sapiens.registry import MODELS

    im = cv2.imread(str(a.image))
    if im is None:
        raise ValueError(f'Unreadable image: {a.image}')
    h, w = im.shape[:2]
    bbox = a.bbox or [0, 0, w, h]
    if not (0 <= bbox[0] < bbox[2] <= w and 0 <= bbox[1] < bbox[3] <= h):
        raise ValueError('bbox must lie inside the original image')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    weight = a.checkpoint
    digest = hashlib.sha256()
    with weight.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != a.checkpoint_sha256:
        raise ValueError('Pose checkpoint SHA-256 mismatch')
    config = repo / f'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_{a.size}_keypoints308_shutterstock_goliath_3po-1024x768.py'
    cfg = Config.fromfile(config)
    cfg.model['backbone'].pop('init_cfg', None)
    print('Building official pose model', flush=True)
    model = MODELS.build(cfg.model)
    state = load_file(str(weight), device='cpu')
    model.load_state_dict(state, strict=True)
    tensor_count = len(state)
    del state
    model = model.cuda().eval()
    preprocessor = MODELS.build(cfg.data_preprocessor).cuda()
    pipeline = Compose(cfg.test_pipeline)
    meta = parse_pose_metainfo(dict(from_file=str(repo / 'sapiens/pose/configs/_base_/keypoints308.py')))
    names = [meta['keypoint_id2name'][i] for i in range(308)]
    codec = UDPHeatmap(**{k: v for k, v in cfg.codec.items() if k != 'type'})
    data = preprocessor(pipeline(dict(img=im, bbox=np.array([bbox], dtype=np.float32), bbox_score=np.ones(1, dtype=np.float32))))
    inputs, sample = data['inputs'], data['data_samples']
    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        heatmaps = model(inputs)
        if cfg.val_cfg.get('flip_test', False):
            flipped = model(inputs.flip(-1)).flip(-1)[:, meta['flip_indices']]
            heatmaps = (heatmaps + flipped) / 2
    heatmaps = heatmaps.float().cpu().numpy()
    xy, scores = codec.decode(heatmaps[0])
    xy = xy / sample['meta']['input_size'] * sample['meta']['bbox_scale'] + sample['meta']['bbox_center'] - 0.5 * sample['meta']['bbox_scale']
    xy, scores = xy[0], scores[0]
    assert xy.shape == (308, 2) and np.isfinite(xy).all() and np.isfinite(scores).all()
    joints = [dict(index=i, name=n, x=float(xy[i, 0]), y=float(xy[i, 1]), score=float(scores[i]), inside_image=bool(0 <= xy[i, 0] < w and 0 <= xy[i, 1] < h)) for i, n in enumerate(names)]
    payload = dict(model=f'Sapiens2-{a.size.upper()}', image_path=str(a.image.resolve()),
                   image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
                   image_width=w, image_height=h, joint_set='sociopticon308',
                   coordinate_space='original_image_pixels', bbox_xyxy=bbox,
                   visibility='unknown; heatmap scores are not calibrated visibility',
                   keypoints=joints, inference_seconds=time.time() - start)
    (a.output_dir / 'keypoints.json').write_text(json.dumps(payload, indent=2) + '\n')
    np.savez_compressed(a.output_dir / 'heatmaps.npz', heatmap=heatmaps[0])
    body = im.copy()
    for j in joints:
        if j['name'] in {'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'} and j['inside_image']:
            pt = (round(j['x']), round(j['y']))
            color = (0, 220, 0) if j['score'] >= 0.3 else (0, 150, 255)
            cv2.circle(body, pt, 5, color, -1, cv2.LINE_AA)
            cv2.putText(body, f"{j['name']} {j['score']:.2f}", (pt[0] + 5, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(a.output_dir / 'body_keypoints.jpg'), body)
    manifest = dict(checkpoint=str(weight), checkpoint_sha256=digest.hexdigest(),
                    configured_sha256_verified=True, strict_load=True, checkpoint_tensors=tensor_count,
                    config=str(config), python=sys.executable, torch=torch.__version__,
                    input_size_hw=[1024, 768], precision='bf16',
                    flip_test=cfg.val_cfg.get('flip_test', False), manual_labels_used=False,
                    gpu=torch.cuda.get_device_name(), peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
