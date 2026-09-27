#!/usr/bin/env python3
"""Fit only SMPL-X global orientation and translation to Sapiens2 2D points.

Reuses the existing calibrated-fisheye rigid-fit helpers. Only automatic Sapiens2 targets are accepted.
The body rotates about the SMPL-X pelvis; saved parameters are regenerated and
checked against the optimized geometry before any result is delivered.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['repo-root', 'source-npz', 'calibration', 'keypoints', 'image', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--score-threshold', type=float, default=0.3)
    p.add_argument('--steps', type=int, default=1800)
    p.add_argument('--max-orient-delta-deg', type=float, default=60.)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Rigid fitting is small; CPU avoids old CUDA numerical differences')
    a = p.parse_args()
    if not 0 <= a.score_threshold <= 1 or a.steps < 1:
        raise ValueError('Invalid score threshold or step count')
    os.environ['EGOSMPLX_REPO_ROOT'] = str(a.repo_root.resolve())
    from egosmplx_pipeline.bootstrap import setup
    setup(a.repo_root)
    from egosmplx_pipeline.body_common import axis_angle_rotation, save_obj, JOINT_MAPPING
    from FishEyeCalibrated import FishEyeCameraCalibrated
    from utils.human_models import smpl_x
    from utils.vis import render_mesh_fisheye

    image = cv2.imread(str(a.image))
    if image is None:
        raise ValueError('Unreadable image')
    h, w = image.shape[:2]
    payload = json.loads(a.keypoints.read_text())
    if payload['coordinate_space'] != 'original_image_pixels':
        raise ValueError('Sapiens2 targets must use original image pixels')
    if [payload['image_width'], payload['image_height']] != [w, h]:
        raise ValueError('Target dimensions differ from image')
    if payload['image_sha256'] != hashlib.sha256(a.image.read_bytes()).hexdigest():
        raise ValueError('Target image hash mismatch')
    camera = FishEyeCameraCalibrated(str(a.calibration.resolve()))
    if tuple(map(int, camera.img_size)) != (w, h):
        raise ValueError('Calibration dimensions differ from image')
    raw = {k: v.copy() for k, v in np.load(a.source_npz).items()}
    fixed_keys = ['body_pose', 'betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
    for key in fixed_keys + ['global_orient', 'transl', 'smplx_mesh_cam', 'smplx_body_joints_3d_local']:
        if key not in raw or not np.isfinite(raw[key]).all():
            raise ValueError('Missing or nonfinite source field: ' + key)
    dev = torch.device(a.device)
    # Rigid composition and parameter regeneration require full FP32 mantissas;
    # TF32 matrix products can introduce millimeter-scale discrepancies.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    layer = smpl_x.layer['neutral'].to(dev).eval()
    fixed = {key: t(raw[key]).reshape(1, -1) for key in fixed_keys}
    cam0 = t(raw['transl']).reshape(3)
    glob0 = t(raw['global_orient']).reshape(1, 3)
    with torch.no_grad():
        initial = layer(global_orient=glob0, **fixed)
        local = initial.joints[0, smpl_x.joint_idx[:25]].detach()
        vertices = initial.vertices[0].detach()
        pivot = initial.joints[0, 0].detach()
        mesh_error = float((vertices + cam0 - t(raw['smplx_mesh_cam'])).abs().max())
        joints_error = float((local - t(raw['smplx_body_joints_3d_local'])).abs().max())
    print('Source reconstruction errors (m):', mesh_error, joints_error, flush=True)
    # Legacy CUDA inference and full-precision CPU LBS differ by submillimeter
    # rounding. Audit this separately from the much tighter final round trip.
    if max(mesh_error, joints_error) > 1e-3:
        raise ValueError(f'Source parameter reconstruction mismatch: mesh={mesh_error}, joints={joints_error} m')

    # Selection is determined solely by model scores and in-image coordinates.
    lookup = {j['name']: j for j in payload['keypoints']}
    if len(lookup) != len(payload['keypoints']):
        raise ValueError('Duplicate keypoint names')
    selected, excluded = [], []
    for name, model_name in JOINT_MAPPING.items():
        j = lookup.get(name)
        valid = j is not None and np.isfinite([j['x'], j['y'], j['score']]).all()
        valid = valid and 0 <= j['x'] < w and 0 <= j['y'] < h and j['score'] >= a.score_threshold
        if valid:
            selected.append(dict(name=name, model_name=model_name, index=smpl_x.joints_name.index(model_name), x=j['x'], y=j['y'], score=j['score']))
        else:
            excluded.append(dict(name=name, prediction=j, reason='missing, nonfinite, outside image, or below score threshold'))
    if len(selected) < 4:
        raise ValueError(f'Only {len(selected)} usable automatic targets; need at least four')
    target = t([[j['x'], j['y']] for j in selected])
    if np.linalg.svd(target.cpu().numpy() - target.cpu().numpy().mean(0), compute_uv=False)[1] < 10:
        raise ValueError('Automatic targets have insufficient 2D spatial spread')
    indices = torch.tensor([j['index'] for j in selected], device=dev)
    weights = t([np.clip(j['score'], 0, 1)**2 for j in selected])
    cam = cam0.clone().requires_grad_(True)
    delta = torch.zeros(3, device=dev, requires_grad=True)
    scale = t([w, h])
    def geometry():
        rotation = axis_angle_rotation(delta)
        joints = (local - pivot) @ rotation.t() + pivot + cam
        mesh = (vertices - pivot) @ rotation.t() + pivot + cam
        pixels = camera.world2camera_pytorch(joints)
        return joints, mesh, pixels
    with torch.no_grad():
        _, _, before = geometry()
        before = before.clone()
    opt = torch.optim.Adam([{'params': [cam], 'lr': .005}, {'params': [delta], 'lr': .002}])
    best_loss, best_state = float('inf'), None
    history = []
    for step in range(a.steps + 1):
        opt.zero_grad()
        joints, mesh, pixels = geometry()
        residual = (pixels[indices] - target) / scale
        robust = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.005, reduction='none').mean(1)
        reprojection = (robust * weights).sum() / weights.sum()
        prior_t = (((cam - cam0) / t([.25, .25, .50]))**2).mean()
        loss = reprojection + 1e-4 * prior_t + 2e-4 * delta.square().mean() + F.relu(.02 - mesh[:, 2].min()).square()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite optimization objective')
        value = float(loss.detach())
        if value < best_loss:
            best_loss, best_state = value, (cam.detach().clone(), delta.detach().clone())
        if step % 100 == 0 or step == a.steps:
            history.append(dict(step=step, loss=value, reprojection=float(reprojection.detach())))
        if step == a.steps:
            break
        loss.backward()
        if not torch.isfinite(cam.grad).all() or not torch.isfinite(delta.grad).all():
            raise RuntimeError('Nonfinite rigid-fit gradient')
        opt.step()
        with torch.no_grad():
            norm = delta.norm()
            limit = np.deg2rad(a.max_orient_delta_deg)
            if norm > limit:
                delta.mul_(limit / norm)
            cam[2].clamp_(.05, 3.)
    with torch.no_grad():
        cam.copy_(best_state[0])
        delta.copy_(best_state[1])
        joints, mesh, after = geometry()
        delta_rotation = axis_angle_rotation(delta).cpu().numpy()
        new_rotation = delta_rotation @ cv2.Rodrigues(raw['global_orient'].astype(np.float64).reshape(3))[0]
        new_global = cv2.Rodrigues(new_rotation.astype(np.float64))[0].astype(np.float32).reshape(1, 3)
        regenerated = layer(global_orient=t(new_global), **fixed)
        rebuilt_mesh = regenerated.vertices[0] + cam
        rebuilt_local = regenerated.joints[0, smpl_x.joint_idx[:25]]
        rebuilt_pixels = camera.world2camera_pytorch(rebuilt_local + cam)
        regen_mesh_error = float((rebuilt_mesh - mesh).abs().max())
        regen_pixel_error = float((rebuilt_pixels - after).abs().max())
        if regen_mesh_error > 1e-4 or regen_pixel_error > .05:
            print('REGEN_DIAGNOSTIC', dict(
                difference_mean=(rebuilt_mesh - mesh).mean(0).cpu().tolist(),
                difference_std=(rebuilt_mesh - mesh).std(0).cpu().tolist(),
                pivot=pivot.cpu().tolist(), regenerated_pivot=regenerated.joints[0, 0].cpu().tolist(),
                rotation_roundtrip_error=float(np.max(np.abs(cv2.Rodrigues(new_global.astype(np.float64).reshape(3))[0] - new_rotation))),
                rotation_orthogonality_error=float(np.max(np.abs(new_rotation.T @ new_rotation - np.eye(3)))),
                delta=delta.cpu().tolist(), global_new=new_global.tolist()), flush=True)
            raise RuntimeError(f'Regenerated geometry mismatch: {regen_mesh_error} m, {regen_pixel_error} px')
    cpu = lambda x: x.detach().cpu().numpy()
    saved = {k: v.copy() for k, v in raw.items()}
    saved.update(global_orient=new_global,
                 transl=cpu(cam).reshape(1, 3), smplx_mesh_cam=cpu(rebuilt_mesh),
                 smplx_body_joints_3d_local=cpu(rebuilt_local),
                 smplx_body_joints_3d_render_cam=cpu(rebuilt_local + cam),
                 smplx_body_joints_2d_render_cam=cpu(rebuilt_pixels),
                 sapiens2_target_joint_pixels=cpu(target), sapiens2_selected_joint_indices=cpu(indices),
                 sapiens2_target_scores=np.array([j['score'] for j in selected], dtype=np.float32),
                 global_orient_delta_rotation=delta_rotation)
    full_joints = regenerated.joints[0, smpl_x.joint_idx]
    saved['smplx_joint_cam_root_relative'] = cpu(full_joints - full_joints[:1])
    if not all(np.array_equal(saved[k], raw[k]) for k in fixed_keys):
        raise RuntimeError('A fixed parameter was changed')
    def errors(pred, truth):
        d = np.linalg.norm(pred - truth, axis=1)
        return dict(rmse_px=float(np.sqrt(np.mean(d**2))), mean_px=float(d.mean()), median_px=float(np.median(d)))
    report = dict(source_npz=str(a.source_npz.resolve()), image=str(a.image.resolve()), calibration=str(a.calibration.resolve()),
                  sapiens2_keypoints=str(a.keypoints.resolve()), optimized_parameters=['global_orient', 'cam_trans'],
                  fixed_parameters=fixed_keys, fixed_parameters_bitwise_equal=True, network_was_rerun=False,
                  manual_labels_used_for_optimization=False, score_threshold=a.score_threshold,
                  score_weight='clip(score,0,1)^2; scores are not calibrated visibility probabilities',
                  selected=selected, excluded=excluded, initial_cam_trans_m=cpu(cam0).tolist(),
                  optimized_cam_trans_m=cpu(cam).tolist(), initial_global_orient=raw['global_orient'].tolist(),
                  optimized_global_orient=new_global.tolist(), global_orient_delta_deg=float(np.rad2deg(np.linalg.norm(cpu(delta)))),
                  source_reconstruction_max_abs_m=dict(mesh=mesh_error, joints=joints_error),
                  source_reconstruction_tolerance_m=1e-3, optimization_device=str(dev),
                  saved_parameter_regeneration_max_abs_m=regen_mesh_error,
                  saved_parameter_regeneration_max_abs_px=regen_pixel_error,
                  source_npz_sha256=hashlib.sha256(a.source_npz.read_bytes()).hexdigest(),
                  raw_to_sapiens2=errors(cpu(before[indices]), cpu(target)),
                  optimized_to_sapiens2=errors(cpu(rebuilt_pixels[indices]), cpu(target)), history=history,
                  minimum_vertex_z_m=float(rebuilt_mesh[:, 2].min()),
                  geometry_convention='SMPL-X vertices/joints include global orientation; add transl; rotate about SMPL-X pelvis',
                  preserved_network_fields='Fields containing network or model_output and positionnet are original inference diagnostics')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output_dir / 'optimized_params.npz', **saved)
    save_obj(cpu(rebuilt_mesh), smpl_x.face, a.output_dir / 'optimized_mesh.obj')
    overlays = []
    for name, verts in [('raw', raw['smplx_mesh_cam']), ('optimized', cpu(rebuilt_mesh))]:
        overlay = render_mesh_fisheye(image, verts, smpl_x.face, {}, camera, mode='wireframe', mesh_color=(255, 255, 255), alpha=.78, proj_trans=None)
        cv2.imwrite(str(a.output_dir / f'{name}_wireframe.jpg'), overlay)
        panel = cv2.resize(overlay, (640, round(h * 640 / w)))
        cv2.rectangle(panel, (0, 0), (640, 38), (25, 25, 25), -1)
        cv2.putText(panel, 'Raw EgoSMPLX' if name == 'raw' else 'Sapiens2 -> global orient + translation', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
        overlays.append(panel)
    cv2.imwrite(str(a.output_dir / 'comparison.jpg'), np.concatenate(overlays, axis=1))
    diagnostic = image.copy()
    for j, initial_xy, final_xy in zip(selected, cpu(before[indices]), cpu(rebuilt_pixels[indices])):
        target_xy = (round(j['x']), round(j['y']))
        cv2.line(diagnostic, target_xy, tuple(np.rint(final_xy).astype(int)), (0, 220, 220), 1, cv2.LINE_AA)
        for xy, color in [(initial_xy, (255, 100, 0)), (final_xy, (0, 0, 255)), (target_xy, (0, 220, 0))]:
            cv2.circle(diagnostic, tuple(np.rint(xy).astype(int)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(diagnostic, j['name'], (target_xy[0] + 7, target_xy[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(diagnostic, 'BLUE raw | GREEN Sapiens2 | RED optimized', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(a.output_dir / 'keypoint_diagnostic.jpg'), diagnostic)
    (a.output_dir / 'optimization_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
