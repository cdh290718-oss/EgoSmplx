#!/usr/bin/env python3
"""Optimize 19 body joints and small rigid changes using automatic points.

Shape, hands, feet and facial parameters stay fixed. No local surface deformation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['repo-root', 'source-npz', 'image', 'calibration', 'automatic-keypoints', 'output-dir']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--steps', type=int, default=1600)
    a = p.parse_args()
    os.environ['EGOSMPLX_REPO_ROOT'] = str(a.repo_root.resolve())
    from egosmplx_pipeline.bootstrap import setup
    setup(a.repo_root)
    from egosmplx_pipeline.body_common import BODY_POSE_NAMES, MAX_DELTA_DEG, JOINT_MAPPING, save_obj
    from FishEyeCalibrated import FishEyeCameraCalibrated
    from utils.human_models import smpl_x
    from utils.vis import render_mesh_fisheye

    a.output_dir.mkdir(parents=True, exist_ok=False)
    image = cv2.imread(str(a.image))
    if image is None:
        raise ValueError('Unreadable image')
    h, w = image.shape[:2]
    camera = FishEyeCameraCalibrated(str(a.calibration))
    assert tuple(map(int, camera.img_size)) == (w, h)
    raw = {k: v.copy() for k, v in np.load(a.source_npz).items()}
    auto = json.loads(a.automatic_keypoints.read_text())
    assert auto['coordinate_space'] == 'original_image_pixels'
    assert [auto['image_width'], auto['image_height']] == [w, h]
    assert auto['image_sha256'] == hashlib.sha256(a.image.read_bytes()).hexdigest()
    lookup = {j['name']: j for j in auto['keypoints']}
    # Filter automatic observations before optimization.
    names = [n for n in JOINT_MAPPING if n in lookup and np.isfinite([lookup[n][k] for k in ['x', 'y', 'score']]).all()
             and lookup[n]['score'] >= .3 and 0 <= lookup[n]['x'] < w and 0 <= lookup[n]['y'] < h]
    if len(names) < 4:
        raise ValueError('Too few usable automatic targets')
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    indices = torch.tensor([smpl_x.joints_name.index(JOINT_MAPPING[n]) for n in names])
    auto_xy = np.array([[lookup[n]['x'], lookup[n]['y']] for n in names], dtype=np.float32)
    layer = smpl_x.layer['neutral'].cpu().eval()
    layer.requires_grad_(False)
    fixed_keys = ['betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
    fixed = {k: t(raw[k]).reshape(1, -1) for k in fixed_keys}
    body0, glob0, cam0 = t(raw['body_pose']).reshape(21, 3), t(raw['global_orient']).reshape(3), t(raw['transl']).reshape(3)
    active = list(MAX_DELTA_DEG)
    active_indices = torch.tensor([BODY_POSE_NAMES.index(n) for n in active])
    caps = t([np.deg2rad(MAX_DELTA_DEG[n]) for n in active])
    wrists = [smpl_x.joints_name.index('L_Wrist'), smpl_x.joints_name.index('R_Wrist')]
    def forward(body, glob, cam):
        out = layer(body_pose=body.reshape(1, -1), global_orient=glob.reshape(1, 3), **fixed)
        local = out.joints[0, smpl_x.joint_idx]
        mesh = out.vertices[0] + cam
        joints = local + cam
        pixels = camera.world2camera_pytorch(joints)
        return mesh, local, joints, pixels
    with torch.no_grad():
        initial = tuple(v.clone() for v in forward(body0, glob0, cam0))
        initial_wrists = initial[2][wrists].norm(dim=1)
        wrist_floor = initial_wrists * .85
        source_error = float((initial[0] - t(raw['smplx_mesh_cam'])).abs().max())
        assert source_error < 1e-5
    def cpu(x):
        return x.detach().numpy()
    def error(pred, target):
        d = np.linalg.norm(pred - target, axis=1)
        return dict(rmse_px=float(np.sqrt(np.mean(d*d))), mean_px=float(d.mean()), median_px=float(np.median(d)), per_joint_px={n: float(v) for n, v in zip(names, d)})
    def angle_between(x, y):
        rx, ry = cv2.Rodrigues(np.asarray(x, dtype=np.float64))[0], cv2.Rodrigues(np.asarray(y, dtype=np.float64))[0]
        return float(np.rad2deg(np.arccos(np.clip((np.trace(rx @ ry.T)-1)/2, -1, 1))))
    settings = dict(steps=a.steps, device='cpu', target_weighting='uniform automatic targets',
                    body_lr=.0015, camera_lr=.0008, orient_lr=.0005,
                    pose_prior_weight=.0008, camera_prior_weight=.0002, orient_prior_weight=.0004,
                    smooth_l1_beta=.003, max_global_axis_angle_increment_deg=8.,
                    body_axis_angle_increment_limits_deg=MAX_DELTA_DEG,
                    wrist_distance_floor_fraction=.85, wrist_guard_weight=.002,
                    wrist_distance_floor_m=cpu(wrist_floor).tolist(), minimum_vertex_z_m=.02,
                    depth_barrier_onset_m=.03, depth_barrier_scale_m=.02, depth_barrier_weight=.02,
                    best_candidate_rule='lowest total objective among candidates satisfying depth guards; manual evaluation never used')

    def fit(label, target_xy):
        start = time.time()
        target = t(target_xy)
        delta = torch.zeros((len(active), 3), requires_grad=True)
        gd = torch.zeros(3, requires_grad=True)
        cam = cam0.clone().requires_grad_(True)
        opt = torch.optim.Adam([{'params': [delta], 'lr': .0015}, {'params': [gd], 'lr': .0005}, {'params': [cam], 'lr': .0008}])
        best_loss, best = float('inf'), None
        history = []
        for step in range(a.steps + 1):
            opt.zero_grad()
            body = body0.clone()
            body[active_indices] = body[active_indices] + delta
            mesh, local, joints, pixels = forward(body, glob0 + gd, cam)
            rep = F.smooth_l1_loss((pixels[indices] - target) / t([w, h]), torch.zeros_like(target), beta=.003)
            pp = (delta / caps[:, None]).square().mean()
            cp = ((cam - cam0) / t([.10, .10, .18])).square().mean()
            gp = (gd / np.deg2rad(8.)).square().mean()
            wrist_dist = joints[wrists].norm(dim=1)
            guard = F.relu((wrist_floor - wrist_dist) / .05).square().mean()
            # Start the barrier before the hard near plane. The legacy penalty
            # in raw meters was too weak relative to normalized pixel residuals.
            depth = F.relu((.03 - mesh[:, 2].min()) / .02).square()
            loss = rep + .0008*pp + .0002*cp + .0004*gp + .002*guard + .02*depth
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite objective')
            feasible = bool((wrist_dist >= wrist_floor - 1e-6).all() and mesh[:, 2].min() >= .02 - 1e-6)
            value = float(loss.detach())
            if feasible and value < best_loss:
                best_loss = value
                best = (delta.detach().clone(), gd.detach().clone(), cam.detach().clone(), step)
            if step % 100 == 0 or step == a.steps:
                item = dict(step=step, loss=value, reprojection=float(rep.detach()), feasible=feasible,
                            minimum_z_m=float(mesh[:, 2].min().detach()))
                history.append(item)
                print(label, item, flush=True)
            if step == a.steps:
                break
            loss.backward()
            if not all(torch.isfinite(x.grad).all() for x in [delta, gd, cam]):
                raise RuntimeError('Nonfinite optimization gradients')
            opt.step()
            with torch.no_grad():
                delta.mul_(torch.minimum(torch.ones_like(caps), caps / delta.norm(dim=1).clamp_min(1e-8))[:, None])
                gd.mul_(torch.clamp(t(np.deg2rad(8.)) / gd.norm().clamp_min(1e-8), max=1.))
                cam[2].clamp_(.05, 3.)
        if best is None:
            raise RuntimeError('No feasible refinement candidate')
        with torch.no_grad():
            bd, gg, cc, best_step = best
            final_body = body0.clone()
            final_body[active_indices] += bd
            final_global = glob0 + gg
            mesh, local, joints, pixels = forward(final_body, final_global, cc)
        dst = a.output_dir / label
        dst.mkdir()
        saved = {k: v.copy() for k, v in raw.items()}
        saved.update(body_pose_rigid_input=raw['body_pose'], global_orient_rigid_input=raw['global_orient'], transl_rigid_input=raw['transl'],
                     body_pose=cpu(final_body).reshape(raw['body_pose'].shape), global_orient=cpu(final_global).reshape(1, 3), transl=cpu(cc).reshape(1, 3),
                     smplx_mesh_cam=cpu(mesh), smplx_body_joints_3d_local=cpu(local[:25]), smplx_body_joints_3d_render_cam=cpu(joints[:25]),
                     smplx_body_joints_2d_render_cam=cpu(pixels[:25]), smplx_joint_cam_root_relative=cpu(local-local[:1]),
                     stage2_target_joint_pixels=target_xy, stage2_selected_joint_indices=cpu(indices))
        assert all(np.array_equal(saved[k], raw[k]) for k in fixed_keys)
        np.savez_compressed(dst / 'refined_params.npz', **saved)
        # Independently reload the actual artifact and regenerate its geometry.
        reloaded = np.load(dst / 'refined_params.npz')
        assert all(np.isfinite(reloaded[k]).all() for k in reloaded.files if np.issubdtype(reloaded[k].dtype, np.number))
        with torch.no_grad():
            rm, _, _, rp = forward(t(reloaded['body_pose']).reshape(21, 3), t(reloaded['global_orient']).reshape(3), t(reloaded['transl']).reshape(3))
        reload_mesh_error = float(np.abs(cpu(rm)-reloaded['smplx_mesh_cam']).max())
        reload_pixel_error = float(np.abs(cpu(rp[:25])-reloaded['smplx_body_joints_2d_render_cam']).max())
        assert reload_mesh_error < 1e-5 and reload_pixel_error < .01
        save_obj(cpu(mesh), smpl_x.face, dst / 'refined_mesh.obj')
        overlay = render_mesh_fisheye(image, cpu(mesh), smpl_x.face, {}, camera, mode='wireframe', mesh_color=(255, 255, 255), alpha=.78, proj_trans=None)
        cv2.imwrite(str(dst / 'refined_wireframe.jpg'), overlay)
        diag = image.copy()
        for n, target_pt, pred in zip(names, target_xy, cpu(pixels[indices])):
            tx, px = tuple(np.rint(target_pt).astype(int)), tuple(np.rint(pred).astype(int))
            cv2.line(diag, tx, px, (0, 220, 220), 1, cv2.LINE_AA)
            cv2.circle(diag, tx, 6, (0, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(diag, px, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(diag, n, (tx[0]+6, tx[1]-6), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 180, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(dst / 'keypoint_diagnostic.jpg'), diag)
        increments = np.rad2deg(cpu(bd.norm(dim=1)))
        actual_angles = {n: angle_between(cpu(final_body[i]), cpu(body0[i])) for n, i in zip(active, active_indices)}
        report = dict(branch=label, target_source='Sapiens2',
                      manual_labels_used_for_optimization=False,
                      target_fit_before=error(cpu(initial[3][indices]), target_xy), target_fit_after=error(cpu(pixels[indices]), target_xy),
                      best_step=best_step, best_objective=best_loss, elapsed_seconds=time.time()-start,
                      pose_increment_deg={n: float(v) for n, v in zip(active, increments)}, pose_geodesic_change_deg=actual_angles,
                      joints_at_increment_limit=[n for n, v in zip(active, increments) if v >= MAX_DELTA_DEG[n]-.05],
                      global_geodesic_change_deg=angle_between(cpu(final_global), cpu(glob0)),
                      cam_trans_m=cpu(cc).tolist(), cam_trans_change_m=cpu(cc-cam0).tolist(),
                      wrist_camera_distance_m=cpu(joints[wrists].norm(dim=1)).tolist(),
                      initial_wrist_camera_distance_m=cpu(initial_wrists).tolist(), minimum_vertex_z_m=float(mesh[:, 2].min()),
                      fixed_fields_bitwise_equal=fixed_keys, reload_mesh_error_m=reload_mesh_error, reload_pixel_error_px=reload_pixel_error,
                      history=history)
        (dst / 'refinement_report.json').write_text(json.dumps(report, indent=2)+'\n')
        return report, cpu(pixels[indices]), cpu(mesh), overlay, cpu(final_body)

    automatic = fit('automatic', auto_xy)
    if True:
        rows = {
            'rigid_shared_input': {'against_sapiens2': error(cpu(initial[3][indices]), auto_xy)},
            'automatic_stage2': {'against_sapiens2': error(automatic[1], auto_xy)},
        }
        summary = dict(source_npz=str(a.source_npz), source_sha256=hashlib.sha256(a.source_npz.read_bytes()).hexdigest(),
                       automatic_keypoints=str(a.automatic_keypoints), image=str(a.image), calibration=str(a.calibration),
                       comparison='Automatic Sapiens2 stage-two refinement from the supplied rigid initialization',
                       names=names, settings=settings, source_mesh_reload_error_m=source_error, results=rows,
                       automatic_diagnostics=automatic[0])
        (a.output_dir / 'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        rigid_overlay = render_mesh_fisheye(image, cpu(initial[0]), smpl_x.face, {}, camera, mode='wireframe', mesh_color=(255, 255, 255), alpha=.78, proj_trans=None)
        panels = []
        for title, overlay, key in [('Rigid input', rigid_overlay, 'rigid_shared_input'), ('Stage 2: Sapiens2 targets', automatic[3], 'automatic_stage2')]:
            panel = cv2.resize(overlay, (640, round(h*640/w)))
            cv2.rectangle(panel, (0, 0), (640, 62), (25, 25, 25), -1)
            cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Sapiens2 RMSE: {rows[key]['against_sapiens2']['rmse_px']:.2f} px", (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .55, (210, 210, 210), 1, cv2.LINE_AA)
            panels.append(panel)
        cv2.imwrite(str(a.output_dir / 'stage2_comparison.jpg'), np.concatenate(panels, axis=1))
        print('AUTOMATIC_ONLY', json.dumps(rows, indent=2), flush=True)
        return


if __name__ == '__main__':
    main()
