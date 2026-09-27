"""Execute fitting and fixed-rule fusion for one automatically observed frame."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from egosmplx_pipeline.io import read, write, load, sha, rmse


def canonicalize(parameters, model, camera):
    from egosmplx_pipeline.fusion import geometry
    vertices, joints = geometry(parameters)
    pixels = camera.world2camera(joints.astype(np.float64)).astype(np.float32)
    parameters.update(smplx_mesh_cam=vertices, smplx_joint_cam=joints, smplx_joint_img=pixels,
                      smplx_joint_cam_root_relative=joints-joints[:1],
                      smplx_body_joints_3d_local=(joints-parameters['transl'].reshape(3))[:25],
                      smplx_body_joints_3d_render_cam=joints[:25],
                      smplx_body_joints_2d_render_cam=pixels[:25])
    return parameters


def invoke(function, arguments):
    previous = sys.argv
    try:
        sys.argv = [function.__module__] + list(map(str, arguments))
        function()
    finally:
        sys.argv = previous


def run(config, row, output):
    from egosmplx_pipeline.bootstrap import setup
    model, Camera = setup(config['paths']['egosmplx_repo'])
    from egosmplx_pipeline import fusion, body_rigid, body_pose
    from egosmplx_pipeline.matching import assign
    from egosmplx_pipeline.render import render, write_obj, read_obj
    fusion.initialize(model)
    camera = Camera(row['calibration'])
    image = cv2.imread(row['image'])
    if image is None or tuple(camera.img_size) != tuple(image.shape[1::-1]):
        raise ValueError('Image or calibration dimensions invalid')
    output.mkdir(parents=True, exist_ok=False)
    predictions = row['predictions']
    targets = read(predictions['sapiens'])
    image_digest = sha(row['image'])
    if targets['image_sha256'] != image_digest:
        raise ValueError('Automatic target image hash differs from input')
    raw = load(predictions['egosmplx'])
    for key in ['global_orient', 'transl']:
        if key + '_network' not in raw:
            raise ValueError('Raw prediction requires explicit original network field: ' + key)
        raw[key] = raw[key + '_network'].copy()
    # Retain reconstruction inputs and original network rigid outputs only.
    keep = fusion.PARAMS + ['transl', 'global_orient_network', 'transl_network']
    raw = canonicalize({k: raw[k] for k in keep}, model, camera)
    raw_dir = output / 'raw'
    raw_dir.mkdir()
    np.savez_compressed(raw_dir / 'params.npz', **raw)
    write_obj(raw_dir / 'mesh.obj', raw['smplx_mesh_cam'], model.face)
    cv2.imwrite(str(raw_dir / 'wireframe.jpg'), render(image, raw['smplx_mesh_cam'].astype(np.float64), model.face, camera, .12))
    common = ['--repo-root', config['paths']['egosmplx_repo'], '--image', row['image'], '--calibration', row['calibration']]
    invoke(body_rigid.main, common + ['--source-npz', raw_dir / 'params.npz', '--keypoints', predictions['sapiens'],
                                    '--output-dir', output / 'rigid', '--steps', config['fitting']['rigid_steps'], '--device', 'cpu'])
    invoke(body_pose.main, common + ['--source-npz', output / 'rigid/optimized_params.npz',
                                   '--automatic-keypoints', predictions['sapiens'], '--segmentation', predictions['segmentation'],
                                   '--output-dir', output / 'body_fit', '--steps', config['fitting']['pose_steps'],
                                   '--silhouette-weight', config['fitting']['silhouette_weight']])
    body = canonicalize(load(output / 'body_fit/automatic/refined_params.npz'), model, camera)
    body_dir = output / 'body'
    body_dir.mkdir()
    np.savez_compressed(body_dir / 'params.npz', **body)
    write_obj(body_dir / 'mesh.obj', body['smplx_mesh_cam'], model.face)
    cv2.imwrite(str(body_dir / 'wireframe.jpg'), render(image, body['smplx_mesh_cam'].astype(np.float64), model.face, camera, .12))
    hands = load(predictions['wilor'])
    if 'image_sha256' in hands and str(hands['image_sha256']) != image_digest:
        raise ValueError('WiLoR image hash differs from input')
    if 'focal_length_px' not in hands:
        if 'cached_wilor_focal_px' not in config:
            raise ValueError('Legacy WiLoR cache requires an explicit verified focal length')
        hands['focal_length_px'] = np.array(config['cached_wilor_focal_px'])
        hands['image_size_wh'] = np.array(image.shape[1::-1])
    if not np.array_equal(hands['image_size_wh'], image.shape[1::-1]):
        raise ValueError('WiLoR image size mismatch')
    lookup = {q['name']: q for q in targets['keypoints']}
    matches = assign(hands, lookup, body['smplx_joint_img'], *image.shape[1::-1])
    mesh, hand_reports = fusion.stitch(body, hands, matches, camera)
    fused = output / 'fused'
    fused.mkdir()
    np.savez_compressed(fused / 'mesh.npz', **mesh)
    np.savez_compressed(fused / 'wilor_predictions.npz', **hands)
    write_obj(fused / 'mesh.obj', mesh['vertices_cam'], mesh['faces'])
    write(fused / 'recipe.json', dict(matches=matches, blend_rings=7, body_parameter_path='../body/params.npz',
                                     manual_annotations_used=False, local_surface_optimizer=False,
                                     standard_smplx_parameter_only_reconstruction=False))
    cv2.imwrite(str(fused / 'wireframe.jpg'), render(image, mesh['vertices_cam'].astype(np.float64), mesh['faces'], camera, .12))
    cv2.imwrite(str(fused / 'skeleton.jpg'), fusion.draw_geometry(image, camera, mesh))
    # Reload actual saved artifacts, including the fusion inputs and recipe.
    reloaded_body = load(body_dir / 'params.npz')
    v, j = fusion.geometry(reloaded_body)
    body_error = float(np.max(np.abs(v-reloaded_body['smplx_mesh_cam'])))
    rebuilt, _ = fusion.stitch(reloaded_body, load(fused / 'wilor_predictions.npz'), read(fused / 'recipe.json')['matches'], camera)
    stored = load(fused / 'mesh.npz')
    fusion_error = float(np.max(np.abs(rebuilt['vertices_cam']-stored['vertices_cam'])))
    obj_vertices, obj_faces = read_obj(fused / 'mesh.obj')
    obj_error = float(np.max(np.abs(obj_vertices-stored['vertices_cam'])))
    if max(body_error, fusion_error, obj_error) > 1e-5 or not np.array_equal(obj_faces, stored['faces']):
        raise RuntimeError('Saved geometry failed reload validation')
    indices = body['stage2_selected_joint_indices'].astype(int)
    truth = body['stage2_target_joint_pixels']
    body_errors = np.linalg.norm(body['smplx_joint_img'][indices]-truth, axis=1)
    hand_errors = []
    for side, match in matches.items():
        k = fusion.SIDES.index(side)
        projected = camera.world2camera(mesh['hand_joints_cam'][k].astype(np.float64))
        hand_errors.extend(np.linalg.norm(projected-hands['joints_2d'][match['candidate']], axis=1).tolist())
    report = dict(id=row['id'], passed=True, matched_hands=len(matches), matches=matches,
                  body_target_count=len(indices), body_error_px=body_errors.tolist(), hand_error_px=hand_errors,
                  body_rmse_px=rmse(body['smplx_joint_img'][indices], truth),
                  hand_rmse_px=float(np.sqrt(np.mean(np.square(hand_errors)))) if hand_errors else None,
                  body_reload_max_abs_m=body_error, fusion_reload_max_abs_m=fusion_error,
                  obj_reload_max_abs_m=obj_error, minimum_fused_vertex_z_m=float(mesh['vertices_cam'][:,2].min()),
                  hand_reports=hand_reports, manual_annotations_used=False,
                  accuracy_reference='automatic predictions, not human ground truth',
                  input_image_sha256=image_digest, calibration_sha256=sha(row['calibration']))
    report['artifact_sha256'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    write(output / 'validation.json', report)
    print('FRAME_COMPLETE', row['id'], report['body_rmse_px'], report['hand_rmse_px'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = read(args.job)
    run(job['config'], job['row'], Path(job['output']))


if __name__ == '__main__':
    main()
