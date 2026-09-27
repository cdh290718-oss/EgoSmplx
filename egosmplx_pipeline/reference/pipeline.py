"""Run the archived wrist/contour stages after the two-stage body fit.

The numerical worker functions are migrated from the reference ZIP. This module
only adapts paths, invokes them in the recorded order, and exports verified data.
"""
import argparse
import os
from pathlib import Path
import shutil
import numpy as np
import cv2
from egosmplx_pipeline.io import read, write, load, sha
from egosmplx_pipeline.configuration import PACKAGE_ROOT
from egosmplx_pipeline.pipeline import command, environment

PROFILE = 'session_hand6_full_pipeline_20260926_v1'


def prepare(rows, output):
    root = output / 'reference'
    prepared = []
    for row in rows:
        frame = output / 'frames' / row['id']
        baseline = frame / 'baseline_fused'
        if not baseline.exists():
            shutil.copytree(frame / 'fused', baseline)
        matches = read(baseline / 'recipe.json')['matches']
        if not matches:
            continue
        if cv2.imread(row['image']).shape[1::-1] != (1280, 720):
            raise ValueError('The archived reference profile requires 1280 x 720 images')
        camera = row.get('camera', 'camera')
        if camera not in ('cam3', 'cam4', 'camera'):
            raise ValueError('Expected cam3/cam4 for the reference profile')
        condition = row['id']
        destination = root / 'baseline/frames' / condition / camera
        destination.mkdir(parents=True, exist_ok=True)
        for source, name in [(frame / 'body/params.npz', 'body_params.npz'),
                             (baseline / 'mesh.npz', 'hybrid_mesh.npz'),
                             (baseline / 'recipe.json', 'recipe.json')]:
            shutil.copyfile(source, destination / name)
        segmentation = root / 'segmentation/frames' / condition / camera
        segmentation.mkdir(parents=True, exist_ok=True)
        for name in ['labels.npy', 'probabilities.npz']:
            target = segmentation / name
            source = Path(row['predictions']['segmentation']) / name
            if target.exists() and sha(target) != sha(source):
                raise ValueError('Changed segmentation input: ' + row['id'])
            if not target.exists():
                target.symlink_to(source.resolve())
        prepared.append(dict(id=row['id'], condition=condition, camera=camera,
                             image=row['image'], calibration=row['calibration'],
                             source_npz=str(frame / 'body/params.npz'),
                             rigid_npz=str(frame / 'rigid/optimized_params.npz'),
                             sapiens_json=row['predictions']['sapiens'],
                             wilor_npz=str(baseline / 'wilor_predictions.npz')))
    write(root / 'baseline/automatic_input_manifest.json', dict(frames=prepared))
    return prepared


def run_workers(config, rows, output):
    root = output / 'reference'
    env = environment(config)
    env['EGOSMPLX_REFERENCE_ROOT'] = str(root)
    def execute(module, arguments, log_name):
        command([config['paths']['body_python'], '-m',
                 'egosmplx_pipeline.reference.' + module] + arguments,
                root / (log_name + '.log'), PACKAGE_ROOT, env)
    execute('wrist_refine', ['--steps', '500'], '01_wrist')
    execute('wrist_refine', ['--steps', '400', '--active-projection'], '02_wrist_active')
    for row in rows:
        relative = Path(row['condition']) / row['camera']
        source, destination = root / 'wrist/active_frames' / relative, root / 'wrist/final_frames' / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['body_params.npz', 'refinement.json']:
            shutil.copyfile(source / name, destination / name)
    execute('wrist_seam', ['--frame-dir', 'final_frames'], '03_wrist_surface')
    execute('contour_refine', ['--steps', '400', '--active-projection'], '04_contour_standard')
    execute('contour_refine_strong', ['--steps', '400', '--active-projection'], '05_contour_strong')
    execute('select_candidates', [], '06_select')
    execute('finalize', [], '07_surface')
    execute('finalize', ['--verify'], '08_independent_reload')


def export(config, rows, prepared, output):
    os.environ['EGOSMPLX_REPO_ROOT'] = config['paths']['egosmplx_repo']
    os.environ['EGOSMPLX_CONFIG'] = config['paths']['egosmplx_config']
    os.environ['EGOSMPLX_REFERENCE_ROOT'] = str(output / 'reference')
    from egosmplx_pipeline.reference.support import FishEyeCameraCalibrated, geometry, render, draw_geometry
    from egosmplx_pipeline.render import read_obj
    mapping = {row['id']: row for row in prepared}
    reports = []
    for row in rows:
        frame = output / 'frames' / row['id']
        report = read(frame / 'validation.json')
        report['pipeline_profile'] = PROFILE
        if row['id'] in mapping:
            reference_row = mapping[row['id']]
            source = output / 'reference/contour/final_frames' / reference_row['condition'] / reference_row['camera']
            fused = frame / 'fused'
            acceptance = read(source / 'acceptance.json')
            if not acceptance.get('independent_reload'):
                raise ValueError('Final surface did not pass independent reconstruction')
            for name, renamed in [('body_params.npz', 'body_params.npz'), ('hybrid_mesh.npz', 'mesh.npz'),
                                  ('hybrid_mesh.obj', 'mesh.obj'), ('refinement.json', 'refinement.json'),
                                  ('geometry_report.json', 'geometry_report.json'), ('acceptance.json', 'acceptance.json'),
                                  ('silhouette_targets.json', 'silhouette_targets.json')]:
                shutil.copyfile(source / name, fused / renamed)
            shutil.copyfile(Path(row['predictions']['segmentation']) / 'labels.npy', fused / 'segmentation_labels.npy')
            parameters, mesh = load(fused / 'body_params.npz'), load(fused / 'mesh.npz')
            v, joints = geometry(parameters)
            camera = FishEyeCameraCalibrated(row['calibration'])
            image = cv2.imread(row['image'])
            cv2.imwrite(str(fused / 'wireframe.jpg'), render(image, mesh['vertices_cam'].astype(np.float64), mesh['faces'], camera, .12))
            cv2.imwrite(str(fused / 'skeleton.jpg'), draw_geometry(image, camera, mesh))
            targets = load(frame / 'body/params.npz')
            body_errors = np.linalg.norm(camera.world2camera(joints.astype(np.float64))[targets['stage2_selected_joint_indices'].astype(int)]
                                         - targets['stage2_target_joint_pixels'], axis=1)
            hands = load(fused / 'wilor_predictions.npz')
            matches = read(frame / 'baseline_fused/recipe.json')['matches']
            hand_errors = []
            for k, side in enumerate(['left', 'right']):
                if side in matches:
                    pixels = camera.world2camera(mesh['hand_joints_cam'][k].astype(np.float64))
                    hand_errors.extend(np.linalg.norm(pixels-hands['joints_2d'][matches[side]['candidate']], axis=1).tolist())
            obj_v, obj_f = read_obj(fused / 'mesh.obj')
            obj_error = float(np.abs(obj_v-mesh['vertices_cam']).max())
            body_error = float(np.abs(v-parameters['smplx_mesh_cam']).max())
            if obj_error >= 1e-7 or body_error > 1e-5 or not np.array_equal(obj_f, mesh['faces']):
                raise RuntimeError('Exported final geometry differs from verified mesh')
            if 'stage2_body_error_px' not in report:
                report['stage2_body_error_px'] = report['body_error_px']
                report['stage2_body_rmse_px'] = report['body_rmse_px']
            report.update(body_error_px=body_errors.tolist(), hand_error_px=hand_errors,
                          body_rmse_px=float(np.sqrt(np.mean(body_errors**2))),
                          hand_rmse_px=float(np.sqrt(np.mean(np.square(hand_errors)))),
                          body_reload_max_abs_m=body_error, fusion_reload_max_abs_m=0., obj_reload_max_abs_m=obj_error,
                          minimum_fused_vertex_z_m=float(mesh['vertices_cam'][:, 2].min()),
                          hand_reports=read(fused / 'geometry_report.json')['hands'], local_surface_optimizer=True,
                          selected_body_candidate=read(fused / 'refinement.json')['selected_candidate'])
            write(fused / 'recipe.json', dict(matches=matches, pipeline_profile=PROFILE,
                  method='bounded_standard_wrist_harmonic_seam_with_visible_contour',
                  body_parameter_path='body_params.npz', frozen_targets='silhouette_targets.json',
                  segmentation_labels='segmentation_labels.npy', local_surface_optimizer=True,
                  baseline_smoothstep_rings=7, manual_annotations_used=False,
                  standard_smplx_parameter_only_reconstruction=False))
        else:
            report.update(local_surface_optimizer=False, reference_refinement_skipped='no_matched_hand')
        report['artifact_sha256'] = {str(p.relative_to(frame)): sha(p) for p in frame.rglob('*')
                                     if p.is_file() and p != frame / 'validation.json'}
        write(frame / 'validation.json', report)
        reports.append(report)
    write(output / 'reference/validation.json', dict(pipeline_profile=PROFILE, frames=reports))
    return reports


def run(config, rows, output):
    output = Path(output).resolve()
    prepared = prepare(rows, output)
    if prepared:
        run_workers(config, prepared, output)
    return export(config, rows, prepared, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = read(args.job)
    run(job['config'], job['rows'], job['output'])


if __name__ == '__main__':
    main()
