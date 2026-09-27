"""Reconstruct the specified result bundle from saved parameters and constraints.

This verifies the final mesh-producing algorithm. It does not rerun the networks
or the body/wrist optimizers, and does not need the omitted probability caches.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import cv2
import numpy as np
from egosmplx_pipeline.io import write, sha

REFERENCE_SHA256 = '19b9baf80998dbd3f6f0da8f620295251d97a2c9aeeb7c2d282d5c31835e7fd6'


def frozen_targets(labels, report):
    """Recover exactly the saved visible strips, without re-estimating targets."""
    skin = np.isin(labels, [6,7,11,15,16,20])
    outside = cv2.distanceTransform((~skin).astype(np.uint8), cv2.DIST_L2, 5)
    return [dict(side=side, ids=np.array(item['vertex_ids'], dtype=int), outside=outside,
                 contour=np.array([b[k] for b in item['bins'] for k in ['left','right']], np.float32))
            for side, item in report.items() if item['enabled']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='EgoSMPLX backend model config')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.archive) != REFERENCE_SHA256:
        raise ValueError('This verifier is pinned to the exact session_hand6 result bundle')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ['EGOSMPLX_REPO_ROOT'] = str(args.repo_root.resolve())
    os.environ['EGOSMPLX_CONFIG'] = str(args.config.resolve())
    os.environ['EGOSMPLX_REFERENCE_ROOT'] = str((args.output/'reference').resolve())
    from egosmplx_pipeline.reference.support import geometry, FishEyeCameraCalibrated, smpl_x, render, write_obj
    from egosmplx_pipeline.reference.contour_seam import build
    records = []
    with zipfile.ZipFile(args.archive) as archive:
        def read(name):
            return json.loads(archive.read(name))
        def array(name):
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as z:
                return {key:z[key].copy() for key in z.files}
        manifest = read('ARCHIVE_MANIFEST.json')
        for name, entry in manifest.items():
            content = archive.read(name)
            if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Archive member hash mismatch: ' + name)
        cases = sorted(name[len('frames/'):-len('/03_fusion/body_params.npz')]
                       for name in manifest if name.startswith('frames/') and name.endswith('/03_fusion/body_params.npz'))
        for case in cases:
            # No arbitrary ZIP paths are extracted. Only these generated case paths are used.
            condition, camera_name = case.split('/')
            if not condition.startswith('sample_') or camera_name not in ['cam3','cam4']:
                raise ValueError('Unexpected reference case')
            d = args.output/'frames'/condition/camera_name
            d.mkdir(parents=True)
            calibration = d/'calibration.json'
            calibration.write_bytes(archive.read('calibration/'+camera_name+'.json'))
            camera = FishEyeCameraCalibrated(calibration)
            prefix = 'frames/'+case+'/'
            parameters = array(prefix+'03_fusion/body_params.npz')
            hands = array(prefix+'automatic_observations/wilor_predictions.npz')
            hands.update(focal_length_px=np.array(25000.), image_size_wh=np.array([1280,720]))
            matches = read(prefix+'automatic_observations/identity_recipe.json')['matches']
            labels = np.load(io.BytesIO(archive.read(prefix+'automatic_observations/segmentation_labels.npy')), allow_pickle=False)
            targets = frozen_targets(labels, read(prefix+'03_fusion/silhouette_targets.json'))
            mesh, surface_report = build(parameters, hands, matches, camera, targets)
            expected = array(prefix+'03_fusion/hybrid_mesh.npz')
            if set(mesh) != set(expected):
                raise RuntimeError('Rebuilt mesh fields differ from archive')
            equality = {key:bool(np.array_equal(mesh[key], expected[key])) for key in mesh}
            maximum = float(np.max(np.abs(mesh['vertices_cam']-expected['vertices_cam'])))
            if not all(equality.values()):
                raise RuntimeError('Archive reconstruction differs: '+case+' '+str(equality))
            np.savez_compressed(d/'hybrid_mesh.npz', **mesh)
            write_obj(d/'hybrid_mesh.obj', mesh['vertices_cam'], mesh['faces'])
            image = cv2.imdecode(np.frombuffer(archive.read(prefix+'input.jpg'),np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(d/'projection.jpg'), render(image,mesh['vertices_cam'].astype(np.float64),mesh['faces'],camera,.12))
            errors = []
            for k, side in enumerate(['left','right']):
                if side in matches:
                    pixels = camera.world2camera(mesh['hand_joints_cam'][k].astype(np.float64))
                    errors.extend(np.linalg.norm(pixels-hands['joints_2d'][matches[side]['candidate']],axis=1).tolist())
            record = dict(case=case, all_fields_bitwise_equal=True, vertex_max_abs_m=maximum,
                          matched_hands=len(matches), hand_error_px=errors, hand_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),
                          selected_surface_candidates={s:r['selected']['alpha'] for s,r in surface_report.items()})
            write(d/'verification.json', record)
            records.append(record)
            print('ARCHIVE_REBUILT',case,maximum,flush=True)
    errors = [e for record in records for e in record['hand_error_px']]
    write(args.output/'archive_verification.json', dict(reference_archive_sha256=REFERENCE_SHA256,frame_count=len(records),
          verified_archive_members=len(manifest), all_mesh_fields_bitwise_equal=True,
          hand_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),frames=records,
          verification_scope='Final meshes rebuilt from archived body parameters, WiLoR predictions, identities and frozen contour targets; networks and optimizers were not rerun'))


if __name__ == '__main__':
    main()
