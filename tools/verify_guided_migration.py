"""Verify the migrated guided-body/original-seam implementation on saved automatic inputs.

Requires a frozen automatic manifest and the 2026-09-29 expected experiment.
Selected frames rerun the full added optimizer; all frames rebuild and reload
final meshes. Neural networks and upstream two-stage fitting are not rerun.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egosmplx_pipeline.io import read, write, load, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--expected', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repo-root', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--optimize-indices', type=int, nargs='*', default=[0, 6])
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ['EGOSMPLX_REPO_ROOT'] = str(args.repo_root.resolve())
    os.environ['EGOSMPLX_CONFIG'] = str(args.config.resolve())
    os.environ['EGOSMPLX_REFERENCE_ROOT'] = str(args.output.resolve())
    from egosmplx_pipeline.reference import guided_body as guided
    from egosmplx_pipeline.reference.guided_fuse import run as fuse
    rows = read(args.manifest)['frames']
    records = []
    for i, row in enumerate(rows):
        expected = args.expected / 'frames' / row['condition'] / row['camera']
        destination = guided.frame_dir(row)
        destination.mkdir(parents=True, exist_ok=True)
        # context() verifies the manifest's per-input SHA-256 before using observations.
        guided.context(row)
        if i in args.optimize_indices:
            guided.fit(row, 1400)
        else:
            shutil.copyfile(expected / 'body_params.npz', destination / 'body_params.npz')
        actual_p, expected_p = load(destination/'body_params.npz'), load(expected/'body_params.npz')
        core = guided.PARAMS + ['transl']
        exact = all(np.array_equal(actual_p[k], expected_p[k]) for k in core)
        if not exact:
            raise AssertionError('Body optimization differs from expected frame ' + row['id'])
        fuse(row)
        fuse(row, verify=True)
        actual, baseline = load(destination/'hybrid_mesh.npz'), load(expected/'hybrid_mesh.npz')
        equality = {k:bool(np.array_equal(actual[k], baseline[k])) for k in baseline if k in actual}
        if set(actual) != set(baseline) or not all(equality.values()):
            raise AssertionError('Fusion differs from expected frame ' + row['id'])
        record = dict(id=row['id'], condition=row['condition'], camera=row['camera'],
                      optimizer_rerun=i in args.optimize_indices, body_parameters_bitwise_equal=exact,
                      all_mesh_fields_bitwise_equal=True,
                      vertex_max_abs_m=float(np.abs(actual['vertices_cam']-baseline['vertices_cam']).max()),
                      local_crossings_passed=read(destination/'geometry_validation.json')['all_local_crossing_checks_passed'],
                      independent_reload=read(destination/'reload_validation.json')['passed'])
        records.append(record)
        write(args.output/'progress.json', dict(frames=records))
        print('MIGRATION_VERIFIED', row['id'], flush=True)
    write(args.output/'migration_validation.json', dict(frame_count=len(records),
          manifest_sha256=sha(args.manifest), optimizer_rerun_count=len(args.optimize_indices),
          all_body_parameters_bitwise_equal=all(r['body_parameters_bitwise_equal'] for r in records),
          all_mesh_fields_bitwise_equal=True, local_surface_pass_count=sum(r['local_crossings_passed'] for r in records),
          scope='Added optimizer rerun on selected frozen automatic inputs; all final meshes independently rebuilt and reloaded; networks and upstream fitting not rerun', frames=records))


if __name__ == '__main__':
    main()
