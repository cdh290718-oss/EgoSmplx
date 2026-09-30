"""Sequential multi-environment controller with input hashing and verified resume."""
import copy
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from egosmplx_pipeline.configuration import PACKAGE_ROOT, load_config, preflight
from egosmplx_pipeline.io import read, write, sha
from egosmplx_pipeline.profiles import selected


def command(args, log, cwd, environment):
    print('RUN', log.name, flush=True)
    with log.open('w') as stream:
        process = subprocess.run(list(map(str, args)), cwd=str(cwd), env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError('Worker failed; inspect ' + str(log))


def environment(config):
    result = os.environ.copy()
    result.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                  CUDA_VISIBLE_DEVICES=str(config.get('gpu', '0')),
                  EGOSMPLX_CONFIG=config['paths']['egosmplx_config'],
                  EGOSMPLX_REPO_ROOT=config['paths']['egosmplx_repo'])
    result['PYTHONPATH'] = os.pathsep.join([str(PACKAGE_ROOT), config['paths'].get('wilor_repo', '')])
    return result


def observe(config, rows, output, env):
    """Invoke independent external networks. No manual annotation input."""
    p = config['paths']
    observations = output / 'observations'
    observations.mkdir(exist_ok=False)
    staged = observations / 'images'
    staged.mkdir()
    observed_rows = copy.deepcopy(rows)
    for row in observed_rows:
        image = Path(row['image'])
        target = staged / (row['id'] + image.suffix.lower())
        shutil.copy2(image, target)
        row['image'] = str(target)
    manifest = observations / 'manifest.json'
    write(manifest, dict(frames=observed_rows))
    for row in observed_rows:
        directory = observations / row['id']
        directory.mkdir()
        command([p['body_python'], '-m', 'egosmplx_pipeline.egosmplx_predict', '--image-dir', staged, '--image-glob', Path(row['image']).name,
                 '--fisheye-calibration', row['calibration'], '--config-path', p['egosmplx_config'],
                 '--checkpoint-path', p['egosmplx_checkpoint'], '--output-dir', directory / 'network',
                 '--smplx-input-mode', 'det_undistort_contiguous', '--contiguous-roi-mode', 'aspect_pad'],
                directory / 'network.log', p['egosmplx_repo'], env)
        candidates = sorted((directory / 'network').rglob(row['id'] + '*_smplx.npz'))
        # Some backend versions return the timestamp directory beside the request.
        if not candidates:
            candidates = sorted(directory.glob('network*/smplx/' + row['id'] + '*_smplx.npz'))
        if len(candidates) != 1:
            raise RuntimeError('Expected one wearer prediction, found ' + str(len(candidates)))
        args = [p['pose_python'], '-m', 'egosmplx_pipeline.sapiens_pose', '--repo-root', p['sapiens_repo'],
                '--image', row['image'], '--output-dir', directory / 'sapiens',
                '--checkpoint', p['pose_checkpoint'], '--checkpoint-sha256', config['checkpoint_sha256']['pose']]
        if p.get('sapiens_dependencies'):
            args += ['--dependencies', p['sapiens_dependencies']]
        command(args, directory / 'sapiens.log', PACKAGE_ROOT, env)
        row['predictions'] = dict(egosmplx=str(candidates[0]), sapiens=str(directory / 'sapiens/keypoints.json'))
    command([p['wilor_python'], '-m', 'egosmplx_pipeline.wilor_predict', '--manifest', manifest,
             '--output-dir', observations / 'wilor', '--checkpoint', p['wilor_checkpoint'],
             '--config', p['wilor_config'], '--detector', p['wilor_detector']],
            observations / 'wilor.log', p['wilor_repo'], env)
    args = [p['pose_python'], '-m', 'egosmplx_pipeline.sapiens_seg', '--repo-root', p['sapiens_repo'],
            '--manifest', manifest, '--output-dir', observations / 'segmentation',
            '--checkpoint', p['seg_checkpoint'], '--checkpoint-sha256', config['checkpoint_sha256']['seg']]
    if p.get('sapiens_dependencies'):
        args += ['--dependencies', p['sapiens_dependencies']]
    command(args, observations / 'segmentation.log', PACKAGE_ROOT, env)
    for row in observed_rows:
        row['predictions'].update(wilor=str(observations / 'wilor/npz' / (row['id'] + '_wilor.npz')),
                                  segmentation=str(observations / 'segmentation/frames' / row['id']))
    write(output / 'automatic_manifest.json', dict(frames=observed_rows))
    return observed_rows


def fingerprint(config, rows):
    import json
    hashes = {}
    for row in rows:
        for key in ['image', 'calibration']:
            hashes[row['id'] + ':' + key] = sha(row[key])
        for key, value in row.get('predictions', {}).items():
            if key == 'segmentation':
                for name in ['labels.npy', 'probabilities.npz']:
                    hashes[row['id'] + ':' + name] = sha(Path(value) / name)
            else:
                hashes[row['id'] + ':' + key] = sha(value)
    code = {str(p.relative_to(PACKAGE_ROOT)): sha(p) for p in (PACKAGE_ROOT / 'egosmplx_pipeline').rglob('*.py')}
    return hashlib.sha256(json.dumps(dict(config=config, inputs=hashes, code=code), sort_keys=True).encode()).hexdigest()


def run(config_path, cached=False, resume=False):
    config, rows = load_config(config_path)
    check = preflight(config, rows, cached)
    output = Path(config['output'])
    if output.exists() and not resume:
        raise FileExistsError('Output exists; choose a new directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = environment(config)
        write(output / 'preflight.json', check)
        signature = fingerprint(config, rows)
        state_path = output / 'run_state.json'
        if state_path.exists() and read(state_path)['fingerprint'] != signature:
            raise ValueError('Inputs, code, or configuration changed; use a new output directory')
        write(state_path, dict(fingerprint=signature, state='running', cached_predictions=cached))
        if not cached:
            ready = output / 'automatic_manifest.json'
            if resume and ready.exists():
                rows = read(ready)['frames']
            else:
                if (output / 'observations').exists():
                    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
                    (output / 'observations').rename(output / ('incomplete_observations_' + stamp))
                rows = observe(config, rows, output, env)
        records = []
        for index, row in enumerate(rows):
            destination = output / 'frames' / row['id']
            validation = destination / 'validation.json'
            if resume and validation.exists():
                report = read(validation)
                valid = report.get('passed') and all((destination / f).is_file() and sha(destination / f) == h
                                                   for f, h in report['artifact_sha256'].items())
                if valid:
                    records.append(report)
                    print('VERIFIED_RESUME', row['id'], flush=True)
                    continue
            if destination.exists():
                stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
                destination.rename(destination.with_name(destination.name + '_incomplete_' + stamp))
            job_path = output / 'jobs' / (row['id'] + '.json')
            write(job_path, dict(config=config, row=row, output=str(destination)))
            command([config['paths']['body_python'], '-m', 'egosmplx_pipeline.frame', '--job', job_path],
                    output / (row['id'] + '.log'), PACKAGE_ROOT, env)
            records.append(read(validation))
            write(output / 'progress.json', dict(completed=index+1, total=len(rows), last_frame=row['id']))
        if not all(r.get('pipeline_profile') == selected(config) for r in records):
            job_path = output / 'jobs/reference.json'
            write(job_path, dict(config=config, rows=rows, output=str(output)))
            command([config['paths']['body_python'], '-m', 'egosmplx_pipeline.reference.pipeline', '--job', job_path],
                    output / 'reference.log', PACKAGE_ROOT, env)
            records = [read(output / 'frames' / row['id'] / 'validation.json') for row in rows]
        from egosmplx_pipeline.report import build
        build(output, records)
        write(state_path, dict(fingerprint=signature, state='complete', frame_count=len(records), cached_predictions=cached))
    return output
