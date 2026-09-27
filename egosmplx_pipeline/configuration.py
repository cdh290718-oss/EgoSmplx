"""Portable JSON configuration and fail-fast input checks."""
import os
import re
from pathlib import Path
from egosmplx_pipeline.io import read, sha

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def expand(value, base):
    if isinstance(value, dict):
        return {k: expand(v, base) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, base) for v in value]
    if isinstance(value, str):
        value = value.replace('${PACKAGE_ROOT}', str(PACKAGE_ROOT))
        value = os.path.expandvars(value)
        if re.search(r'\$\{[^}]+\}', value):
            raise ValueError('Unresolved environment variable: ' + value)
    return value


def absolute(path, base):
    p = Path(path).expanduser()
    return str((base / p).resolve() if not p.is_absolute() else p.resolve())


def load_config(path):
    path = Path(path).resolve()
    config = expand(read(path), path.parent)
    for key, value in config['paths'].items():
        if value:
            config['paths'][key] = absolute(value, path.parent)
    config['manifest'] = absolute(config['manifest'], path.parent)
    config['output'] = absolute(config['output'], path.parent)
    fit = config.setdefault('fitting', {})
    for key, default in [('rigid_steps', 1800), ('pose_steps', 1600), ('silhouette_weight', .08)]:
        fit.setdefault(key, default)
    if min(fit['rigid_steps'], fit['pose_steps']) < 1 or fit['silhouette_weight'] < 0:
        raise ValueError('Invalid fitting settings')
    manifest = read(config['manifest'])
    frames = expand(manifest['frames'], Path(config['manifest']).parent)
    ids = set()
    for row in frames:
        identifier = row['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in ids:
            raise ValueError('Frame IDs must be unique safe filenames: ' + identifier)
        ids.add(identifier)
        for key in ['image', 'calibration']:
            row[key] = absolute(row[key], Path(config['manifest']).parent)
        for key, value in row.get('predictions', {}).items():
            row['predictions'][key] = absolute(value, Path(config['manifest']).parent)
    if not frames:
        raise ValueError('Empty input manifest')
    return config, frames


def preflight(config, frames, cached):
    """Check required files and image/observation provenance before any output."""
    import cv2
    required = ['body_python', 'egosmplx_repo', 'egosmplx_config']
    if not cached:
        required += ['pose_python', 'wilor_python', 'sapiens_repo', 'wilor_repo',
                     'egosmplx_checkpoint', 'pose_checkpoint', 'seg_checkpoint',
                     'wilor_checkpoint', 'wilor_config', 'wilor_detector']
    for key in required:
        if key not in config['paths'] or not Path(config['paths'][key]).exists():
            raise FileNotFoundError('Required configured path: ' + key)
    for row in frames:
        image = cv2.imread(row['image'])
        if image is None:
            raise ValueError('Unreadable image: ' + row['image'])
        calibration = read(row['calibration'])
        if calibration['size'] != list(image.shape[1::-1]):
            raise ValueError('Image and calibration sizes differ: ' + row['id'])
        if cached:
            for key in ['egosmplx', 'sapiens', 'segmentation', 'wilor']:
                if not Path(row.get('predictions', {}).get(key, '')).exists() or key not in row.get('predictions', {}):
                    raise FileNotFoundError(row['id'] + ': missing cached prediction ' + key)
            points = read(row['predictions']['sapiens'])
            if points['image_sha256'] != sha(row['image']):
                raise ValueError('Stale Sapiens2 observation: ' + row['id'])
            for name in ['labels.npy', 'probabilities.npz']:
                if not (Path(row['predictions']['segmentation']) / name).is_file():
                    raise FileNotFoundError('Incomplete segmentation: ' + row['id'])
    if not cached:
        for name in ['pose', 'seg']:
            expected = config.get('checkpoint_sha256', {}).get(name)
            if expected is None or not re.fullmatch('[a-f0-9]{64}', expected):
                raise ValueError('Configure checkpoint_sha256.' + name)
    return dict(passed=True, frame_count=len(frames), cached_predictions=cached,
                scope='paths, input sizes, cache image hashes; models are loaded strictly by workers')
