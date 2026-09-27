"""Strict JSON/NPZ I/O and reproducibility metadata."""
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    for key, value in result.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError('Nonfinite NPZ field: ' + key)
    return result


def arr(tensor):
    return tensor.detach().cpu().numpy()


def rmse(prediction, target):
    """Root mean squared 2D Euclidean distance, not coordinate-wise RMSE."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError('Expected matching N x 2 arrays')
    if not len(prediction):
        return None
    return float(np.sqrt(np.mean(np.sum((prediction - target)**2, axis=1))))
