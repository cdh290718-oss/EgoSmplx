"""Decode evenly spaced video frames; do not claim stereo synchronization."""
from pathlib import Path
import re
import cv2
from egosmplx_pipeline.io import write, sha


def sample_video(video, calibration, camera, count, output):
    if count < 1 or not re.fullmatch('[A-Za-z0-9_-]+', camera):
        raise ValueError('Positive frame count and safe camera name required')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    cap = cv2.VideoCapture(str(video))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or n < count or fps <= 0:
            raise ValueError('Video cannot provide requested frames')
        output.mkdir(parents=True)
        rows = []
        for k in range(count):
            index = min(n-1, int((k+.5) * n/count))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError('Failed decoding frame ' + str(index))
            identifier = camera + '_f' + str(index).zfill(8)
            target = output / (identifier + '.png')
            if not cv2.imwrite(str(target), image):
                raise IOError(target)
            rows.append(dict(id=identifier, camera=camera, image=target.name,
                             calibration=str(Path(calibration).resolve()), image_sha256=sha(target),
                             frame_index_zero_based=index, approximate_video_seconds=index/fps))
        write(output / 'manifest.json', dict(frames=rows, exact_stereo_synchronization_claimed=False))
        return output / 'manifest.json'
    finally:
        cap.release()
