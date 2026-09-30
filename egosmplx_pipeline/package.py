"""Package a verified run, its three-stage results and reconstruction inputs.

This exports a new bundle; it does not overwrite or impersonate the historical
session_hand6 ZIP. Network weights and segmentation probability caches stay out.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from egosmplx_pipeline.io import read, sha
from egosmplx_pipeline.configuration import PACKAGE_ROOT
from egosmplx_pipeline.profiles import SUPPORTED


def package(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if read(run / 'run_state.json')['state'] != 'complete':
        raise ValueError('Only complete, verified runs can be packaged')
    if destination.exists():
        raise FileExistsError(destination)
    sources = {}
    for path in sorted((run / 'frames').glob('*/validation.json')):
        report = read(path)
        if not report.get('passed') or report.get('pipeline_profile') not in SUPPORTED or report.get('pipeline_profile') != read(run / 'summary.json')['pipeline_profile']:
            raise ValueError('Frame is not a verified reference-profile result')
        frame = path.parent
        for name, digest in report['artifact_sha256'].items():
            candidate = (frame / name).resolve()
            if frame not in candidate.parents or sha(candidate) != digest:
                raise ValueError('Frame artifact changed: ' + name)
        identifier = report['id']
        if frame.name != identifier:
            raise ValueError('Frame ID mismatch')
        row = read(run / 'jobs' / (identifier + '.json'))['row']
        if sha(row['image']) != report['input_image_sha256'] or sha(row['calibration']) != report['calibration_sha256']:
            raise ValueError('Original image or calibration changed')
        prefix = 'frames/' + identifier + '/'
        sources[prefix + 'input' + Path(row['image']).suffix.lower()] = Path(row['image'])
        sources[prefix + 'calibration.json'] = Path(row['calibration'])
        sources[prefix + 'automatic_observations/sapiens2_keypoints.json'] = Path(row['predictions']['sapiens'])
        for stage in ['raw', 'body', 'guided_body', 'fused']:
            if stage == 'guided_body' and not (frame / stage).exists():
                continue
            for asset in sorted((frame / stage).iterdir()):
                if asset.is_file():
                    sources[prefix + stage + '/' + asset.name] = asset
        sources[prefix + 'validation.json'] = path
    if len(list((run / 'frames').glob('*/validation.json'))) != read(run / 'summary.json')['frame_count']:
        raise ValueError('Run frame count differs from summary')
    for name in ['index.html', 'contact_sheet.jpg']:
        sources['comparison/' + name] = run / name
    for path in run.glob('*_comparison.jpg'):
        sources['comparison/' + path.name] = path
    sources['summary.json'] = run / 'summary.json'
    sources['REFERENCE_PIPELINE_zh.md'] = PACKAGE_ROOT / 'docs/REFERENCE_PIPELINE_zh.md'
    sources['CURRENT_PIPELINE_zh.md'] = PACKAGE_ROOT / 'docs/CURRENT_PIPELINE_zh.md'
    for path in sorted((PACKAGE_ROOT / 'egosmplx_pipeline').rglob('*.py')):
        sources['scripts/' + str(path.relative_to(PACKAGE_ROOT))] = path
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    # Exclusive creation prevents an existing historical bundle from being replaced.
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(sources.items()):
            data = path.read_bytes()
            archive.writestr(name, data)
            manifest[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        archive.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Exported ZIP failed CRC validation')
    return dict(path=str(destination), sha256=sha(destination), manifest_members=len(manifest),
                contains_weights=False, contains_full_segmentation_probabilities=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.run, args.archive), indent=2))


if __name__ == '__main__':
    main()
