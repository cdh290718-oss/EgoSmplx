"""Command-line interface: check, run, sample."""
import argparse
from pathlib import Path
from egosmplx_pipeline.configuration import load_config, preflight
from egosmplx_pipeline.io import write


def main():
    parser = argparse.ArgumentParser(description='EgoSmplx V1.0 automatic fisheye pose pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['check', 'run']:
        cmd = sub.add_parser(name)
        cmd.add_argument('--config', type=Path, required=True)
        cmd.add_argument('--cached-predictions', action='store_true')
        if name == 'run':
            cmd.add_argument('--resume', action='store_true')
    sample = sub.add_parser('sample')
    sample.add_argument('--video', type=Path, required=True)
    sample.add_argument('--calibration', type=Path, required=True)
    sample.add_argument('--camera', required=True)
    sample.add_argument('--count', type=int, default=4)
    sample.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        config, rows = load_config(args.config)
        print(preflight(config, rows, args.cached_predictions))
    elif args.command == 'run':
        from egosmplx_pipeline.pipeline import run
        print(run(args.config, args.cached_predictions, args.resume))
    else:
        from egosmplx_pipeline.sampling import sample_video
        print(sample_video(args.video, args.calibration, args.camera, args.count, args.output))


if __name__ == '__main__':
    main()
