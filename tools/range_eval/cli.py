# Copyright (c) OpenMMLab. All rights reserved.
"""CLI entry point for the per-range stratified nuScenes evaluation.

Usage:
    python -m tools.range_eval.cli \\
        --predictions /path/to/results_nusc.json \\
        --dataroot /data/nuscenes \\
        --split val \\
        --version v1.0-trainval \\
        --range-bins "0,30 30,50 50,100" \\
        --output-dir ./eval_outputs/run_001 \\
        --class-range-override 100.0
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

DEFAULT_PREDICTIONS = (
    '/ISFusion-claude-a30-gpu-compatibility-mrHY3/work_dirs/'
    'results_nusc/pts_bbox/results_nusc.json'
)


def _parse_range_bins(spec: str) -> List[Tuple[float, float]]:
    """Parse "0,30 30,50 50,100" into [(0.0, 30.0), (30.0, 50.0), (50.0, 100.0)]."""
    bins: List[Tuple[float, float]] = []
    for token in spec.split():
        try:
            lo_s, hi_s = token.split(',')
            lo = float(lo_s)
            hi = float(hi_s)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f'invalid range bin {token!r}; expected "r_min,r_max"') from e
        if hi <= lo or lo < 0:
            raise argparse.ArgumentTypeError(
                f'invalid range bin {token!r}; require 0 <= r_min < r_max')
        bins.append((lo, hi))
    if not bins:
        raise argparse.ArgumentTypeError('--range-bins must contain at least one bin')
    return bins


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Per-range, per-class stratified nuScenes detection evaluation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--predictions', type=str, default=DEFAULT_PREDICTIONS,
        help='Path to results_nusc.json (nuScenes detection format).')
    parser.add_argument(
        '--dataroot', type=str, required=True,
        help='nuScenes data root directory.')
    parser.add_argument(
        '--split', type=str, default='val',
        help='Eval split (val, mini_val, test).')
    parser.add_argument(
        '--version', type=str, default='v1.0-trainval',
        help='nuScenes dataset version.')
    parser.add_argument(
        '--range-bins', type=_parse_range_bins, default='0,30 30,50 50,100',
        help='Space-separated "r_min,r_max" pairs.')
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory where summary.{json,csv,md} are written.')
    parser.add_argument(
        '--class-range-override', type=float, default=100.0,
        help='Uniform class_range applied to all classes, in meters.')
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-bin progress logging.')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    predictions_path = Path(args.predictions)
    dataroot_path = Path(args.dataroot)
    output_dir = Path(args.output_dir)

    if not predictions_path.is_file():
        print(f'ERROR: predictions file not found: {predictions_path}', file=sys.stderr)
        return 2
    if not dataroot_path.is_dir():
        print(f'ERROR: dataroot directory not found: {dataroot_path}', file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # Imports deferred so that --help and arg validation work even if
    # nuscenes-devkit is not installed.
    from nuscenes import NuScenes

    from tools.range_eval.range_eval import RangeStratifiedEval
    from tools.range_eval.reporters import (write_csv, write_json_summary,
                                            write_markdown_report)

    if verbose:
        print(f'[range_eval] initializing NuScenes(version={args.version}, '
              f'dataroot={dataroot_path}); this can take 30-60s')
    nusc = NuScenes(version=args.version, dataroot=str(dataroot_path), verbose=verbose)

    evaluator = RangeStratifiedEval(
        nusc=nusc,
        predictions_path=str(predictions_path),
        range_bins=args.range_bins,
        class_range_override=args.class_range_override,
        split=args.split,
        verbose=verbose,
    )
    results = evaluator.run()

    json_path = output_dir / 'summary.json'
    csv_path = output_dir / 'summary.csv'
    md_path = output_dir / 'summary.md'
    write_json_summary(results, json_path)
    write_csv(results, csv_path)
    write_markdown_report(results, md_path)

    if verbose:
        print()
        print(f'[range_eval] wrote {json_path}')
        print(f'[range_eval] wrote {csv_path}')
        print(f'[range_eval] wrote {md_path}')
        print()
        print(f'overall mAP = {results["overall"]["mAP"]:.4f}')
        print(f'overall NDS = {results["overall"]["NDS"]:.4f}')
        for label, section in results['per_range'].items():
            n_gt_total = sum(section['n_gt'].values())
            print(f'  bin {label}: mAP = {section["mAP"]:.4f} '
                  f'(GT total = {n_gt_total})')
        if results['warnings']:
            print()
            print(f'[range_eval] {len(results["warnings"])} warning(s):')
            for w in results['warnings']:
                print(f'  - {w}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
