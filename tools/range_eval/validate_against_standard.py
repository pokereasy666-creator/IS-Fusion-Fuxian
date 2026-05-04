# Copyright (c) OpenMMLab. All rights reserved.
"""Sanity check: running RangeStratifiedEval with one wide range bin should
match the standard nuScenes DetectionEval (modulo per-class range cap).

Both runs use class_range = 200.0 for every class, so the only difference
between them is whether range filtering happens. Since (0, 200) ego-distance
filter keeps every real box, the mAPs and per-class APs should match.

Tolerances:
  - overall mAP: 1e-3 (strict)
  - per-class AP: 1e-2 (allow more slack; small numerical reorderings in
    accumulate() can flip stable sorts)
  - fail only if overall mAP exceeds tolerance OR more than one class exceeds.

This is the gating test before trusting any other output. Run it before
running the full pipeline on the actual predictions file.
"""
import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict

OVERALL_MAP_TOL = 1e-3
PER_CLASS_AP_TOL = 1e-2
MAX_PER_CLASS_VIOLATIONS = 1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate RangeStratifiedEval against standard DetectionEval.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--predictions', type=str, required=True,
        help='Path to results_nusc.json.')
    parser.add_argument(
        '--dataroot', type=str, required=True,
        help='nuScenes data root.')
    parser.add_argument(
        '--split', type=str, default='val',
        help='Eval split.')
    parser.add_argument(
        '--version', type=str, default='v1.0-trainval',
        help='nuScenes dataset version.')
    parser.add_argument(
        '--class-range-override', type=float, default=200.0,
        help='Uniform class_range applied to both runs (must match for parity).')
    return parser.parse_args(argv)


def _run_standard(nusc, predictions_path: str, split: str,
                  class_range_override: float) -> Dict:
    """Run the unmodified DetectionEval with class_range overridden to a uniform
    value, so it matches what RangeStratifiedEval will use.

    Reads results directly from the in-memory DetectionMetrics object via
    get_label_ap, mirroring the wrapper's pattern. This avoids depending on
    the metrics_summary.json key layout, which differs subtly across devkit
    versions.
    """
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    cfg = config_factory('detection_cvpr_2019')
    for c in cfg.class_range:
        cfg.class_range[c] = class_range_override

    with tempfile.TemporaryDirectory() as tmp:
        evaluator = DetectionEval(
            nusc, config=cfg, result_path=predictions_path,
            eval_set=split, output_dir=tmp, verbose=False)
        metrics, _metric_data_list = evaluator.evaluate()

    per_class_ap: Dict[str, float] = {}
    for cls in cfg.class_range:
        ap_values = [float(metrics.get_label_ap(cls, dist_th))
                     for dist_th in cfg.dist_ths]
        per_class_ap[cls] = float(sum(ap_values) / len(ap_values))

    return {
        'mAP': float(metrics.mean_ap),
        'NDS': float(metrics.nd_score),
        'per_class_AP': per_class_ap,
    }


def _run_range_eval(nusc, predictions_path: str, split: str,
                    class_range_override: float) -> Dict:
    from tools.range_eval.range_eval import RangeStratifiedEval

    evaluator = RangeStratifiedEval(
        nusc=nusc,
        predictions_path=predictions_path,
        range_bins=[(0.0, 200.0)],
        class_range_override=class_range_override,
        split=split,
        verbose=False,
    )
    results = evaluator.run()
    return {
        'mAP': results['overall']['mAP'],
        'NDS': results['overall']['NDS'],
        'per_class_AP': results['overall']['per_class_AP'],
    }


def _diff_report(standard: Dict, range_eval: Dict) -> Dict:
    map_diff = abs(standard['mAP'] - range_eval['mAP'])
    nds_diff = abs(standard['NDS'] - range_eval['NDS'])
    per_class_diffs: Dict[str, float] = {}
    for cls in standard['per_class_AP']:
        s = standard['per_class_AP'][cls]
        r = range_eval['per_class_AP'].get(cls, float('nan'))
        per_class_diffs[cls] = abs(s - r)
    return {
        'mAP_diff': map_diff,
        'NDS_diff': nds_diff,
        'per_class_diffs': per_class_diffs,
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    if not Path(args.predictions).is_file():
        print(f'ERROR: predictions file not found: {args.predictions}',
              file=sys.stderr)
        return 2
    if not Path(args.dataroot).is_dir():
        print(f'ERROR: dataroot not found: {args.dataroot}', file=sys.stderr)
        return 2

    from nuscenes import NuScenes

    print(f'[validate] initializing NuScenes(version={args.version})...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f'[validate] running standard DetectionEval '
          f'(class_range={args.class_range_override} for all classes)')
    standard = _run_standard(nusc, args.predictions, args.split,
                             args.class_range_override)

    print(f'[validate] running RangeStratifiedEval bins=[(0, 200)] '
          f'(class_range_override={args.class_range_override})')
    range_eval = _run_range_eval(nusc, args.predictions, args.split,
                                 args.class_range_override)

    diff = _diff_report(standard, range_eval)

    print()
    print('=' * 60)
    print(f'{"metric":24s}  {"standard":>10s}  {"range_eval":>10s}  {"diff":>10s}')
    print('-' * 60)
    print(f'{"overall mAP":24s}  {standard["mAP"]:10.4f}  '
          f'{range_eval["mAP"]:10.4f}  {diff["mAP_diff"]:10.4f}')
    print(f'{"overall NDS":24s}  {standard["NDS"]:10.4f}  '
          f'{range_eval["NDS"]:10.4f}  {diff["NDS_diff"]:10.4f}')
    for cls, d in sorted(diff['per_class_diffs'].items()):
        s = standard['per_class_AP'][cls]
        r = range_eval['per_class_AP'].get(cls, float('nan'))
        marker = '  <<' if d > PER_CLASS_AP_TOL else ''
        print(f'{("AP " + cls):24s}  {s:10.4f}  {r:10.4f}  {d:10.4f}{marker}')
    print('=' * 60)

    overall_fail = diff['mAP_diff'] > OVERALL_MAP_TOL
    per_class_violations = [
        cls for cls, d in diff['per_class_diffs'].items()
        if d > PER_CLASS_AP_TOL
    ]
    per_class_fail = len(per_class_violations) > MAX_PER_CLASS_VIOLATIONS

    if overall_fail or per_class_fail:
        print()
        print('FAIL')
        if overall_fail:
            print(f'  overall mAP diff {diff["mAP_diff"]:.4f} exceeds '
                  f'tolerance {OVERALL_MAP_TOL}')
        if per_class_fail:
            print(f'  {len(per_class_violations)} class(es) exceed '
                  f'per-class AP tolerance {PER_CLASS_AP_TOL}: '
                  f'{per_class_violations}')
        return 1

    print()
    print('PASS')
    if per_class_violations:
        print(f'  (note: {len(per_class_violations)} class within tolerance limit '
              f'{MAX_PER_CLASS_VIOLATIONS}: {per_class_violations})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
