# Copyright (c) OpenMMLab. All rights reserved.
"""End-to-end smoke test on v1.0-mini.

For each GT box in mini-val, generate a synthetic prediction at the same global
location with score 0.99 and the correct class. Run the full pipeline and
assert mAP > 0.95 (since predictions are near-perfect).

Skips automatically if NUSCENES_DATAROOT is not set or the path doesn't exist.

Run with:
    NUSCENES_DATAROOT=/data/nuscenes pytest tools/range_eval/tests/test_eval_smoke.py -v
"""
import json
import os
import tempfile
from pathlib import Path

import pytest


# attribute_name conventions per nuScenes detection challenge.
# barrier and traffic_cone have no attributes (empty string).
DEFAULT_ATTRIBUTES_PER_CLASS = {
    'car': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.parked',
    'trailer': 'vehicle.parked',
    'construction_vehicle': 'vehicle.parked',
    'pedestrian': 'pedestrian.standing',
    'bicycle': 'cycle.without_rider',
    'motorcycle': 'cycle.without_rider',
    'barrier': '',
    'traffic_cone': '',
}


def _dataroot():
    p = os.environ.get('NUSCENES_DATAROOT')
    if not p or not Path(p).is_dir():
        pytest.skip('NUSCENES_DATAROOT env var unset or path does not exist; '
                    'set it to a directory containing v1.0-mini.')
    return p


def _has_nuscenes():
    try:
        import nuscenes  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_nuscenes(), reason='nuscenes-devkit not installed')


def _build_synthetic_predictions(nusc, mini_split: str = 'mini_val') -> dict:
    """Build a results_nusc.json-format dict with one prediction per GT box."""
    from nuscenes.eval.common.loaders import load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox

    gt_boxes = load_gt(nusc, mini_split, DetectionBox, verbose=False)

    results = {}
    for sample_token in gt_boxes.sample_tokens:
        per_sample = []
        for gt in gt_boxes[sample_token]:
            cls = gt.detection_name
            attribute_name = DEFAULT_ATTRIBUTES_PER_CLASS.get(cls, '')
            per_sample.append({
                'sample_token': gt.sample_token,
                'translation': list(gt.translation),
                'size': list(gt.size),
                'rotation': list(gt.rotation),
                'velocity': list(gt.velocity[:2]) if hasattr(gt, 'velocity') else [0.0, 0.0],
                'detection_name': cls,
                'detection_score': 0.99,
                'attribute_name': attribute_name,
            })
        results[sample_token] = per_sample

    return {
        'meta': {
            'use_camera': True, 'use_lidar': True,
            'use_radar': False, 'use_map': False, 'use_external': False,
        },
        'results': results,
    }


def test_full_pipeline_on_mini():
    dataroot = _dataroot()

    from nuscenes import NuScenes

    from tools.range_eval.range_eval import RangeStratifiedEval
    from tools.range_eval.reporters import (write_csv, write_json_summary,
                                            write_markdown_report)

    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)

    synthetic = _build_synthetic_predictions(nusc, mini_split='mini_val')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        preds_path = tmp / 'results_nusc.json'
        with open(preds_path, 'w') as f:
            json.dump(synthetic, f)

        evaluator = RangeStratifiedEval(
            nusc=nusc,
            predictions_path=str(preds_path),
            range_bins=[(0.0, 30.0), (30.0, 50.0), (50.0, 100.0)],
            class_range_override=100.0,
            split='mini_val',
            verbose=False,
        )
        results = evaluator.run()

        assert results['overall']['mAP'] > 0.95, (
            f'expected mAP > 0.95 with synthetic perfect predictions; '
            f'got {results["overall"]["mAP"]:.4f}')

        for label, section in results['per_range'].items():
            n_gt_total = sum(section['n_gt'].values())
            assert n_gt_total > 0, f'range bin {label} has zero GT total'

        out_dir = tmp / 'out'
        out_dir.mkdir()
        write_json_summary(results, out_dir / 'summary.json')
        write_csv(results, out_dir / 'summary.csv')
        write_markdown_report(results, out_dir / 'summary.md')

        assert (out_dir / 'summary.json').is_file()
        assert (out_dir / 'summary.csv').is_file()
        assert (out_dir / 'summary.md').is_file()

        with open(out_dir / 'summary.json') as f:
            loaded = json.load(f)
        for key in ('config', 'overall', 'per_range', 'coverage', 'warnings'):
            assert key in loaded, f'summary.json missing key: {key}'
        assert 'mAP' in loaded['overall']
        assert 'NDS' in loaded['overall']
        assert 'per_class_AP' in loaded['overall']
