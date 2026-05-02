# Copyright (c) OpenMMLab. All rights reserved.
"""Per-range stratified nuScenes detection evaluation.

Wraps ``nuscenes.eval.detection.evaluate.DetectionEval`` to produce per-range
mAP/AP. Predictions and GT are loaded once, annotated with ``ego_dist`` via
``add_center_dist``, and then evaluated per range bin by overriding the
evaluator's pred_boxes/gt_boxes attributes with filtered subsets.

Why this exists: standard nuScenes eval reports a single overall mAP and applies
per-class range caps (car=50m, pedestrian=40m, traffic_cone=30m, etc.) that make
a 50-100m bin empty for every class. We override class_range to a uniform value
so each bin sees comparable data.

NDS is reported only for the overall row. NDS aggregates true-positive errors
at recall thresholds tied to a fixed match-distance budget; truncating range
distorts this in non-comparable ways across bins.
"""
import math
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.range_eval.filters import filter_eval_boxes_by_ego_dist


def _bin_label(r_min: float, r_max: float) -> str:
    return f'{int(r_min)}-{int(r_max)}'


def _count_boxes_per_class(eval_boxes, classes: List[str]) -> Dict[str, int]:
    counts = {c: 0 for c in classes}
    for token in eval_boxes.sample_tokens:
        for box in eval_boxes[token]:
            name = getattr(box, 'detection_name', None)
            if name in counts:
                counts[name] += 1
    return counts


def _total_boxes(eval_boxes) -> int:
    return sum(len(eval_boxes[t]) for t in eval_boxes.sample_tokens)


class RangeStratifiedEval:
    """Wraps nuScenes DetectionEval to produce per-range stratified results.

    Loads predictions and GT once at construction, annotates with ``ego_dist``,
    then runs DetectionEval per range bin by overriding the evaluator's
    pred_boxes and gt_boxes attributes with filtered versions.
    """

    def __init__(self,
                 nusc,
                 predictions_path: str,
                 range_bins: List[Tuple[float, float]],
                 class_range_override: float = 100.0,
                 split: str = 'val',
                 verbose: bool = False):
        from nuscenes.eval.common.loaders import load_gt, load_prediction
        from nuscenes.eval.common.utils import filter_eval_boxes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.data_classes import DetectionBox
        from nuscenes.eval.detection.utils import add_center_dist

        predictions_path = str(Path(predictions_path).resolve())
        if not Path(predictions_path).is_file():
            raise FileNotFoundError(f'predictions file not found: {predictions_path}')

        self.nusc = nusc
        self.predictions_path = predictions_path
        self.range_bins = [tuple(b) for b in range_bins]
        self.class_range_override = float(class_range_override)
        self.split = split
        self.verbose = verbose

        self.cfg = config_factory('detection_cvpr_2019')
        self.classes = list(self.cfg.class_range.keys())

        if verbose:
            print(f'[range_eval] class_range_override = {self.class_range_override}m '
                  f'(applied to: {", ".join(self.classes)})')

        for c in self.cfg.class_range:
            self.cfg.class_range[c] = self.class_range_override

        if verbose:
            print(f'[range_eval] loading predictions from {self.predictions_path}')
        t0 = time.time()
        self.pred_boxes, self.meta = load_prediction(
            self.predictions_path,
            self.cfg.max_boxes_per_sample,
            DetectionBox,
            verbose=verbose,
        )

        if verbose:
            print(f'[range_eval] loading GT for split={self.split}')
        self.gt_boxes = load_gt(self.nusc, self.split, DetectionBox, verbose=verbose)

        if verbose:
            print('[range_eval] annotating boxes with ego_dist via add_center_dist')
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)

        if verbose:
            print('[range_eval] applying filter_eval_boxes (MIN_LIDAR_PTS + permissive class_range)')
        self.pred_boxes = filter_eval_boxes(
            self.nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        self.gt_boxes = filter_eval_boxes(
            self.nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        if verbose:
            print(f'[range_eval] init complete in {time.time() - t0:.1f}s; '
                  f'{_total_boxes(self.pred_boxes)} preds, '
                  f'{_total_boxes(self.gt_boxes)} GT, '
                  f'{len(self.gt_boxes.sample_tokens)} samples')

    def _evaluate_one(self, pred_filtered, gt_filtered, label: str) -> Dict:
        from nuscenes.eval.detection.evaluate import DetectionEval

        with tempfile.TemporaryDirectory() as tmp:
            evaluator = DetectionEval(
                self.nusc,
                config=self.cfg,
                result_path=self.predictions_path,
                eval_set=self.split,
                output_dir=tmp,
                verbose=False,
            )
            evaluator.pred_boxes = pred_filtered
            evaluator.gt_boxes = gt_filtered
            evaluator.sample_tokens = evaluator.gt_boxes.sample_tokens
            for c in evaluator.cfg.class_range:
                evaluator.cfg.class_range[c] = self.class_range_override

            metrics, _metric_data_list = evaluator.evaluate()

        n_gt = _count_boxes_per_class(gt_filtered, self.classes)
        n_pred = _count_boxes_per_class(pred_filtered, self.classes)

        per_class_ap: Dict[str, float] = {}
        per_class_ap_per_dist: Dict[str, Dict[str, float]] = {}
        for cls in self.classes:
            ap_per_dist: Dict[str, float] = {}
            for dist_th in self.cfg.dist_ths:
                ap_per_dist[str(dist_th)] = float(metrics.label_aps[cls][dist_th])
            per_class_ap_per_dist[cls] = ap_per_dist
            if n_gt[cls] == 0:
                per_class_ap[cls] = float('nan')
            else:
                per_class_ap[cls] = float(sum(ap_per_dist.values()) / len(ap_per_dist))

        valid_aps = [v for v in per_class_ap.values() if not math.isnan(v)]
        if valid_aps:
            mAP = float(sum(valid_aps) / len(valid_aps))
        else:
            mAP = float('nan')

        return {
            'label': label,
            'mAP': mAP,
            'NDS': float(metrics.nd_score),
            'per_class_AP': per_class_ap,
            'per_class_AP_per_dist': per_class_ap_per_dist,
            'n_gt': n_gt,
            'n_pred': n_pred,
        }

    def _compute_coverage(self) -> Dict[str, int]:
        """Count how many preds/GT fall in any user range bin vs outside all bins."""
        def in_any_bin(ego_dist: float) -> bool:
            for r_min, r_max in self.range_bins:
                if r_min <= ego_dist < r_max:
                    return True
            return False

        pred_in = pred_out = 0
        for tok in self.pred_boxes.sample_tokens:
            for box in self.pred_boxes[tok]:
                if in_any_bin(box.ego_dist):
                    pred_in += 1
                else:
                    pred_out += 1

        gt_in = gt_out = 0
        for tok in self.gt_boxes.sample_tokens:
            for box in self.gt_boxes[tok]:
                if in_any_bin(box.ego_dist):
                    gt_in += 1
                else:
                    gt_out += 1

        return {
            'predictions_in_any_bin': pred_in,
            'predictions_outside_all_bins': pred_out,
            'gt_boxes_in_any_bin': gt_in,
            'gt_boxes_outside_all_bins': gt_out,
        }

    def run(self) -> Dict:
        """Run evaluation across all bins. Returns the summary dict."""
        warnings: List[str] = []

        if self.verbose:
            print('[range_eval] === overall ===')
        t_overall = time.time()
        overall = self._evaluate_one(self.pred_boxes, self.gt_boxes, 'overall')
        if self.verbose:
            print(f'[range_eval] overall: mAP={overall["mAP"]:.4f} '
                  f'NDS={overall["NDS"]:.4f} '
                  f'n_pred={sum(overall["n_pred"].values())} '
                  f'n_gt={sum(overall["n_gt"].values())} '
                  f'({time.time() - t_overall:.1f}s)')

        per_range: Dict[str, Dict] = {}
        for (r_min, r_max) in self.range_bins:
            label = _bin_label(r_min, r_max)
            if self.verbose:
                print(f'[range_eval] === bin {label} ===')
            t_bin = time.time()

            pred_filtered = filter_eval_boxes_by_ego_dist(self.pred_boxes, r_min, r_max)
            gt_filtered = filter_eval_boxes_by_ego_dist(self.gt_boxes, r_min, r_max)

            n_pred_bin = _total_boxes(pred_filtered)
            n_gt_bin = _total_boxes(gt_filtered)
            if self.verbose:
                print(f'[range_eval] bin {label}: filtered {n_pred_bin} preds, '
                      f'{n_gt_bin} GT')

            result = self._evaluate_one(pred_filtered, gt_filtered, label)
            per_range[label] = {
                'mAP': result['mAP'],
                'per_class_AP': result['per_class_AP'],
                'per_class_AP_per_dist': result['per_class_AP_per_dist'],
                'n_gt': result['n_gt'],
                'n_pred': result['n_pred'],
            }

            for cls, count in result['n_gt'].items():
                if 0 < count < 50:
                    warnings.append(
                        f"class '{cls}' in range '{label}' has only {count} GT "
                        f"instances; AP unstable")
                elif count == 0:
                    warnings.append(
                        f"class '{cls}' in range '{label}' has 0 GT instances; "
                        f"AP reported as NaN")

            if self.verbose:
                print(f'[range_eval] bin {label}: mAP={result["mAP"]:.4f} '
                      f'({time.time() - t_bin:.1f}s)')

        coverage = self._compute_coverage()
        if coverage['predictions_outside_all_bins'] > 0:
            warnings.append(
                f"{coverage['predictions_outside_all_bins']} predictions fall "
                f"outside all configured range bins; consider widening the "
                f"outermost bin")

        if self.verbose:
            print(f'[range_eval] coverage: {coverage}')

        nusc_version = getattr(self.nusc, 'version', None)
        return {
            'config': {
                'predictions_path': self.predictions_path,
                'split': self.split,
                'version': nusc_version,
                'range_bins': [list(b) for b in self.range_bins],
                'class_range_override': self.class_range_override,
                'evaluated_at': datetime.now(timezone.utc).isoformat(),
                'num_samples': len(self.gt_boxes.sample_tokens),
            },
            'overall': {
                'mAP': overall['mAP'],
                'NDS': overall['NDS'],
                'per_class_AP': overall['per_class_AP'],
                'per_class_AP_per_dist': overall['per_class_AP_per_dist'],
                'n_gt': overall['n_gt'],
                'n_pred': overall['n_pred'],
            },
            'per_range': per_range,
            'coverage': coverage,
            'warnings': warnings,
        }
