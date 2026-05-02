# Range-stratified nuScenes detection evaluation

Per-range, per-class mAP/AP for nuScenes 3D detection results. Wraps
`nuscenes.eval.detection.evaluate.DetectionEval` to slice the eval set into
ego-distance bins (e.g. `0-30m`, `30-50m`, `50-100m`) so you can see how
detection quality degrades with range.

The standard nuScenes evaluator does not support per-range eval natively, and
its built-in per-class range caps (`car=50m`, `pedestrian=40m`,
`traffic_cone=30m`, etc.) make a `50-100m` bin empty for every class. This tool
overrides `class_range` to a uniform value (default `100m`) so each bin sees
comparable data.

## Requirements

- `nuscenes-devkit` (already installed in the IS-Fusion environment).
- nuScenes data root with `v1.0-trainval` (for full val) or `v1.0-mini` (for
  the smoke test).
- A predictions file in standard nuScenes detection format
  (`results_nusc.json`).

## Quick start

```bash
python -m tools.range_eval.cli \
    --predictions /ISFusion-claude-a30-gpu-compatibility-mrHY3/work_dirs/results_nusc/pts_bbox/results_nusc.json \
    --dataroot /data/nuscenes \
    --split val \
    --version v1.0-trainval \
    --range-bins "0,30 30,50 50,100" \
    --output-dir ./eval_outputs/run_001 \
    --class-range-override 100.0
```

Writes:

- `summary.json` — full machine-readable summary.
- `summary.csv` — long-format per (class, range) rows.
- `summary.md` — human-readable Markdown report.

## Validate before trusting output

Run the gating sanity check first. It compares one wide range bin against the
standard `DetectionEval` (with a matching `class_range` override) and asserts
they agree to within tolerance:

```bash
python -m tools.range_eval.validate_against_standard \
    --predictions /ISFusion-claude-a30-gpu-compatibility-mrHY3/work_dirs/results_nusc/pts_bbox/results_nusc.json \
    --dataroot /data/nuscenes
```

Tolerances:

- overall mAP: `1e-3`
- per-class AP: `1e-2`
- fail if overall exceeds OR more than one class exceeds.

If this fails, the bug is in the wrapper (filter, override, or
`sample_tokens` reset). Do not trust per-range numbers until this passes.

## Run tests

```bash
# Unit tests for the filter (no data, no devkit required):
python -m pytest tools/range_eval/tests/test_filters.py -v

# End-to-end smoke test on v1.0-mini (requires data):
NUSCENES_DATAROOT=/data/nuscenes \
    python -m pytest tools/range_eval/tests/test_eval_smoke.py -v
```

## Why per-range?

Lidar-camera fusion gains are typically range-dependent: cameras help
disambiguate distant objects where lidar returns are sparse, while lidar
dominates close-range. A single overall mAP averages these regimes and hides
the actual fusion contribution. The `0-30 / 30-50 / 50-100` split matches the
informal convention used in recent IS-Fusion-style ablation reports.

## Output schema (summary.json)

```json
{
  "config": {
    "predictions_path": "...",
    "split": "val",
    "version": "v1.0-trainval",
    "range_bins": [[0, 30], [30, 50], [50, 100]],
    "class_range_override": 100.0,
    "evaluated_at": "ISO 8601",
    "num_samples": 6019
  },
  "overall": {
    "mAP": 0.6744,
    "NDS": 0.7029,
    "per_class_AP": {"car": 0.880, ...},
    "per_class_AP_per_dist": {"car": {"0.5": ..., "1.0": ..., "2.0": ..., "4.0": ...}, ...},
    "n_gt": {...},
    "n_pred": {...}
  },
  "per_range": {
    "0-30": {"mAP": ..., "per_class_AP": ..., "per_class_AP_per_dist": ..., "n_gt": ..., "n_pred": ...},
    "30-50": {...},
    "50-100": {...}
  },
  "coverage": {
    "predictions_in_any_bin": ...,
    "predictions_outside_all_bins": ...,
    "gt_boxes_in_any_bin": ...,
    "gt_boxes_outside_all_bins": ...
  },
  "warnings": ["..."]
}
```

NaN floats are encoded as the string `"NaN"` so the file remains strictly
valid JSON. NDS is reported only for the `overall` row — for range-filtered
subsets it is interpretively muddy because TP-error metrics aren't comparable
across truncated subsets.

## Notes

- Translation in `results_nusc.json` is in the **global** frame, not the ego
  frame. Filtering uses `box.ego_dist` set by
  `nuscenes.eval.detection.utils.add_center_dist`. A defensive assertion
  panics if any `ego_dist` exceeds 1000m, which would indicate the
  global-frame translation was used by mistake.
- Range bins are half-open: `[r_min, r_max)`. A box at exactly `30.0m` falls
  into `[30, 50)`, not `[0, 30)`.
- Empty (class, range) cell: AP is reported as `NaN` (distinguishing "no
  data" from "perfect failure"). Cells with `n_gt < 50` are flagged
  `unreliable` in CSV and starred in Markdown.
