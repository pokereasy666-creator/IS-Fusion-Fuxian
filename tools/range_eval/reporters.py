# Copyright (c) OpenMMLab. All rights reserved.
"""Output formatters for the per-range evaluation summary dict.

Three independent functions, each consuming the dict returned by
``RangeStratifiedEval.run()`` and writing one file format.
"""
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

UNRELIABLE_THRESHOLD = 50


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _fmt_num(x: Any, ndigits: int = 4) -> str:
    if _is_nan(x):
        return 'n/a'
    if isinstance(x, float):
        return f'{x:.{ndigits}f}'
    return str(x)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NaN floats to the string 'NaN' and Path to str."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return 'NaN'
    return obj


def write_json_summary(results: Dict, output_path: Path) -> None:
    """Standard json.dump with indent=2.

    NaN floats are encoded as the string ``"NaN"`` to keep the file strictly
    valid JSON; consumers should treat that value as missing data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sanitized = _sanitize_for_json(results)
    with open(output_path, 'w') as f:
        json.dump(sanitized, f, indent=2, allow_nan=False)


def _csv_rows_for_section(section_name: str, section: Dict, classes: Iterable[str]):
    n_gt = section.get('n_gt', {})
    n_pred = section.get('n_pred', {})
    per_class_ap = section.get('per_class_AP', {})
    rows = []
    for cls in classes:
        gt = int(n_gt.get(cls, 0))
        pred = int(n_pred.get(cls, 0))
        ap = per_class_ap.get(cls, float('nan'))
        rows.append({
            'class': cls,
            'range': section_name,
            'AP': 'n/a' if _is_nan(ap) else f'{ap:.4f}',
            'n_gt': gt,
            'n_pred': pred,
            'unreliable': gt < UNRELIABLE_THRESHOLD,
        })
    return rows


def write_csv(results: Dict, output_path: Path) -> None:
    """Long-format CSV. Columns: class, range, AP, n_gt, n_pred, unreliable.

    'range' is one of: 'overall', '0-30', '30-50', '50-100' (or whatever bins
    were used). 'unreliable' is True when n_gt < 50 for that (class, range) cell.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes: List[str] = list(results['overall']['per_class_AP'].keys())

    rows: List[Dict] = []
    rows.extend(_csv_rows_for_section('overall', results['overall'], classes))
    for label, section in results.get('per_range', {}).items():
        rows.extend(_csv_rows_for_section(label, section, classes))

    fieldnames = ['class', 'range', 'AP', 'n_gt', 'n_pred', 'unreliable']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(results: Dict, output_path: Path) -> None:
    """Human-readable Markdown with three tables:

    1. Overall summary: mAP, NDS, per-class AP (single-row table).
    2. Per-range summary: range x {mAP, total_gt_boxes, total_predictions}.
    3. Per-class x per-range AP matrix. Append '*' to cells where n_gt < 50.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overall = results['overall']
    per_range = results.get('per_range', {})
    classes: List[str] = list(overall['per_class_AP'].keys())
    cfg = results.get('config', {})

    lines: List[str] = []
    lines.append('# Range-Stratified nuScenes Detection Evaluation\n')

    lines.append('## Configuration\n')
    lines.append(f'- predictions: `{cfg.get("predictions_path", "?")}`')
    lines.append(f'- split: `{cfg.get("split", "?")}`')
    lines.append(f'- version: `{cfg.get("version", "?")}`')
    lines.append(f'- range_bins: `{cfg.get("range_bins", [])}`')
    lines.append(f'- class_range_override: `{cfg.get("class_range_override", "?")}m`')
    lines.append(f'- num_samples: `{cfg.get("num_samples", "?")}`')
    lines.append(f'- evaluated_at: `{cfg.get("evaluated_at", "?")}`\n')

    # Table 1: overall summary.
    lines.append('## Overall summary\n')
    overall_headers = ['mAP', 'NDS'] + classes
    overall_row = [_fmt_num(overall['mAP']), _fmt_num(overall['NDS'])]
    for cls in classes:
        overall_row.append(_fmt_num(overall['per_class_AP'].get(cls, float('nan'))))
    lines.append('| ' + ' | '.join(overall_headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(overall_headers)) + ' |')
    lines.append('| ' + ' | '.join(overall_row) + ' |\n')

    # Table 2: per-range summary.
    lines.append('## Per-range summary\n')
    lines.append('| range | mAP | total_gt_boxes | total_predictions |')
    lines.append('| --- | --- | --- | --- |')
    for label, section in per_range.items():
        total_gt = sum(section.get('n_gt', {}).values())
        total_pred = sum(section.get('n_pred', {}).values())
        lines.append(
            f'| {label} | {_fmt_num(section["mAP"])} | {total_gt} | {total_pred} |')
    lines.append('')

    # Table 3: per-class x per-range AP matrix.
    lines.append('## Per-class x per-range AP\n')
    range_labels = list(per_range.keys())
    header = ['class'] + range_labels
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    any_unreliable = False
    for cls in classes:
        row = [cls]
        for label in range_labels:
            section = per_range[label]
            ap = section['per_class_AP'].get(cls, float('nan'))
            n_gt = int(section.get('n_gt', {}).get(cls, 0))
            cell = _fmt_num(ap)
            if n_gt < UNRELIABLE_THRESHOLD:
                cell = cell + '*'
                any_unreliable = True
            row.append(cell)
        lines.append('| ' + ' | '.join(row) + ' |')

    if any_unreliable:
        lines.append('')
        lines.append(
            '\\* fewer than 50 GT instances; AP estimate is unreliable')

    warnings = results.get('warnings', [])
    if warnings:
        lines.append('')
        lines.append('## Warnings\n')
        for w in warnings:
            lines.append(f'- {w}')

    coverage = results.get('coverage')
    if coverage:
        lines.append('')
        lines.append('## Coverage\n')
        lines.append(f'- predictions in any bin: {coverage["predictions_in_any_bin"]}')
        lines.append(f'- predictions outside all bins: '
                     f'{coverage["predictions_outside_all_bins"]}')
        lines.append(f'- GT in any bin: {coverage["gt_boxes_in_any_bin"]}')
        lines.append(f'- GT outside all bins: '
                     f'{coverage["gt_boxes_outside_all_bins"]}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
