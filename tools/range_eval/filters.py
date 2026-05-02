# Copyright (c) OpenMMLab. All rights reserved.
"""Range-based filters for nuScenes EvalBoxes.

The single function exported here, ``filter_eval_boxes_by_ego_dist``, slices
an EvalBoxes collection on each box's ``ego_dist`` attribute (the L2 distance
from ego pose to box center, set by
``nuscenes.eval.detection.utils.add_center_dist``).
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuscenes.eval.common.data_classes import EvalBoxes

EGO_DIST_PANIC_THRESHOLD = 1000.0


def filter_eval_boxes_by_ego_dist(eval_boxes, r_min: float, r_max: float):
    """Filter boxes to half-open interval [r_min, r_max) by ego_dist attribute.

    Requires that ``nuscenes.eval.detection.utils.add_center_dist`` has already
    been called on ``eval_boxes`` (so each box has an ``ego_dist`` attribute).

    Args:
        eval_boxes: EvalBoxes (or compatible duck-typed object) with
            ``sample_tokens``, ``__getitem__``, and ``add_boxes``.
        r_min: lower bound (inclusive).
        r_max: upper bound (exclusive).

    Returns:
        A fresh EvalBoxes-compatible object of the same type as the input.
        Does not mutate the input.

    Raises:
        ValueError: if any box has ``ego_dist`` greater than
            ``EGO_DIST_PANIC_THRESHOLD`` (1000m). This indicates the global-frame
            translation was used by mistake instead of an ego-relative distance.
        AttributeError or ValueError: if a box lacks the ``ego_dist`` attribute.
    """
    if r_min < 0.0 or r_max <= r_min:
        raise ValueError(
            f'invalid range bounds: r_min={r_min}, r_max={r_max} '
            f'(require 0 <= r_min < r_max)')

    out = type(eval_boxes)()

    for sample_token in eval_boxes.sample_tokens:
        kept = []
        for box in eval_boxes[sample_token]:
            ego_dist = getattr(box, 'ego_dist', None)
            if ego_dist is None:
                raise ValueError(
                    f'box for sample {sample_token} has no ego_dist attribute; '
                    'did you forget to call add_center_dist?')
            if ego_dist > EGO_DIST_PANIC_THRESHOLD:
                raise ValueError(
                    f'ego_dist={ego_dist:.1f} for sample {sample_token} exceeds '
                    f'{EGO_DIST_PANIC_THRESHOLD}m; this almost certainly means '
                    'distance was computed from global-frame translation '
                    'instead of from add_center_dist (ego-relative).')
            if r_min <= ego_dist < r_max:
                kept.append(box)
        if kept:
            out.add_boxes(sample_token, kept)

    return out
