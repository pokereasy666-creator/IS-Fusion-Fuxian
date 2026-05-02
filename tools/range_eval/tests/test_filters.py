# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for filter_eval_boxes_by_ego_dist.

These tests use pure-Python mocks (duck-typed boxes with .ego_dist and
.sample_token), so they run without nuscenes-devkit installed. The filter
implementation operates on duck-typed objects, matching how nuScenes'
EvalBoxes iterates its underlying list.
"""
from collections import defaultdict

import pytest

from tools.range_eval.filters import filter_eval_boxes_by_ego_dist


class _MockBox:
    """Duck-typed stand-in for nuscenes.eval.detection.data_classes.DetectionBox."""

    def __init__(self, sample_token, ego_dist, detection_name='car',
                 detection_score=0.5, attribute_name='vehicle.moving',
                 translation=(0.0, 0.0, 0.0)):
        self.sample_token = sample_token
        self.ego_dist = ego_dist
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name
        self.translation = translation


class _MockEvalBoxes:
    """Minimal EvalBoxes substitute with the methods filter_eval_boxes_by_ego_dist uses.

    Mirrors nuscenes.eval.common.data_classes.EvalBoxes:
      - .sample_tokens property
      - .__getitem__(token) -> List[box]
      - .add_boxes(token, boxes) builder method
      - .all -> flat list of boxes
    """

    def __init__(self):
        self.boxes = defaultdict(list)

    @property
    def sample_tokens(self):
        return list(self.boxes.keys())

    def __getitem__(self, token):
        return self.boxes[token]

    def add_boxes(self, token, boxes):
        self.boxes[token].extend(boxes)

    @property
    def all(self):
        return [b for token in self.boxes for b in self.boxes[token]]

    def __len__(self):
        return sum(len(v) for v in self.boxes.values())


def _build(*box_specs):
    """box_specs: tuples of (sample_token, ego_dist, **extra_kwargs)."""
    eb = _MockEvalBoxes()
    for spec in box_specs:
        token, dist = spec[0], spec[1]
        kwargs = spec[2] if len(spec) > 2 else {}
        eb.add_boxes(token, [_MockBox(token, dist, **kwargs)])
    return eb


def test_box_at_origin_in_first_bin():
    """Box with ego_dist=0 should be in [0, 30) but not [30, 50)."""
    eb = _build(('s1', 0.0))

    in_first = filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)
    assert len(in_first) == 1

    in_second = filter_eval_boxes_by_ego_dist(eb, 30.0, 50.0)
    assert len(in_second) == 0


def test_boundary_handling_left_inclusive():
    """ego_dist=30.0 → excluded from [0, 30), included in [30, 50)."""
    eb = _build(('s1', 30.0))

    in_first = filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)
    assert len(in_first) == 0

    in_second = filter_eval_boxes_by_ego_dist(eb, 30.0, 50.0)
    assert len(in_second) == 1


def test_empty_input():
    """Empty EvalBoxes → empty EvalBoxes, no exception."""
    eb = _MockEvalBoxes()
    out = filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)
    assert len(out) == 0
    assert out.sample_tokens == []


def test_filter_does_not_mutate_input():
    """Original EvalBoxes is unchanged after filter call."""
    eb = _build(('s1', 5.0), ('s1', 35.0), ('s2', 15.0))
    tokens_before = sorted(eb.sample_tokens)
    counts_before = {t: len(eb[t]) for t in tokens_before}

    _ = filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)

    tokens_after = sorted(eb.sample_tokens)
    counts_after = {t: len(eb[t]) for t in tokens_after}
    assert tokens_before == tokens_after
    assert counts_before == counts_after


def test_filter_preserves_box_metadata():
    """Sample tokens, attribute names, and other fields preserved on surviving boxes."""
    eb = _build(
        ('tok_A', 10.0, dict(detection_name='pedestrian',
                             detection_score=0.91,
                             attribute_name='pedestrian.standing',
                             translation=(1.0, 2.0, 3.0))),
        ('tok_B', 100.0, dict(detection_name='car',
                              detection_score=0.42,
                              attribute_name='vehicle.parked')),
    )

    out = filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)
    assert out.sample_tokens == ['tok_A']
    survivors = out['tok_A']
    assert len(survivors) == 1
    s = survivors[0]
    assert s.sample_token == 'tok_A'
    assert s.detection_name == 'pedestrian'
    assert s.detection_score == 0.91
    assert s.attribute_name == 'pedestrian.standing'
    assert s.translation == (1.0, 2.0, 3.0)


def test_panic_on_huge_ego_dist():
    """ego_dist > 1000m suggests global-frame translation was used; should raise."""
    eb = _build(('s1', 2000.0))
    with pytest.raises(ValueError, match='ego_dist'):
        filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)


def test_half_open_interval_at_max():
    """ego_dist=r_max is excluded (half-open)."""
    eb = _build(('s1', 49.999), ('s2', 50.0), ('s3', 50.0001))
    out = filter_eval_boxes_by_ego_dist(eb, 30.0, 50.0)
    surviving_tokens = out.sample_tokens
    assert 's1' in surviving_tokens
    assert 's2' not in surviving_tokens
    assert 's3' not in surviving_tokens


def test_multiple_boxes_per_sample():
    """Per-sample filtering keeps only the in-range boxes; sample-token entry preserved."""
    eb = _MockEvalBoxes()
    eb.add_boxes('s1', [
        _MockBox('s1', 5.0, detection_name='car'),
        _MockBox('s1', 25.0, detection_name='car'),
        _MockBox('s1', 45.0, detection_name='car'),
    ])

    out = filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)
    assert sorted(out.sample_tokens) == ['s1']
    assert len(out['s1']) == 2
    for b in out['s1']:
        assert 0.0 <= b.ego_dist < 30.0


def test_missing_ego_dist_attribute_raises():
    """Box without ego_dist attribute → AttributeError with helpful message."""
    eb = _MockEvalBoxes()
    bare = _MockBox('s1', 0.0)
    del bare.ego_dist
    eb.add_boxes('s1', [bare])

    with pytest.raises((AttributeError, ValueError)):
        filter_eval_boxes_by_ego_dist(eb, 0.0, 30.0)
