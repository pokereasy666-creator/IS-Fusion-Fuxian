"""Tests for the inference-time fusion_weight_override on ImageClassifierHead.

Run with: pytest tests/test_fixed_alpha_override.py
"""
import ast

import pytest
import torch

from mmdet3d.models.dense_heads import image_classifier_head as ich_module
from mmdet3d.models.dense_heads.image_classifier_head import (
    ImageClassifierHead,
    fuse_scores_additive,
)


def _make_head(**overrides):
    cfg = dict(
        in_channels=8,
        num_classes=3,
        hidden_channels=8,
        num_views=2,
        feature_stride=8,
        img_shape=(64, 80),
        out_size_factor=8,
        voxel_size=(0.075, 0.075),
        pc_range=(-54.0, -54.0),
        alpha_init=0.0,
        depth_min=0.5,
    )
    cfg.update(overrides)
    return ImageClassifierHead(**cfg)


def test_module_file_parses_cleanly():
    """The modified head module must be syntactically valid Python."""
    with open(ich_module.__file__, 'r') as f:
        source = f.read()
    ast.parse(source)


def test_default_fusion_weight_override_is_none():
    head = _make_head()
    assert head.fusion_weight_override is None


def test_fusion_weight_override_stored():
    override = [0.5, 1.0, 2.0]
    head = _make_head(fusion_weight_override=override)
    assert head.fusion_weight_override == override


def test_fusion_weight_override_length_mismatch_raises():
    with pytest.raises(AssertionError):
        _make_head(fusion_weight_override=[0.5, 1.0])  # num_classes=3


def test_override_zero_yields_s_bev():
    """override=[0.0]*C -> fused == s_bev everywhere (in-view & out-of-view).

    With alpha_pos==0 the additive term vanishes, and out-of-view positions
    pass through s_bev unchanged, so the entire output equals s_bev.
    """
    B, K, C = 2, 4, 3
    s_bev = torch.randn(B, C, K)
    s_img = torch.randn(B, K, C)
    # alpha_pre would give softplus(0) ~= 0.693 if not overridden.
    alpha_pre = torch.zeros(C)
    in_any_view = torch.tensor(
        [[True, False, True, False], [False, True, False, True]]
    )
    fused = fuse_scores_additive(
        s_bev, s_img, alpha_pre, in_any_view,
        fusion_weight_override=[0.0] * C,
    )
    assert torch.equal(fused, s_bev)


def test_override_one_yields_s_bev_plus_s_img():
    """override=[1.0]*C with all in-view -> fused == s_bev + s_img.permute(0,2,1)."""
    B, K, C = 2, 4, 3
    s_bev = torch.randn(B, C, K)
    s_img = torch.randn(B, K, C)
    # alpha_pre would give ~0 if not overridden; override forces 1.0.
    alpha_pre = torch.full((C,), -1e6)
    in_any_view = torch.ones(B, K, dtype=torch.bool)
    fused = fuse_scores_additive(
        s_bev, s_img, alpha_pre, in_any_view,
        fusion_weight_override=[1.0] * C,
    )
    expected = s_bev + s_img.permute(0, 2, 1)
    assert torch.allclose(fused, expected, atol=0.0)


def test_override_device_dtype_match():
    """Override tensor lands on s_bev_logits' device/dtype."""
    B, K, C = 1, 2, 3
    s_bev = torch.randn(B, C, K, dtype=torch.float64)
    s_img = torch.randn(B, K, C, dtype=torch.float64)
    alpha_pre = torch.zeros(C)
    in_any_view = torch.ones(B, K, dtype=torch.bool)
    fused = fuse_scores_additive(
        s_bev, s_img, alpha_pre, in_any_view,
        fusion_weight_override=[1.0, 1.0, 1.0],
    )
    assert fused.dtype == torch.float64
    assert fused.device == s_bev.device


def test_default_path_unchanged():
    """Without override, behavior matches the original alpha-softplus path."""
    B, K, C = 1, 3, 3
    s_bev = torch.randn(B, C, K)
    s_img = torch.randn(B, K, C)
    alpha_pre = torch.zeros(C)
    in_any_view = torch.ones(B, K, dtype=torch.bool)
    fused = fuse_scores_additive(s_bev, s_img, alpha_pre, in_any_view)
    alpha_pos = torch.nn.functional.softplus(alpha_pre)
    expected = s_bev + alpha_pos[None, :, None] * s_img.permute(0, 2, 1)
    assert torch.allclose(fused, expected, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
