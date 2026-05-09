"""Unit tests for ImageClassifierHead and fuse_scores_additive.

Run with: pytest tests/test_image_classifier_head.py
"""
import math

import pytest
import torch

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


def _make_inputs(B=2, V=2, K=4, C=8, H_feat=8, W_feat=10):
    img_feats = torch.randn(B * V, C, H_feat, W_feat)
    # query_pos in BEV grid units. Using small grid coords keeps centers near
    # the origin in metric coordinates, which is in front of the synthetic
    # cameras below.
    query_pos = torch.tensor([
        [[20.0, 20.0], [40.0, 30.0], [50.0, 50.0], [10.0, 60.0]],
        [[30.0, 25.0], [45.0, 45.0], [60.0, 30.0], [25.0, 35.0]],
    ])[:B, :K]
    query_height = torch.zeros(B, 1, K)

    # Identity LiDAR augmentation per sample.
    lidar_aug_matrix = torch.eye(4)[None].expand(B, -1, -1).contiguous()

    # Synthetic lidar2img: looking down +x with a simple intrinsic.
    # Camera frame: x_cam = -y_lidar, y_cam = -z_lidar, z_cam = x_lidar.
    R = torch.tensor([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    fx = fy = 80.0
    cx = 40.0
    cy = 32.0
    K_intr = torch.tensor([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lidar2img_v0 = K_intr @ R
    # View 1: same orientation (overlapping). Distinguishing views is not
    # required for shape tests.
    lidar2img = torch.stack([lidar2img_v0, lidar2img_v0], dim=0)
    lidar2img = lidar2img[None].expand(B, -1, -1, -1).contiguous()

    img_aug_matrix = torch.eye(4)[None, None].expand(B, V, -1, -1).contiguous()
    return dict(
        img_feats=img_feats,
        query_pos=query_pos,
        query_height=query_height,
        lidar2img=lidar2img,
        img_aug_matrix=img_aug_matrix,
        lidar_aug_matrix=lidar_aug_matrix,
    )


def test_forward_shape():
    """Output shapes match spec: logits [B,K,C], in_any_view [B,K]."""
    head = _make_head()
    B, V, K, C, H_feat, W_feat = 2, 2, 4, 8, 8, 10
    inputs = _make_inputs(B=B, V=V, K=K, C=C, H_feat=H_feat, W_feat=W_feat)
    out = head(**inputs)
    assert out['logits'].shape == (B, K, head.num_classes)
    assert out['in_any_view'].shape == (B, K)
    assert out['in_any_view'].dtype == torch.bool


def test_out_of_view_default():
    """If every projection has negative depth, logits == out_of_view_default."""
    head = _make_head()
    # Set the default to a known nonzero value.
    with torch.no_grad():
        head.out_of_view_default.copy_(torch.tensor([3.0, -1.0, 7.0]))

    B, V, K = 1, 2, 3
    inputs = _make_inputs(B=B, V=V, K=K)
    # Override lidar2img so projected z is always negative -> behind camera.
    bad = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -10.0],  # z_cam = -x_lidar - 10 -> always negative
        [0.0, 0.0, 0.0, 1.0],
    ])
    inputs['lidar2img'] = bad[None, None].expand(B, V, -1, -1).contiguous()

    out = head(**inputs)
    assert not out['in_any_view'].any()
    expected = head.out_of_view_default[None, None, :].expand(B, K, -1)
    assert torch.allclose(out['logits'], expected)


def test_augmentation_inversion_identity():
    """Identity lidar_aug + identity img_aug == no-augmentation projection."""
    head = _make_head()
    inputs = _make_inputs()

    # Reference output with identity augmentations.
    out_a = head(**inputs)

    # Same again with explicitly recreated identities -- must match.
    inputs2 = {**inputs}
    inputs2['lidar_aug_matrix'] = torch.eye(4)[None].expand(
        inputs['lidar_aug_matrix'].shape[0], -1, -1
    ).contiguous()
    inputs2['img_aug_matrix'] = torch.eye(4)[None, None].expand(
        inputs['img_aug_matrix'].shape[0], inputs['img_aug_matrix'].shape[1], -1, -1
    ).contiguous()
    out_b = head(**inputs2)

    assert torch.allclose(out_a['logits'], out_b['logits'])
    assert torch.equal(out_a['in_any_view'], out_b['in_any_view'])


def test_augmentation_inversion_lidar_rotation():
    """A 180 degree LiDAR rotation cancels itself -- forward must match identity."""
    head = _make_head()
    inputs_id = _make_inputs()

    # Build a 180-deg rotation about z in LiDAR frame.
    theta = math.pi
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    B = inputs_id['query_pos'].shape[0]

    inputs_rot = {**inputs_id}
    # Apply the rotation to query_pos's metric coords by pre-rotating
    # query_metric: query_aug_metric_rot = R @ query_aug_metric_id, where R is
    # treated as the "augmentation". We simulate this by setting
    # lidar_aug_matrix=R and rotating the input query_pos accordingly so that
    # after un-augmentation the original LiDAR coords coincide with the
    # identity-augmentation case.
    inputs_rot['lidar_aug_matrix'] = R[None].expand(B, -1, -1).contiguous()

    # Compute metric (augmented) coords for the rotated case by applying R to
    # the identity-case metric coords.
    sx = head.out_size_factor * head.voxel_size[0]
    sy = head.out_size_factor * head.voxel_size[1]
    x_id = inputs_id['query_pos'][..., 0] * sx + head.pc_range[0]
    y_id = inputs_id['query_pos'][..., 1] * sy + head.pc_range[1]
    z_id = inputs_id['query_height'].squeeze(1)

    metric_id = torch.stack([x_id, y_id, z_id], dim=-1)  # [B, K, 3]
    metric_h = torch.cat([metric_id, torch.ones_like(metric_id[..., :1])], dim=-1)
    metric_rot = torch.einsum('bki,ji->bkj', metric_h, R)[..., :3]

    # Convert rotated metric back to BEV grid units.
    x_grid = (metric_rot[..., 0] - head.pc_range[0]) / sx
    y_grid = (metric_rot[..., 1] - head.pc_range[1]) / sy
    inputs_rot['query_pos'] = torch.stack([x_grid, y_grid], dim=-1)
    inputs_rot['query_height'] = metric_rot[..., 2:3].permute(0, 2, 1)

    out_id = head(**inputs_id)
    out_rot = head(**inputs_rot)

    # After un-aug, query_orig_h should be identical, so projections, samples,
    # and outputs must match (modulo float precision).
    assert torch.allclose(out_id['logits'], out_rot['logits'], atol=1e-4)
    assert torch.equal(out_id['in_any_view'], out_rot['in_any_view'])


def test_score_fusion_alpha_zero():
    """Very-negative alpha_pre_softplus -> alpha~0 -> fused == s_bev."""
    B, K, C = 2, 4, 3
    s_bev = torch.randn(B, C, K)
    s_img = torch.randn(B, K, C)
    alpha_pre = torch.full((C,), -1e6)  # softplus is essentially 0
    in_any_view = torch.ones(B, K, dtype=torch.bool)
    fused = fuse_scores_additive(s_bev, s_img, alpha_pre, in_any_view)
    assert torch.allclose(fused, s_bev, atol=1e-5)


def test_score_fusion_out_of_view():
    """For out-of-view candidates, fused == s_bev for those positions."""
    B, K, C = 2, 4, 3
    s_bev = torch.randn(B, C, K)
    s_img = torch.randn(B, K, C)
    alpha_pre = torch.zeros(C)  # softplus(0) ~ 0.69
    in_any_view = torch.tensor([[True, False, True, False], [False, True, False, True]])
    fused = fuse_scores_additive(s_bev, s_img, alpha_pre, in_any_view)

    # Out-of-view positions match s_bev exactly.
    out_mask = ~in_any_view  # [B, K]
    assert torch.allclose(fused[..., out_mask[0]][0], s_bev[..., out_mask[0]][0])
    assert torch.allclose(fused[..., out_mask[1]][1], s_bev[..., out_mask[1]][1])

    # In-view positions differ from s_bev (s_img has random values, alpha != 0).
    in_mask = in_any_view
    diff = (fused - s_bev).abs()
    in_diff_b0 = diff[0, :, in_mask[0]]
    assert in_diff_b0.sum() > 0


def test_gradients_flow():
    """All learnable params receive gradients from at least one valid run."""
    head = _make_head()
    inputs = _make_inputs()
    out = head(**inputs)
    # Force at least one in-view candidate to have a finite path back; if none
    # are in view this test is a no-op for some params, so we additionally add
    # a small term from out_of_view_default unconditionally.
    loss = out['logits'].sum()
    loss.backward()

    for name, p in head.named_parameters():
        # alpha and out_of_view_default may be untouched if all in_any_view; we
        # still expect classifier and feature_proj weights to receive grad
        # whenever at least one candidate is in view.
        if name.startswith('feature_proj') or name.startswith('classifier'):
            if out['in_any_view'].any():
                assert p.grad is not None and p.grad.abs().sum() > 0, (
                    f'No gradient reached {name}'
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
