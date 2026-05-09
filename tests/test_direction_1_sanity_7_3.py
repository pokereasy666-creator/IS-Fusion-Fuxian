"""Sanity test 7.3 (Direction 1, revision 2 spec).

Verify that when ``image_classifier`` is enabled, the BEV head's final-layer
classification supervision (``layer_-1_loss_cls``) is fed the **pre-fusion**
heatmap (``heatmap_pre_fusion``) and NOT the post-fusion ``heatmap``. This
guards against the spec C6 corruption: image gradients leaking back into BEV
training via the BEV cls supervision path.

Approach (per user clarification): rather than compare layer_-1_loss_cls
across two model variants (flaky because new modules' random init shifts the
RNG state), use a single model with image_classifier enabled, monkey-patch
``loss_cls`` to capture its first argument, run ``loss()``, and verify the
captured cls input matches the reshaped ``heatmap_pre_fusion`` rather than
the reshaped post-fusion ``heatmap``.

Run with: pytest tests/test_direction_1_sanity_7_3.py -v
"""
import torch

from mmdet3d.models.dense_heads.image_classifier_head import (
    fuse_scores_additive,
)


def _build_loss_inputs(B=2, num_classes=3, num_layers=1, K=4):
    """Construct synthetic preds_dict, labels, label_weights, etc.

    Mimics the shapes that ``TransFusionHeadV2.loss`` consumes:
    ``preds_dict['heatmap']``: [B, num_classes, num_layers * K]
    ``preds_dict['heatmap_pre_fusion']``: [B, num_classes, K] (final-layer only)
    ``labels`` / ``label_weights``: [B, num_layers * K]
    """
    torch.manual_seed(0)
    s_bev_logits_final = torch.randn(B, num_classes, K)
    s_img_logits = torch.randn(B, K, num_classes)
    in_any_view = torch.ones(B, K, dtype=torch.bool)

    fused_final = fuse_scores_additive(
        s_bev_logits=s_bev_logits_final,
        s_img_logits=s_img_logits,
        alpha_pre_softplus=torch.zeros(num_classes),
        in_any_view=in_any_view,
    )

    if num_layers > 1:
        s_bev_other_layers = torch.randn(B, num_classes, K * (num_layers - 1))
        full_heatmap = torch.cat([s_bev_other_layers, fused_final], dim=-1)
    else:
        full_heatmap = fused_final

    preds_dict = dict(
        heatmap=full_heatmap,
        heatmap_pre_fusion=s_bev_logits_final,
        s_img_logits=s_img_logits,
        in_any_view=in_any_view,
    )
    return preds_dict, fused_final, s_bev_logits_final


def test_layer_minus1_loss_cls_uses_pre_fusion():
    """Captured cls input == reshape(heatmap_pre_fusion), NOT reshape(fused)."""
    B, num_classes, num_layers, K = 2, 3, 1, 4
    preds_dict, fused_final, pre_fusion_final = _build_loss_inputs(
        B=B, num_classes=num_classes, num_layers=num_layers, K=K
    )

    # Reproduce TransFusionHeadV2.loss's slicing for the final layer.
    idx_layer = num_layers - 1
    is_final_layer = (idx_layer == num_layers - 1)
    if is_final_layer and 'heatmap_pre_fusion' in preds_dict:
        layer_score = preds_dict['heatmap_pre_fusion']
    else:
        layer_score = preds_dict['heatmap'][
            ..., idx_layer * K : (idx_layer + 1) * K
        ]
    captured = layer_score.permute(0, 2, 1).reshape(-1, num_classes)

    expected_pre = pre_fusion_final.permute(0, 2, 1).reshape(-1, num_classes)
    expected_post = fused_final.permute(0, 2, 1).reshape(-1, num_classes)

    assert torch.allclose(captured, expected_pre), (
        'Final-layer cls supervision should be fed the PRE-fusion heatmap.'
    )
    assert not torch.allclose(captured, expected_post), (
        'Final-layer cls supervision must NOT be fed the post-fusion heatmap; '
        'this would let image gradients corrupt BEV training (spec C6).'
    )


def test_routing_falls_back_to_heatmap_when_pre_fusion_missing():
    """If image_classifier disabled (no heatmap_pre_fusion key), use 'heatmap'."""
    B, num_classes, num_layers, K = 2, 3, 1, 4
    torch.manual_seed(1)
    bev_only = torch.randn(B, num_classes, num_layers * K)
    preds_dict = dict(heatmap=bev_only)

    idx_layer = num_layers - 1
    is_final_layer = (idx_layer == num_layers - 1)
    if is_final_layer and 'heatmap_pre_fusion' in preds_dict:
        layer_score = preds_dict['heatmap_pre_fusion']
    else:
        layer_score = preds_dict['heatmap'][
            ..., idx_layer * K : (idx_layer + 1) * K
        ]
    captured = layer_score.permute(0, 2, 1).reshape(-1, num_classes)
    expected = bev_only[..., -K:].permute(0, 2, 1).reshape(-1, num_classes)
    assert torch.allclose(captured, expected)


def test_intermediate_layers_still_use_heatmap_when_pre_fusion_present():
    """For non-final layers, use 'heatmap' even if heatmap_pre_fusion exists."""
    B, num_classes, num_layers, K = 2, 3, 3, 4
    preds_dict, _, pre_fusion_final = _build_loss_inputs(
        B=B, num_classes=num_classes, num_layers=num_layers, K=K
    )

    # Non-final layer (e.g., idx_layer = 0).
    idx_layer = 0
    is_final_layer = (idx_layer == num_layers - 1)
    if is_final_layer and 'heatmap_pre_fusion' in preds_dict:
        layer_score = preds_dict['heatmap_pre_fusion']
    else:
        layer_score = preds_dict['heatmap'][
            ..., idx_layer * K : (idx_layer + 1) * K
        ]
    expected = preds_dict['heatmap'][..., 0:K]
    assert torch.equal(layer_score, expected)


def test_end_to_end_loss_cls_capture_with_real_head():
    """End-to-end: monkey-patch loss_cls on a real TransFusionHeadV2 instance.

    Builds a minimal TransFusionHeadV2, hand-constructs a preds_dict that
    contains both ``heatmap`` (post-fusion) and ``heatmap_pre_fusion``
    (pre-fusion), monkey-patches ``loss_cls`` to capture its first arg,
    and asserts that the captured cls input matches the reshaped
    pre-fusion heatmap rather than the reshaped post-fusion heatmap.

    To keep the test focused on routing (not on the image-classification or
    consistency loss machinery), we leave ``head.image_classifier=None``.
    The routing check inside ``loss`` is gated on
    ``'heatmap_pre_fusion' in preds_dict``, not on ``image_classifier``,
    so this is a faithful structural test.
    """
    from unittest import mock

    from mmdet3d.models.dense_heads.transfusion_head_v2 import TransFusionHeadV2

    train_cfg = dict(
        dataset='nuScenes',
        assigner=dict(
            type='HungarianAssigner3D',
            iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
            cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
            reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=0.25),
        ),
        pos_weight=-1,
        gaussian_overlap=0.1,
        min_radius=2,
        grid_size=[1440, 1440, 40],
        voxel_size=[0.075, 0.075, 0.2],
        out_size_factor=8,
        code_weights=[1.0] * 8 + [0.2, 0.2],
        point_cloud_range=[-54, -54, -5, 54, 54, 3],
    )
    test_cfg = dict(
        dataset='nuScenes',
        grid_size=[1440, 1440, 40],
        out_size_factor=8,
        pc_range=[-54, -54],
        voxel_size=[0.075, 0.075],
        nms_type=None,
        use_rotate_nms=True,
        nms_thr=0.2,
        max_num=200,
    )
    head = TransFusionHeadV2(
        num_proposals=4,
        auxiliary=True,
        in_channels=8,
        hidden_channel=8,
        num_classes=3,
        num_decoder_layers=1,
        num_heads=2,
        nms_kernel_size=1,
        ffn_channel=8,
        dropout=0.0,
        common_heads=dict(
            center=(2, 1), height=(1, 1), dim=(3, 1), rot=(2, 1), vel=(2, 1)
        ),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25,
            reduction='mean', loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0
        ),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54, -54],
            voxel_size=[0.075, 0.075],
            out_size_factor=8,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        image_classifier=None,
    )
    # Leave head.image_classifier = None so loss_img_cls / loss_consistency
    # don't run; only the BEV final-layer cls call goes through loss_cls.
    assert head.image_classifier is None

    # Build a synthetic preds_dict matching what forward_single produces.
    B, num_classes, K = 2, 3, head.num_proposals
    torch.manual_seed(42)
    s_bev_pre_final = torch.randn(B, num_classes, K)
    s_img_logits = torch.full((B, K, num_classes), 0.5)
    in_any_view = torch.ones(B, K, dtype=torch.bool)
    fused_final = fuse_scores_additive(
        s_bev_logits=s_bev_pre_final,
        s_img_logits=s_img_logits,
        alpha_pre_softplus=torch.zeros(num_classes),
        in_any_view=in_any_view,
    )
    preds_dict = dict(
        heatmap=fused_final,
        heatmap_pre_fusion=s_bev_pre_final,
        center=torch.randn(B, 2, K),
        height=torch.randn(B, 1, K),
        dim=torch.randn(B, 3, K),
        rot=torch.randn(B, 2, K),
        vel=torch.randn(B, 2, K),
        dense_heatmap=torch.randn(B, num_classes, 8, 8),
        query_heatmap_score=torch.randn(B, num_classes, K),
    )

    fake_labels = torch.randint(0, num_classes + 1, (B, K), dtype=torch.long)
    fake_label_weights = torch.ones(B, K, dtype=torch.float32)
    fake_bbox_targets = torch.zeros(B, K, 10)
    fake_bbox_weights = torch.zeros(B, K, 10)
    fake_ious = torch.zeros(B, K)
    fake_heatmap_target = torch.zeros(B, num_classes, 8, 8)

    def fake_get_targets(_gt_bboxes, _gt_labels, _preds):
        return (
            fake_labels,
            fake_label_weights,
            fake_bbox_targets,
            fake_bbox_weights,
            fake_ious,
            int(B * K // 2),
            0.0,
            fake_heatmap_target,
        )

    captured = []

    def capturing_loss_cls(cls_score, *args, **kwargs):
        captured.append(cls_score.detach().clone())
        return cls_score.sum() * 0.0

    with mock.patch.object(head, 'get_targets', side_effect=fake_get_targets), \
         mock.patch.object(head, 'loss_cls', side_effect=capturing_loss_cls), \
         mock.patch.object(
             head, 'loss_bbox',
             side_effect=lambda *a, **k: torch.tensor(0.0)), \
         mock.patch.object(
             head, 'loss_heatmap',
             side_effect=lambda *a, **k: torch.tensor(0.0)):
        head.loss([], [], [[preds_dict]])

    assert len(captured) == 1, (
        f'expected exactly 1 loss_cls call (BEV final layer), got {len(captured)}'
    )
    last_cls_input = captured[-1]
    expected_pre = s_bev_pre_final.permute(0, 2, 1).reshape(-1, num_classes)
    expected_post = fused_final.permute(0, 2, 1).reshape(-1, num_classes)

    assert last_cls_input.shape == expected_pre.shape, (
        f'shape mismatch: {last_cls_input.shape} vs {expected_pre.shape}'
    )
    assert torch.allclose(last_cls_input, expected_pre, atol=1e-6), (
        'Final-layer loss_cls did NOT receive heatmap_pre_fusion.'
    )
    assert not torch.allclose(last_cls_input, expected_post, atol=1e-6), (
        'Final-layer loss_cls received the POST-fusion heatmap; '
        'BEV training will be corrupted by image gradients (spec C6 violation).'
    )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
