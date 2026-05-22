"""Tests for D1 F3: zero-sum alpha reparam (fix #1) + confidence-masked
alpha-loss (fix #2).

Standalone -- only ``torch`` is required at the module level. The tests that
need ``ImageClassifierHead`` import it lazily inside the function, mirroring
the pattern in ``tests/test_f2_alpha_loss.py``; tests #2a/#2b replicate the
masking tensor ops inline so they don't drag in mmdet3d/CUDA.
"""
import pytest
import torch
import torch.nn.functional as F


# ---------- FIX #1 (zero-sum reparam) ----------

def test_alpha_log_weight_init():
    """At init (alpha=zeros, alpha_beta=0): softplus(alpha_log_weight)=0.6931."""
    from mmdet3d.models.dense_heads.image_classifier_head import (
        ImageClassifierHead,
    )

    num_classes = 10
    head = ImageClassifierHead(
        in_channels=8,
        num_classes=num_classes,
        hidden_channels=8,
        num_views=2,
        feature_stride=8,
        img_shape=(64, 80),
        out_size_factor=8,
        voxel_size=(0.075, 0.075),
        pc_range=(-54.0, -54.0),
        alpha_init=0.0,
        beta_init=0.0,
        learn_beta=False,
        depth_min=0.5,
    )

    expected = torch.full((num_classes,), 0.6931)
    actual = F.softplus(head.alpha_log_weight())
    assert torch.allclose(actual, expected, atol=1e-3), (
        f'softplus(alpha_log_weight()) = {actual.tolist()}, '
        f'expected ~{expected.tolist()}'
    )


def test_alpha_grad_zero_mean_and_beta_frozen():
    """alpha.grad must be zero-mean; alpha_beta.grad must be None (frozen).

    At init alpha_log_weight() is identically zero, so any scalar from it
    has zero gradient — that would make the zero-mean assertion pass
    trivially. To genuinely exercise the mean-centering projection we
    perturb alpha off zero AND use an asymmetric scalar so the raw
    gradient on alpha_log_weight is non-uniform.
    """
    from mmdet3d.models.dense_heads.image_classifier_head import (
        ImageClassifierHead,
    )

    torch.manual_seed(0)
    num_classes = 10
    head = ImageClassifierHead(
        in_channels=8,
        num_classes=num_classes,
        hidden_channels=8,
        num_views=2,
        feature_stride=8,
        img_shape=(64, 80),
        out_size_factor=8,
        voxel_size=(0.075, 0.075),
        pc_range=(-54.0, -54.0),
        alpha_init=0.0,
        beta_init=0.0,
        learn_beta=False,
        depth_min=0.5,
    )

    head.alpha.data = torch.randn(num_classes)
    head.alpha.grad = None

    weights = torch.linspace(-2.0, 2.0, num_classes)
    scalar = (head.alpha_log_weight() * weights).sum()
    scalar.backward()

    assert head.alpha.grad is not None, 'alpha.grad is None after backward'
    grad_mean = head.alpha.grad.mean().abs().item()
    assert grad_mean < 1e-6, (
        f'alpha.grad must be zero-mean (mean-centering projects out the '
        f'1-vector direction); got |mean|={grad_mean:.3e}'
    )
    assert head.alpha_beta.grad is None, (
        'alpha_beta.grad must be None (learn_beta=False sets '
        'requires_grad=False, so the global scale stays frozen)'
    )


# ---------- FIX #2 (confidence-masked alpha-loss) ----------
# Inline replication of the masking lines from transfusion_head_v2.py so the
# tests don't import the full head (which pulls in mmdet3d/CUDA). These four
# lines mirror the production code under test.

def _conf_mask_lines(s_bev_pre, final_label_weights, alpha_conf_threshold):
    """Replicate the four masking lines exactly."""
    bev_conf = s_bev_pre.sigmoid().max(dim=1).values            # [B, K]
    conf_mask = (bev_conf >= alpha_conf_threshold).reshape(-1)  # [B*K]
    alpha_label_weights = final_label_weights * conf_mask.to(final_label_weights.dtype)
    alpha_avg = max(int((alpha_label_weights > 0).sum().item()), 1)
    return alpha_label_weights, alpha_avg


def test_conf_mask_keeps_confident_drops_blind_recall():
    """Confident candidates keep weight 1; BEV-blind matched-positives go to 0.

    Synthetic [B=1, C=3, K=2]:
    - candidate k=0: one logit large enough that sigmoid > 0.3 (confident).
    - candidate k=1: all logits very negative (sigmoid ~ 0; BEV-blind) but
      assigned a positive label (recall case). The fix #2 masking must
      drop it from the alpha-loss.
    """
    B, C, K = 1, 3, 2
    s_bev_pre = torch.full((B, C, K), -10.0)
    # k=0 confident: sigmoid(2.0) ~ 0.88 > 0.3
    s_bev_pre[0, 0, 0] = 2.0
    # k=1 stays at -10 across all classes (sigmoid ~ 4.5e-5)

    # Both candidates are matched positives in the assigner sense (weight 1).
    final_label_weights = torch.ones(B * K)

    alpha_label_weights, alpha_avg = _conf_mask_lines(
        s_bev_pre, final_label_weights, alpha_conf_threshold=0.3
    )

    assert alpha_label_weights[0].item() == pytest.approx(1.0), (
        f'confident k=0 should keep weight 1, got {alpha_label_weights[0].item()}'
    )
    assert alpha_label_weights[1].item() == pytest.approx(0.0), (
        f'BEV-blind matched-positive k=1 should be masked to 0, got '
        f'{alpha_label_weights[1].item()}'
    )
    assert alpha_avg == 1, f'alpha_avg should be 1 (one confident), got {alpha_avg}'


def test_alpha_avg_no_div_by_zero():
    """With no confident candidate, alpha_avg must be >= 1."""
    B, C, K = 1, 3, 4
    s_bev_pre = torch.full((B, C, K), -10.0)
    final_label_weights = torch.ones(B * K)

    alpha_label_weights, alpha_avg = _conf_mask_lines(
        s_bev_pre, final_label_weights, alpha_conf_threshold=0.3
    )

    assert (alpha_label_weights > 0).sum().item() == 0, (
        'sanity: all candidates should be masked out in the degenerate case'
    )
    assert alpha_avg >= 1, (
        f'alpha_avg must be clamped to >=1 to avoid div-by-zero, got {alpha_avg}'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
