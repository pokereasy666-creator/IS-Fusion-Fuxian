"""Unit tests for ISFusionDetector helpers.

Covers the augmentation/calibration matrix normalization at the detector
boundary, introduced to bridge the training-mode (stacked tensor) and
test-mode (list of per-sample tensors) representations from MMDetection3D's
data pipeline.

Run with: pytest tests/test_isfusion_detector.py -v
"""
import pytest
import torch

from mmdet3d.models.detectors.isfusion import ISFusionDetector


def test_normalize_aug_matrix_none():
    assert ISFusionDetector._normalize_aug_matrix(None) is None


def test_normalize_aug_matrix_tensor_passthrough():
    t = torch.eye(4).expand(2, 4, 4).clone()
    out = ISFusionDetector._normalize_aug_matrix(t)
    assert out is t


def test_normalize_aug_matrix_list_stacks():
    lst = [torch.eye(4), torch.eye(4) * 2]
    out = ISFusionDetector._normalize_aug_matrix(lst)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 4)
    assert torch.allclose(out[0], torch.eye(4))
    assert torch.allclose(out[1], torch.eye(4) * 2)


def test_normalize_aug_matrix_empty_list():
    assert ISFusionDetector._normalize_aug_matrix([]) is None


def test_normalize_aug_matrix_rejects_bad_type():
    with pytest.raises(TypeError):
        ISFusionDetector._normalize_aug_matrix(42)
    with pytest.raises(TypeError):
        ISFusionDetector._normalize_aug_matrix([torch.eye(4), 'not a tensor'])


def test_normalize_aug_matrix_unwraps_singleton_batched():
    """Singleton list of an already-batched tensor should unwrap, not stack.

    Verifies the IS-Fusion test pipeline pattern: dataloader delivers
    [tensor[B, 4, 4]] or [tensor[B, V, 4, 4]] which should unwrap to the inner
    tensor, not become [1, B, 4, 4].
    """
    # Case 1: singleton list of [B, 4, 4] (lidar_aug_matrix shape)
    batched_3d = torch.eye(4).expand(2, 4, 4).clone()
    out = ISFusionDetector._normalize_aug_matrix([batched_3d])
    assert out.shape == (2, 4, 4), \
        f'Expected [2, 4, 4], got {tuple(out.shape)}'
    assert torch.equal(out, batched_3d)

    # Case 2: singleton list of [B, V, 4, 4] (img_aug_matrix / lidar2img shape)
    batched_4d = torch.eye(4).expand(2, 6, 4, 4).clone()
    out = ISFusionDetector._normalize_aug_matrix([batched_4d])
    assert out.shape == (2, 6, 4, 4), \
        f'Expected [2, 6, 4, 4], got {tuple(out.shape)}'
    assert torch.equal(out, batched_4d)

    # Sanity: multi-element list of [4, 4] tensors should still stack normally
    per_sample_list = [torch.eye(4), torch.eye(4) * 2]
    out = ISFusionDetector._normalize_aug_matrix(per_sample_list)
    assert out.shape == (2, 4, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
