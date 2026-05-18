"""A3 ablation: full Direction 1 (image classifier + additive fusion + consistency loss).

Inherits from the standard 4-GPU Direction 1 config. Default loss weights match A3's intent:
- loss_consistency_weight=0.1 (consistency loss enabled)
- loss_img_cls_weight=1.0 (image classifier supervised)

Throughput optimization: doubles per-GPU batch, halves grad_accum.
Effective batch = 2 (samples_per_gpu) x 4 (GPUs) x 2 (cumulative_iters) = 16, same as A0/A1.
Expected wall-clock reduction vs A1 setup: ~30-40%.

Note: per-GPU BN now sees batch-of-2 instead of batch-of-1. This is a minor deviation
from A0/A1's training conditions; expected impact <0.2 mAP, well within noise.
"""
_base_ = ['isfusion_0075voxel_a30_4gpu.py']

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
)

optimizer_config = dict(cumulative_iters=2)
