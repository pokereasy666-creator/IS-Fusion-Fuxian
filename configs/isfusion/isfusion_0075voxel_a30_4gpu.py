_base_ = ['isfusion_0075voxel.py']
# Hardware: 4xA30 (24GB each).
# Strategy: match paper's effective batch size 16 via gradient accumulation.
# 4 GPUs * 1 sample_per_gpu * 4 accumulation steps = effective batch 16.
# This matches the paper's 8 GPUs * 2 sample_per_gpu = effective batch 16.
# Therefore use the paper's LR directly (1e-4), no scaling.
#
# Reference: this config supersedes isfusion_0075voxel_a30.py, which used
# 3 GPUs with linearly-scaled LR and reached only 67.4 mAP. The LR scaling
# was diagnosed as the cause of underperformance; gradient accumulation
# restores the paper's effective batch size while staying within A30 memory.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
)

# Gradient accumulation: accumulate gradients across 4 forward passes
# before stepping the optimizer. This raises effective batch from 4 to 16
# without increasing per-step memory.
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=4,
    grad_clip=dict(max_norm=0.01, norm_type=2),
)
