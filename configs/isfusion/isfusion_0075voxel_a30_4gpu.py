_base_ = ['isfusion_0075voxel.py']

# Hardware: 4xA30 (24GB each).
# Strategy: 4 GPUs * 1 sample_per_gpu = effective batch 4 (vs paper's 16).
# No gradient accumulation. LR slightly elevated from the working 3-GPU
# config (1.25e-5) to 2e-5 to account for the 33% larger effective batch.
#
# This is a fallback config after two grad accumulation attempts produced
# broken optimization (loss ~2000, grad_norm ~14000). Goal: confirm 4-GPU
# training works at all in this codebase, then assess whether 67.4 -> 68+
# mAP gain is achievable simply through 33% more data per step.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

optimizer = dict(
    type='AdamW',
    lr=2e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
)

# NO gradient accumulation; let optimizer_config inherit from base
# (which has just grad_clip).
