_base_ = ['isfusion_0075voxel.py']

# Hardware: 4xA30 (24GB each).
# Strategy: gradient accumulation to raise effective batch from 4 toward 16,
# paired with a conservative LR.
#
# First attempt with lr=1e-4 (paper's value) produced unstable training:
# loss_heatmap=1551 and grad_norm=12747 at iteration 50, well outside normal
# range. The paper's LR was designed for batch-size-16 training without
# gradient accumulation; with cumulative_iters=4, gradient norm dynamics
# differ and 1e-4 is too aggressive.
#
# Current LR (3.5e-5) is the geometric mean of the working A30 config's
# 1.25e-5 (which produced 67.4 mAP) and the paper's 1e-4. This trades off
# convergence speed against optimization stability.
#
# 4 GPUs * 1 sample_per_gpu * 4 accumulation steps = effective batch 16.

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

optimizer = dict(
    type='AdamW',
    lr=3.5e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
)

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=4,
    grad_clip=dict(max_norm=0.01, norm_type=2),
)
