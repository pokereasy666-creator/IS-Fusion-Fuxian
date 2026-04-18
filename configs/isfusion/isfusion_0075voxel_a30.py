_base_ = ['isfusion_0075voxel.py']

# Scaled for 2 GPUs x 1 sample_per_gpu = 2 total batch size.
# Original config targets 8 GPUs x 2 samples = 16 total batch size.
# LR scaled linearly: 0.0001 * (2/16) = 0.0000125

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

optimizer = dict(
    type='AdamW',
    lr=0.0000125,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
)
