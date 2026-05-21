"""F2: learned per-class alpha via detached-fusion detection loss.

- alpha gradient comes from loss_alpha (focal on detached-fusion scores),
  NOT from consistency loss (which is disabled here).
- consistency loss disabled (weight 0).
- alpha frozen for epoch 1, unfrozen epochs 2-10 (via AlphaFreezeHook
  which gradient-zeroes alpha during the freeze period; alpha stays
  requires_grad=True from construction so its 0.1x LR param group is
  built by the optimizer constructor).
- alpha learns at 0.1x base LR via paramwise_cfg custom_keys.
- Throughput: samples_per_gpu=2, cumulative_iters=2 (effective batch 16).

Background: A1 saw alpha frozen (no gradient). A3 saw alpha collapse globally
because its only gradient source was the consistency loss, which optimizes
modality agreement rather than detection benefit. F2 fixes the gradient
source: alpha now learns from a per-class focal loss on detached BEV+image
fusion, so the only parameter that loss can move is alpha, and it is
optimized for the right question ("does fusion improve detection of matched
candidates").
"""
_base_ = ['isfusion_0075voxel_a30_4gpu.py']

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
)

optimizer_config = dict(cumulative_iters=2)

model = dict(
    pts_bbox_head=dict(
        loss_consistency_weight=0.0,
        loss_img_cls_weight=1.0,
        loss_alpha_weight=0.5,
    ),
)

# Re-declare optimizer in full because paramwise_cfg.custom_keys must include
# the alpha entry. The substring 'image_classifier.alpha' uniquely matches
# pts_bbox_head.image_classifier.alpha (no other parameter shares that path).
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'image_classifier.alpha': dict(lr_mult=0.1),
        }
    ),
)

# Re-declare custom_hooks in full (mmcv configs replace, not merge, list-valued
# top-level keys). Keep the inherited EmptyCacheHook and add AlphaFreezeHook.
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True, priority='HIGH'),
    dict(type='AlphaFreezeHook', freeze_epochs=1),
]
