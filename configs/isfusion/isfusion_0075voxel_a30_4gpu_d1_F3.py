"""F3: zero-sum alpha reparam + confidence-masked alpha-loss.

Builds on F2 (detached-fusion focal loss as alpha's gradient source) and adds
two diagnostic fixes for the F2 observation that alpha collapsed nearly
uniformly toward muting the image term, with per-class differentiation stuck
at the noise floor:

- FIX #1 (zero-sum reparam): the per-class fusion log-weight is now
  ``alpha_log_weight() = alpha_beta + (alpha - alpha.mean())``. Mean-centering
  projects out the global-scale (1-vector) direction so alpha's gradient is
  zero-mean and only the per-class differential is learnable. ``alpha_beta``
  is a separate scalar parameter that holds the global scale; it is
  ``requires_grad=False`` by default (``learn_beta=False``), so the global
  scale stays frozen at softplus(0)=0.6931 and only per-class deviations
  move. The parameter named ``self.alpha`` is preserved (the AlphaFreezeHook,
  the paramwise lr_mult custom_key ``'image_classifier.alpha'``, and an
  external tracker all depend on the name); ``self.alpha`` is now
  semantically the per-class DEVIATION, not the absolute weight.

- FIX #2 (confident_balanced alpha-loss): the alpha-loss is restricted to
  candidates where the pre-fusion BEV head is already confident
  (``bev_conf = s_bev_pre.sigmoid().max(dim=1).values >= alpha_conf_threshold``).
  This removes the low-s_bev recall population — focal loss puts large
  gradient on GT-positive candidates with low s_bev, which pulls alpha toward
  leaning on the image to RECOVER those candidates (the recall axis,
  barrier/traffic_cone). After masking, only the BEV-confident regime — where
  FPs live and the precision question matters — drives alpha's update. Both
  pos and neg are kept within the confident set so alpha is not pushed to
  trivially mute the image term.

NOTE: ``alpha_conf_threshold=0.3`` is a placeholder pending an s_bev-
separation probe — choose by inspecting the distribution of BEV-confident
candidates' max-class sigmoid on a real validation batch and pick a value
that retains most confident FPs while excluding the recall tail. If the
probe shifts the optimum materially, update this config.

NOTE: the paramwise_cfg custom_key substring ``'image_classifier.alpha'``
also matches ``'image_classifier.alpha_beta'``, but ``alpha_beta`` is
constructed with ``requires_grad=False`` (because ``learn_beta=False``), so
the optimizer's DefaultOptimizerConstructor skips it. Harmless.

Other settings inherited from F2:
- consistency loss disabled (weight 0)
- alpha frozen for epoch 1, unfrozen epochs 2-10 via AlphaFreezeHook (the
  hook acts on the parameter literally named ``alpha``, which under FIX #1
  is the deviation tensor)
- alpha learns at 0.1x base LR via paramwise_cfg custom_keys
- samples_per_gpu=2, cumulative_iters=2 (effective batch 16)
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
        alpha_conf_threshold=0.3,
        image_classifier=dict(beta_init=0.0, learn_beta=False),
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
