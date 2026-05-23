_base_ = ['isfusion_0075voxel_a30_4gpu.py']
# A/B variant of the 4-GPU reproduction baseline: swap the HSF conv-concat fusion
# for AFDT (Adaptive Fusion Dual Transformer). Identity-init (afdt_gamma=0) means
# training starts exactly at the conv-concat baseline. Only fusion_type is changed;
# everything else (model, data, optimizer, schedule) is inherited so the A/B runs
# share the same base.

model = dict(
    fusion_encoder=dict(fusion_type='afdt'),
)
