_base_ = ['isfusion_0075voxel_a30_4gpu.py']
# Ablation: AFDT as the SOLE fusion operator, replacing the HSF conv-concat
# (no conv_fusion contribution, no gamma skip, no identity-init). Unlike the
# _afdt residual variant, this does NOT start at the baseline; judge by final mAP.
# Only fusion_type is changed; everything else is inherited so the runs share
# the same base.

model = dict(
    fusion_encoder=dict(fusion_type='afdt_replace'),
)
