_base_ = ['isfusion_0075voxel_a30_4gpu.py']
# Standalone +SHC ablation row.
#
# Inherits the clean A30 4-GPU baseline (gradient-accumulation training, paper LR)
# and only enables the MGAF-style "additional downsampling + sparse height
# compression" branch inside SparseEncoder via use_shc=True. The fusion path stays
# the baseline hardcoded conv fusion (there is no fusion_type on this baseline), and
# SHC holds B_P at 512ch @ 180x180, so no downstream module changes are required.
model = dict(pts_middle_encoder=dict(use_shc=True))
