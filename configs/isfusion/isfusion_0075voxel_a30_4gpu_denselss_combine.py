_base_ = ['isfusion_0075voxel_a30_4gpu.py']

# Ablation row (e', combine): sparse img_fv_to_bev B_I AND dense LSS B_I are both
# computed, concatenated (512ch), and merged back to 256ch by a norm-only 1x1
# conv (dense_merge) before conv_fusion. conv_fusion stays 768->128. Isolates the
# ADDITIVE contribution of dense coverage on top of the precise sparse path.
# mmcv deep-merges fusion_encoder; all other base settings inherited unchanged.
model = dict(fusion_encoder=dict(dense_image_bev_mode='combine'))
