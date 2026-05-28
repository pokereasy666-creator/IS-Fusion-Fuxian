_base_ = ['isfusion_0075voxel_a30_4gpu.py']

# Isolation row (e): dense LSS image->BEV branch only -- no AFDT, no SHC,
# conv-fusion preserved. This flips the default-OFF gate added to
# ISFusionEncoder; mmcv deep-merges the fusion_encoder dict, so every other
# base setting is inherited unchanged. The baseline configs are untouched.
model = dict(fusion_encoder=dict(use_dense_image_bev=True))
