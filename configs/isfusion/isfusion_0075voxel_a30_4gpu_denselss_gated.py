_base_ = ['isfusion_0075voxel_a30_4gpu.py']

# Occupancy-gated fill: dense LSS B_I fills ONLY the cells the sparse
# img_fv_to_bev B_I left empty (no LiDAR projection); populated cells pass through
# untouched. conv_fusion stays 768->128. Tests whether dense COVERAGE in the holes
# helps, without disturbing the precise sparse signal.
model = dict(fusion_encoder=dict(dense_image_bev_mode='gated'))
