_base_ = ['isfusion_0075voxel_a30_4gpu.py']

# Density-gated fill: dense LSS B_I fills cells with < density_tau LiDAR points;
# well-scanned cells keep the precise sparse B_I. Targets weak-class b1/loc_center.
model = dict(fusion_encoder=dict(dense_image_bev_mode='density_gated', density_tau=10.0))
