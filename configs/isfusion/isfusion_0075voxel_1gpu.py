_base_ = './isfusion_0075voxel.py'

# Single-GPU config for NVIDIA A30 (24GB HBM2e)
# Adjusted batch size and learning rate for 1 GPU training

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

# Scale learning rate: original is 0.0001 for 8 GPUs * 4 samples = 32 effective batch
# For 1 GPU * 2 samples = 2 effective batch
optimizer = dict(lr=0.00002)

gpu_ids = range(0, 1)
