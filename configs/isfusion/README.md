## IS-Fusion configs

| Config | Hardware | Effective batch | LR | Notes |
|---|---|---|---|---|
| `isfusion_0075voxel.py` | 8x A100 | 16 | 1e-4 | Official paper config; reference. |
| `isfusion_0075voxel_a30.py` | 3x A30 | 3 | 1.875e-5 | Initial A30 attempt; 67.4 mAP val. Underperforms paper; LR scaling was incorrect. |
| `isfusion_0075voxel_a30_4gpu.py` | 4x A30 | 16 (via grad accum) | 1e-4 | Recommended A30 config; matches paper's effective batch size and LR. |
