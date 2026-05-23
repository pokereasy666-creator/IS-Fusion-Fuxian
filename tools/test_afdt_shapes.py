"""Torch-only shape + identity-init gate for the AFDT fusion module.

Skips cleanly (exit 0) if torch is absent, so it is safe to run on an offline /
CPU-only box. Does NOT run training, inference, or the full IS-Fusion model.

The AFDT module is loaded directly by file path so this test only needs torch
(importing mmdet3d.models.middle_encoders would drag in mmcv/spconv via the
package __init__).
"""

import os
import sys

try:
    import torch
    import torch.nn as nn
except Exception as e:  # torch not installed
    print("SKIP test_afdt_shapes: torch not available (%s)" % e)
    sys.exit(0)

import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_AFDT_PATH = os.path.join(_HERE, os.pardir, "mmdet3d", "models",
                          "middle_encoders", "afdt.py")


def _load_afdt():
    spec = importlib.util.spec_from_file_location("afdt_standalone", _AFDT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AFDT


def main():
    AFDT = _load_afdt()
    torch.manual_seed(0)

    # matches the integration: AFDT(in_ch_img=256, in_ch_lidar=512, out 128) at
    # IS-Fusion's 180x180 BEV; attn_stride=4 -> 45x45=2025 tokens == max_pos_tokens.
    m = AFDT(in_ch_img=256, in_ch_lidar=512, c_work=256, out_ch=128,
             num_heads=8, attn_stride=4, max_pos_tokens=2025).eval()

    lidar = torch.randn(2, 512, 180, 180)   # B_P (LiDAR BEV)
    img = torch.randn(2, 256, 180, 180)     # B_I (camera BEV)

    with torch.no_grad():
        afdt_out = m(lidar, img)
    assert tuple(afdt_out.shape) == (2, 128, 180, 180), tuple(afdt_out.shape)
    print("PASS shape: AFDT(F_LB[2,512,180,180], F_CB[2,256,180,180]) -> %s"
          % (tuple(afdt_out.shape),))

    # identity-init: bev_feats = conv_concat_skip + gamma(=0) * afdt  must == skip.
    # (mirrors ISFusionEncoder.forward; holds for any skip when gamma == 0.)
    skip_conv = nn.Conv2d(768, 128, kernel_size=3, padding=1)
    afdt_gamma = nn.Parameter(torch.zeros(1))
    with torch.no_grad():
        skip = skip_conv(torch.cat([img, lidar], dim=1))   # [2,128,180,180]
        fused = skip + afdt_gamma * afdt_out
    assert torch.allclose(fused, skip), "identity-init broken: gamma=0 must give skip"
    print("PASS identity-init: gamma=0 => fused == conv-concat skip "
          "(max|fused-skip|=%.3g)" % (fused - skip).abs().max().item())

    print("\nAll AFDT shape/identity gates passed.")


if __name__ == "__main__":
    main()
