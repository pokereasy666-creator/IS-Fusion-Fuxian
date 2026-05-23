"""AFDT - Adaptive Fusion Dual Transformer (AFDT-style reconstruction).

Reconstructed from MGAF (Fan et al., "LiDAR-Camera 3D Object Detection With
Multiple Guidance and Adaptive Fusion", IEEE TPAMI, Jan 2026), Sec. III-D,
Eq. (5)-(6), Fig. 9. MGAF code was unreleased, so this is a reconstruction from
the paper text, NOT a copy of the authors' implementation.

Drop-in adaptation for IS-Fusion's HSF fusion point (IS-Fusion paper Eq.6,
B_F = f_conv([B_I, B_P])). It consumes the unequal-channel BEV tensors directly
(LiDAR B_P, image B_I), projects both to a common working width, applies the
Eq.5 adaptive gate and the Eq.6 dual cross-attention, and returns an out_ch BEV
tensor. The conv-concat skip and the learnable gamma scale are applied at the
call site (ISFusionEncoder.forward), NOT inside this module.

Global attention at IS-Fusion's 180x180 BEV (=32400 tokens) is ~1e9 entries per
head and OOMs on 24GB; attn_stride downsamples the grid for the attention op and
the attended output is upsampled back. attn_stride=1 recovers true global
attention (only viable at small H,W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFDT(nn.Module):
    def __init__(self, in_ch_img=256, in_ch_lidar=512, c_work=256, out_ch=128,
                 num_heads=8, attn_stride=4, mlp_ratio=2, max_pos_tokens=2025):
        super().__init__()
        assert c_work % num_heads == 0, "c_work must be divisible by num_heads"
        self.c_work = c_work
        self.attn_stride = attn_stride
        self._max_pos = max_pos_tokens

        # Eq.5 modality projections to a common working width
        self.proj_l = nn.Conv2d(in_ch_lidar, c_work, kernel_size=1)
        self.proj_c = nn.Conv2d(in_ch_img, c_work, kernel_size=1)
        # Eq.5 adaptive gate: W_F = sigmoid(conv1x1(concat[F'_LB, F'_CB])), C ch in [0,1]
        self.gate = nn.Conv2d(c_work * 2, c_work, kernel_size=1)

        # learnable position embedding P on the camera stream (added in token space)
        self.pos = nn.Parameter(torch.zeros(1, max_pos_tokens, c_work))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # Eq.6 dual cross-attention (batch_first: [B, N, C])
        self.attn_c2l = nn.MultiheadAttention(c_work, num_heads, batch_first=True)
        self.attn_l2c = nn.MultiheadAttention(c_work, num_heads, batch_first=True)

        # Eq.6 tail: LN -> MLP -> LN
        self.ln = nn.LayerNorm(c_work)
        self.mlp = nn.Sequential(
            nn.Linear(c_work, c_work * mlp_ratio),
            nn.GELU(),
            nn.Linear(c_work * mlp_ratio, c_work),
        )
        self.out_ln = nn.LayerNorm(c_work)

        # project the working width to the fused BEV width (matches conv_fusion out)
        self.tail = nn.Conv2d(c_work, out_ch, kernel_size=1)

    def _to_tokens(self, x):
        # [B, C, h, w] -> [B, h*w, C]
        B, C, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (h, w)

    def _to_map(self, x, hw):
        # [B, h*w, C] -> [B, C, h, w]
        h, w = hw
        B, N, C = x.shape
        return x.transpose(1, 2).reshape(B, C, h, w)

    def forward(self, F_LB, F_CB):
        """F_LB: LiDAR BEV  [B, in_ch_lidar, H, W]
        F_CB: camera BEV [B, in_ch_img, H, W]
        returns fused BEV [B, out_ch, H, W].
        """
        B, _, H, W = F_LB.shape

        # Eq.5 projections + adaptive gate
        fl = self.proj_l(F_LB)                                     # F'_LB
        fc = self.proj_c(F_CB)                                     # F'_CB
        w = torch.sigmoid(self.gate(torch.cat([fl, fc], dim=1)))   # W_F in [0, 1]

        cam = w * fc                                               # W_F * F'_CB
        lid = (1.0 - w) * fl                                       # (1 - W_F) * F'_LB

        # attention at reduced resolution (INTERP: feasibility on 24GB at 180x180)
        s = self.attn_stride
        if s > 1:
            cam_a = F.avg_pool2d(cam, kernel_size=s, stride=s)
            lid_a = F.avg_pool2d(lid, kernel_size=s, stride=s)
        else:
            cam_a, lid_a = cam, lid

        cam_t, hw = self._to_tokens(cam_a)                         # [B, n, C]
        lid_t, _ = self._to_tokens(lid_a)
        n = cam_t.shape[1]
        assert n <= self._max_pos, (
            f"attn tokens {n} exceed max_pos_tokens {self._max_pos}; "
            f"raise max_pos_tokens or attn_stride")
        cam_t = cam_t + self.pos[:, :n, :]                         # + P

        # Eq.6 branch C: camera query attends LiDAR key/value
        o_c, _ = self.attn_c2l(query=cam_t, key=lid_t, value=lid_t)
        # Eq.6 branch L: LiDAR query attends camera key/value
        o_l, _ = self.attn_l2c(query=lid_t, key=cam_t, value=cam_t)

        # merge by sum, then LN + MLP (Eq.6 tail)
        merged = self.ln(o_c + o_l)
        merged = merged + self.mlp(merged)
        merged = self.out_ln(merged)

        attended = self._to_map(merged, hw)                        # [B, C, h', w']
        if s > 1:
            attended = F.interpolate(attended, size=(H, W),
                                     mode="bilinear", align_corners=False)
        return self.tail(attended)                                 # [B, out_ch, H, W]
