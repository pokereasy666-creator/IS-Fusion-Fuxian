"""Image classification head for Direction 1 (image-aware FP suppression).

This module reads multi-view image features at the projected 2D locations of
BEV-derived candidates and produces per-class image logits ``s_img``. These
logits are combined with BEV scores via additive logit fusion in
:func:`fuse_scores_additive` to form the final detection confidence on the
candidate's pre-assigned class.

Projection follows the augment-aware pattern used in
``mmdet3d/models/middle_encoders/fusion_encoder.py:968``:
inverse-LiDAR-aug -> project via original ``lidar2img`` -> forward image-aug.

Coordinate conventions match the rest of the IS-Fusion repo:
``query_pos`` arrives in BEV grid units of the AUGMENTED LiDAR frame. We
convert to metric coordinates via ``out_size_factor * voxel_size``, undo the
LiDAR augmentation, project per-view through ``lidar2img``, divide by depth,
then apply the image augmentation to land in augmented-image pixel space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class ImageClassifierHead(BaseModule):
    """Per-candidate image classification head.

    See module docstring for the projection pattern. Returns per-class logits
    plus an in-any-view boolean for downstream masking.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        hidden_channels: int = 256,
        num_views: int = 6,
        feature_stride: int = 8,
        img_shape: tuple = (384, 1056),
        out_size_factor: int = 8,
        voxel_size: tuple = (0.075, 0.075),
        pc_range: tuple = (-54.0, -54.0),
        alpha_init: float = 0.0,
        depth_min: float = 0.5,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_views = num_views
        self.feature_stride = feature_stride
        self.img_shape = tuple(img_shape)
        self.out_size_factor = out_size_factor
        self.voxel_size = tuple(voxel_size)
        self.pc_range = tuple(pc_range)
        self.depth_min = depth_min

        self.feature_proj = nn.Linear(in_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes),
        )

        # Per-class learnable alpha, stored pre-softplus.
        self.alpha = nn.Parameter(torch.full((num_classes,), float(alpha_init)))
        # Per-class default logit for candidates outside every camera view.
        self.out_of_view_default = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self,
        img_feats: torch.Tensor,
        query_pos: torch.Tensor,
        query_height: torch.Tensor,
        lidar2img: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        lidar_aug_matrix: torch.Tensor,
    ) -> dict:
        """Compute per-candidate per-class image logits.

        Args:
            img_feats: ``[B*V, C, H_feat, W_feat]`` stride-8 image features.
            query_pos: ``[B, K, 2]`` BEV grid units in the augmented frame.
            query_height: ``[B, 1, K]`` predicted z (meters), augmented frame.
            lidar2img: ``[B, V, 4, 4]`` original calibration.
            img_aug_matrix: ``[B, V, 4, 4]`` image augmentation.
            lidar_aug_matrix: ``[B, 4, 4]`` lidar augmentation per sample.

        Returns:
            dict with ``logits`` ``[B, K, num_classes]`` and ``in_any_view``
            ``[B, K]`` boolean.
        """
        B, K, _ = query_pos.shape
        V = self.num_views
        device = query_pos.device

        # --- Step 1: BEV grid -> metric (augmented LiDAR frame).
        sx = self.out_size_factor * self.voxel_size[0]
        sy = self.out_size_factor * self.voxel_size[1]
        x_aug = query_pos[..., 0] * sx + self.pc_range[0]
        y_aug = query_pos[..., 1] * sy + self.pc_range[1]
        z_aug = query_height.squeeze(1)  # [B, K]
        query_metric_aug = torch.stack([x_aug, y_aug, z_aug], dim=-1)  # [B, K, 3]

        # --- Step 2: invert LiDAR augmentation -> original LiDAR frame.
        inv_lidar_aug = torch.inverse(lidar_aug_matrix.float()).to(query_metric_aug.dtype)
        ones_BK = torch.ones(B, K, 1, device=device, dtype=query_metric_aug.dtype)
        query_aug_h = torch.cat([query_metric_aug, ones_BK], dim=-1)  # [B, K, 4]
        # out[b,k,j] = sum_i query_aug_h[b,k,i] * inv_lidar_aug[b,j,i]
        query_orig_h = torch.einsum('bki,bji->bkj', query_aug_h, inv_lidar_aug)

        # --- Step 3: project to all views via lidar2img.
        lidar2img = lidar2img.to(query_orig_h.dtype)
        # projected[b,v,k,i] = sum_j query_orig_h[b,k,j] * lidar2img[b,v,i,j]
        projected = torch.einsum('bkj,bvij->bvki', query_orig_h, lidar2img)
        depths = projected[..., 2]  # [B, V, K]
        safe_depths = torch.clamp(depths, min=1e-3)
        pixel_uv_orig = projected[..., :2] / safe_depths[..., None]  # [B, V, K, 2]

        # --- Step 4: apply image augmentation in pixel space.
        img_aug_matrix = img_aug_matrix.to(pixel_uv_orig.dtype)
        ones_BVK = torch.ones(B, V, K, 1, device=device, dtype=pixel_uv_orig.dtype)
        pixel_h = torch.cat(
            [pixel_uv_orig, depths[..., None], ones_BVK], dim=-1
        )  # [B, V, K, 4]
        # out[b,v,k,i] = sum_j pixel_h[b,v,k,j] * img_aug_matrix[b,v,i,j]
        pixel_aug = torch.einsum('bvkj,bvij->bvki', pixel_h, img_aug_matrix)
        pixel_uv = pixel_aug[..., :2]  # [B, V, K, 2]

        # --- Step 5: visibility mask.
        H, W = self.img_shape
        u = pixel_uv[..., 0]
        v_pix = pixel_uv[..., 1]
        in_view = (
            (depths >= self.depth_min)
            & (u >= 0)
            & (u < W)
            & (v_pix >= 0)
            & (v_pix < H)
        )

        # --- Step 6: sample image features via grid_sample.
        u_norm = (u / max(W - 1, 1)) * 2 - 1
        v_norm = (v_pix / max(H - 1, 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1)  # [B, V, K, 2]

        BV, C, H_feat, W_feat = img_feats.shape
        assert BV == B * V, (
            f'img_feats batch {BV} does not match B*V={B}*{V}={B * V}')
        img_feats_flat = img_feats.reshape(B * V, C, H_feat, W_feat)
        grid_flat = grid.reshape(B * V, K, 1, 2)

        sampled = F.grid_sample(
            img_feats_flat,
            grid_flat,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )  # [B*V, C, K, 1]
        sampled = sampled.view(B, V, C, K).permute(0, 1, 3, 2)  # [B, V, K, C]

        # --- Step 7: max-pool aggregate across views.
        invalid_mask = ~in_view  # [B, V, K]
        sampled_masked = sampled.masked_fill(invalid_mask[..., None], float('-inf'))
        aggregated, _ = sampled_masked.max(dim=1)  # [B, K, C]

        any_valid = in_view.any(dim=1)  # [B, K]
        aggregated = torch.where(
            any_valid[..., None], aggregated, torch.zeros_like(aggregated)
        )

        # --- Step 8: per-class logits.
        h = self.feature_proj(aggregated)
        h = F.relu(h)
        logits = self.classifier(h)  # [B, K, num_classes]

        default_logits = self.out_of_view_default.expand(B, K, -1)
        logits = torch.where(any_valid[..., None], logits, default_logits)

        return dict(logits=logits, in_any_view=any_valid)


def fuse_scores_additive(
    s_bev_logits: torch.Tensor,
    s_img_logits: torch.Tensor,
    alpha_pre_softplus: torch.Tensor,
    in_any_view: torch.Tensor,
) -> torch.Tensor:
    """Additive logit-space fusion.

    ``s_final_logit(c) = s_bev_logit(c) + alpha(c) * s_img_logit(c)`` with
    ``alpha(c) = softplus(alpha_pre_softplus(c)) >= 0``.

    NOTE: this is NOT mathematically equivalent to probability multiplication.
    It is a learned linear combination in logit space; treat it as a heuristic
    fusion rule whose effectiveness is empirical.

    For out-of-view candidates returns ``s_bev_logits`` unchanged.

    Args:
        s_bev_logits: ``[B, num_classes, K]``.
        s_img_logits: ``[B, K, num_classes]``.
        alpha_pre_softplus: ``[num_classes]``.
        in_any_view: ``[B, K]`` boolean.

    Returns:
        Fused logits ``[B, num_classes, K]``.
    """
    alpha_pos = F.softplus(alpha_pre_softplus)  # [num_classes]
    s_img_aligned = s_img_logits.permute(0, 2, 1)  # [B, num_classes, K]
    fused = s_bev_logits + alpha_pos[None, :, None] * s_img_aligned

    # Out-of-view: pass through s_bev unchanged.
    fused = torch.where(in_any_view[:, None, :], fused, s_bev_logits)
    return fused
