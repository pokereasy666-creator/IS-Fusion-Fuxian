# Copyright (c) 2026 IS-Fusion contributors.
#
# This file is adapted from BEVFusion (https://github.com/mit-han-lab/bevfusion),
# Copyright (c) 2022 MIT HAN Lab, released under the Apache License, Version 2.0.
# It adapts two donor classes:
#   * BaseDepthTransform  - mmdet3d/models/vtransforms/base.py
#   * DepthLSSTransform   - mmdet3d/models/vtransforms/depth_lss.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dense LSS-style image->BEV branch for IS-Fusion.

A self-contained Lift-Splat-Shoot view transformer (DepthNet + per-camera depth
distribution + outer product + ``bev_pool`` splat) adapted from BEVFusion's
``DepthLSSTransform``/``BaseDepthTransform``. It produces a dense ``[B, 256, 180,
180]`` image-BEV tensor that drop-in replaces IS-Fusion's sparse
``img_fv_to_bev`` output (B_I) when ``use_dense_image_bev=True``.

Four adaptations vs. the donor (each tagged ``# ADAPTATION (n)`` below):
  (1) Stride-16 ``dtransform`` (donor is stride-8): one extra stride-2 conv stage
      so 384x1056 -> 24x66 (the IS-Fusion stride-16 feature grid).
  (2) ``add_depth_features=False`` / scalar depth: the depth tensor is a single
      channel (raw range), matching the 1-channel ``dtransform`` input.
  (3) ``height_expand=False``: LiDAR has accurate height, so the donor's 8x point
      duplication (used for radar) is skipped.
  (4) X/Y -> H/W swap in ``bev_pool``: IS-Fusion's LiDAR BEV is H=Y, W=X
      (SparseEncoder ``sparse_shape=[z, y, x]``), so the donor's (X, Y) column
      order is swapped to (Y, X) and the pooled grid is sized H=nx[1], W=nx[0].
"""
from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

# IS-Fusion's existing (previously unused) CUDA scatter op. Importing this pulls
# the compiled ``bev_pool_ext``; on a box without it (e.g. CI) importing this
# module raises ImportError, which the server-runnable test guards against.
from mmdet3d.ops.bev_pool import bev_pool

__all__ = ["DenseLSSBranch"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class DenseLSSBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        image_size: Tuple[int, int] = (384, 1056),
        feature_size: Tuple[int, int] = (24, 66),
        xbound=[-54.0, 54.0, 0.6],
        ybound=[-54.0, 54.0, 0.6],
        zbound=[-5.0, 3.0, 8.0],
        dbound=[1.0, 60.0, 0.5],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        # ADAPTATION (2): scalar 1-channel depth; no extra per-point features.
        self.depth_input = "scalar"
        self.add_depth_features = False
        # ADAPTATION (3): LiDAR has accurate height -> no point height-expansion.
        self.height_expand = False

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.register_buffer("dx", dx.float())  # cell size [X, Y, Z]
        self.register_buffer("bx", bx.float())  # grid origin [X, Y, Z]
        self.register_buffer("nx", nx.float())  # #bins [X=180, Y=180, Z=1]

        self.C = out_channels
        self.register_buffer("frustum", self.create_frustum())
        self.D = self.frustum.shape[0]  # depth bins from dbound (118)
        self.fp16_enabled = False

        # ADAPTATION (1): stride-16 dtransform. The donor stops at /8 (48x132);
        # the final stride-2 conv (marked NEW) takes 384x1056 -> 24x66 so the
        # depth-context map matches the stride-16 image feature grid.
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),   # /4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # /2 -> /8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # /2 -> /16 (NEW)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post (image-aug) transformation: B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # camera -> lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        # apply lidar augmentation (raw-LiDAR -> augmented-LiDAR), so the pooled
        # BEV lives in the same frame as pc_range / the LiDAR BEV grid.
        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    @force_fp32()
    def get_depth(self, points, lidar2img, img_aug_matrix, lidar_aug_matrix):
        """Inline per-camera depth GT, same projection chain as IS-Fusion's
        ``img_point_sampling`` (fusion_encoder.py:984-1013): inverse-lidar-aug ->
        lidar2img -> perspective divide -> image-aug -> scatter range into the
        nearest pixel. Returns ``[B, N, 1, iH, iW]``. Kept as a standalone method
        so ``tools/viz_dense_lss_depth_gt.py`` can verify the exact code path."""
        # unwrap possible DataContainer-style [tensor] wrapping (as img_point_sampling does)
        if isinstance(lidar2img, list):
            lidar2img = lidar2img[0]
        if isinstance(img_aug_matrix, list):
            img_aug_matrix = img_aug_matrix[0]
        if isinstance(lidar_aug_matrix, list):
            lidar_aug_matrix = lidar_aug_matrix[0]

        batch_size = len(points)
        num_cam = img_aug_matrix.shape[1]
        # ADAPTATION (2): single scalar-depth channel (no one-hot / depth feats).
        depth = torch.zeros(
            batch_size, num_cam, 1, *self.image_size, device=points[0].device
        )

        # ADAPTATION (3): no height_expand loop here (donor duplicates radar pts 8x).
        for b in range(batch_size):
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2img[b]

            # clone so the threaded raw `points` list is never mutated in place
            # (mirrors img_point_sampling, fusion_encoder.py:978).
            cur_coords = points[b][:, :3].clone()

            # inverse lidar aug: augmented-LiDAR -> raw-LiDAR
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image (raw-LiDAR -> camera ray, pre-img-aug)
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # perspective divide -> pixel coords (clone dist to keep true range)
            dist = cur_coords[:, 2, :].clone()
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            # image aug -> network-input pixel
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            # to (row, col)
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        return depth

    @force_fp32()
    def get_cam_feats(self, x, d):
        # x: [B, N, C, fH, fW] image feats ; d: [B, N, 1, iH, iW] depth GT
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)             # ADAPTATION (1): -> [B*N, 64, fH, fW]
        x = torch.cat([d, x], dim=1)       # [B*N, in_channels+64, fH, fW]
        x = self.depthnet(x)               # [B*N, D + C, fH, fW]

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)    # [B, N, D, fH, fW, C]
        return x

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten features
        x = x.reshape(Nprime, C)

        # frustum metric coords -> integer voxel indices, columns [X, Y, Z]
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # [N, 4] -> [X, Y, Z, b]

        # keep only points inside the grid (cols X<nx[0], Y<nx[1], Z<nx[2])
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < int(self.nx[1]))
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < int(self.nx[0]))
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < int(self.nx[2]))
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # ADAPTATION (4): X/Y -> H/W swap. bev_pool maps coords col0->H, col1->W
        # (DONOR_REQUIREMENTS Q2). The donor lays out (X, Y) and pools H=nx[0]=X,
        # W=nx[1]=Y. IS-Fusion's LiDAR BEV is H=Y, W=X (sparse_shape=[z, y, x]),
        # so swap the X/Y columns -> (Y, X, Z, b) AND pool with H=nx[1], W=nx[0],
        # keeping the camera BEV aligned (no silent transpose/flip).
        geom_feats = geom_feats[:, [1, 0, 2, 3]]
        x = bev_pool(x, geom_feats, B, int(self.nx[2]), int(self.nx[1]), int(self.nx[0]))

        # collapse Z (single bin) -> [B, C, H, W]
        final = torch.cat(x.unbind(dim=2), 1)
        return final

    @force_fp32()
    def forward(
        self,
        img_feats,
        points,
        lidar2img,
        img_aug_matrix,
        lidar_aug_matrix,
        camera2lidar,
        camera_intrinsics,
    ):
        # inline depth GT from raw points + calibration (same chain as IS-Fusion)
        depth = self.get_depth(points, lidar2img, img_aug_matrix, lidar_aug_matrix)

        # unwrap possible DataContainer-style [tensor] wrapping for geometry
        if isinstance(camera2lidar, list):
            camera2lidar = camera2lidar[0]
        if isinstance(camera_intrinsics, list):
            camera_intrinsics = camera_intrinsics[0]
        iam = img_aug_matrix[0] if isinstance(img_aug_matrix, list) else img_aug_matrix
        lam = lidar_aug_matrix[0] if isinstance(lidar_aug_matrix, list) else lidar_aug_matrix

        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = iam[..., :3, :3]
        post_trans = iam[..., :3, 3]
        extra_rots = lam[..., :3, :3]
        extra_trans = lam[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img_feats, depth)  # [B, N, D, fH, fW, C]
        x = self.bev_pool(geom, x)                # [B, C, 180, 180]
        return x.type_as(img_feats)
