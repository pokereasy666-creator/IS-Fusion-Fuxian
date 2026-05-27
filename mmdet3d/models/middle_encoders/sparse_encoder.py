# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
# spconv v1
# from mmdet3d.ops import spconv as spconv

from ..builder import MIDDLE_ENCODERS

from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:  # spconv v1
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module',
                 use_shc=False,
                 **kwargs):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False

        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(  # autoalign
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

        self.use_shc = use_shc
        if self.use_shc:
            self._build_shc(norm_cfg)

    def _build_shc(self, norm_cfg):
        """MGAF-style SHC modules (built only when use_shc=True).

        Two extra XY-only stride-2 sparse stages beyond the deepest encoder
        stage, each height-compressed like conv_out, then fused back to the
        baseline B_P width (output_channels * D = 512 @ 180x180).
        """
        oc = self.output_channels
        self.shc_down5 = make_sparse_convmodule(
            oc, oc, kernel_size=3, stride=(1, 2, 2), norm_cfg=norm_cfg,
            padding=1, indice_key='spconv_shc5', conv_type='SparseConv3d')
        self.shc_down6 = make_sparse_convmodule(
            oc, oc, kernel_size=3, stride=(1, 2, 2), norm_cfg=norm_cfg,
            padding=1, indice_key='spconv_shc6', conv_type='SparseConv3d')
        self.shc_out5 = make_sparse_convmodule(
            oc, oc, kernel_size=(3, 1, 1), stride=(2, 1, 1), norm_cfg=norm_cfg,
            padding=0, indice_key='spconv_shc_down5', conv_type='SparseConv3d')
        self.shc_out6 = make_sparse_convmodule(
            oc, oc, kernel_size=(3, 1, 1), stride=(2, 1, 1), norm_cfg=norm_cfg,
            padding=0, indice_key='spconv_shc_down6', conv_type='SparseConv3d')
        # D = 2 for sparse_shape z=41 -> per-scale dense BEV = output_channels*2.
        bev_ch = oc * 2
        self.shc_fuse = nn.Sequential(
            nn.Conv2d(bev_ch * 3, bev_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(inplace=True))

    def _shc_fuse_forward(self, sparse_feat, bev_f4):
        """Fuse multi-scale sparse-height-compressed BEVs into bev_f4's width.

        sparse_feat is encode_features[-1] (XY-stride 8, z=5, output_channels);
        bev_f4 is the baseline dense spatial_features (N, output_channels*D,
        180, 180). F5/F6 add stride-16/32 scales, are height-compressed and
        upsampled to 180x180, then the concatenation is projected back to
        exactly bev_f4's channel width (the preserved B_P contract).
        """
        sparse_f5 = self.shc_down5(sparse_feat)   # XY 180 -> 90 (z kept)
        sparse_f6 = self.shc_down6(sparse_f5)      # XY 90 -> 45 (z kept)

        out5 = self.shc_out5(sparse_f5).dense()    # (N, C, D, 90, 90)
        n, c, d, h, w = out5.shape
        bev_f5 = out5.view(n, c * d, h, w)
        out6 = self.shc_out6(sparse_f6).dense()    # (N, C, D, 45, 45)
        n, c, d, h, w = out6.shape
        bev_f6 = out6.view(n, c * d, h, w)

        size = bev_f4.shape[-2:]
        bev_f5 = F.interpolate(
            bev_f5, size=size, mode='bilinear', align_corners=False)
        bev_f6 = F.interpolate(
            bev_f6, size=size, mode='bilinear', align_corners=False)

        return self.shc_fuse(torch.cat([bev_f4, bev_f5, bev_f6], dim=1))

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size, swin_format=False, img_feats=None, **kwargs):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """

        coors = coors.int()  # must be type int
        input_sp_tensor = SparseConvTensor(voxel_features, coors,          # spconv 2
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        encode_features.append(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        if self.use_shc:
            spatial_features = self._shc_fuse_forward(
                encode_features[-1], spatial_features)

        return spatial_features, encode_features, kwargs


    # spconv 2.0
    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

