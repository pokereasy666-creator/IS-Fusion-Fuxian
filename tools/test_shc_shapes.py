"""Shape-contract test for the SHC path in SparseEncoder.

This is environment-guarded: it prints ``SKIP`` and exits 0 when torch, spconv,
or CUDA are unavailable (spconv sparse convolutions require a GPU), so it is safe
to run in a CPU/CI box without dependencies. The real assertion runs on the GPU
server, where it checks that the ``use_shc=True`` forward returns a dense
``spatial_features`` of shape ``[B, 512, 180, 180]`` -- identical to the baseline
B_P contract (output_channels(256) * D(2) = 512 channels at 180x180).
"""
import sys


def _skip(msg):
    print('SKIP: ' + msg)
    sys.exit(0)


def main():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        _skip('torch unavailable ({})'.format(e))

    try:
        from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder
        from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE  # noqa: F401
    except Exception as e:  # noqa: BLE001
        _skip('mmdet3d/spconv import failed ({})'.format(e))

    try:
        import spconv  # noqa: F401
    except Exception as e:  # noqa: BLE001
        _skip('spconv not installed; real shape check runs on the GPU server '
              '({})'.format(e))

    if not torch.cuda.is_available():
        _skip('CUDA unavailable; spconv convs need a GPU. Real shape check '
              'runs on the GPU server.')

    device = 'cuda'

    def build(use_shc):
        return SparseEncoder(
            in_channels=64,
            sparse_shape=[41, 1440, 1440],
            base_channels=32,
            output_channels=256,
            order=('conv', 'norm', 'act'),
            encoder_channels=((32, 32, 64), (64, 64, 128),
                              (128, 128, 256), (256, 256)),
            encoder_paddings=((0, 0, 1), (0, 0, 1),
                              (0, 0, [0, 1, 1]), (0, 0)),
            block_type='basicblock',
            use_shc=use_shc,
        ).to(device).eval()

    # Synthetic sparse input: a handful of unique voxels in (batch, z, y, x).
    torch.manual_seed(0)
    num_voxels = 256
    batch_size = 1
    z = torch.randint(0, 41, (num_voxels, 1))
    y = torch.randint(0, 1440, (num_voxels, 1))
    x = torch.randint(0, 1440, (num_voxels, 1))
    coors = torch.cat([z, y, x], dim=1)
    coors = torch.unique(coors, dim=0)              # drop duplicate cells
    b = torch.zeros((coors.shape[0], 1), dtype=coors.dtype)
    coors = torch.cat([b, coors], dim=1).int().to(device)
    voxel_features = torch.randn(coors.shape[0], 64, device=device)

    expected = (batch_size, 512, 180, 180)
    for use_shc in (False, True):
        enc = build(use_shc)
        with torch.no_grad():
            spatial_features, encode_features, _ = enc(
                voxel_features, coors, batch_size)
        got = tuple(spatial_features.shape)
        assert got == expected, (
            'use_shc={}: expected spatial_features {}, got {}'.format(
                use_shc, expected, got))
        print('OK use_shc={}: spatial_features {}'.format(use_shc, got))

    print('PASS: SHC preserves the B_P contract [B, 512, 180, 180].')


if __name__ == '__main__':
    main()
