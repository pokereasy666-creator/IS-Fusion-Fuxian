#!/usr/bin/env python
"""Server-runnable shape gate for DenseLSSBranch.

Guarded so it runs anywhere: if torch / CUDA / the compiled ``bev_pool_ext`` are
absent (e.g. CI), it prints a SKIP line and exits 0. On a GPU server it builds
the production-config branch, runs a synthetic forward, and asserts the
``[1, 256, 180, 180]`` output. A shape mismatch raises (never swallowed) so a
real regression fails the gate.
"""
import os
import sys

# make `import mmdet3d` work when run as `python tools/test_dense_lss_shapes.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _skip(reason):
    print("SKIP: {}; real shape check runs on GPU server".format(reason))
    sys.exit(0)


def main():
    try:
        import torch
    except ImportError as exc:
        _skip("torch not importable ({})".format(exc))

    if not torch.cuda.is_available():
        _skip("CUDA not available")

    try:
        # importing the branch transitively imports the compiled bev_pool_ext
        from mmdet3d.ops.bev_pool import bev_pool  # noqa: F401
        from mmdet3d.models.middle_encoders.dense_lss import DenseLSSBranch
    except ImportError as exc:
        _skip("bev_pool_ext / DenseLSSBranch not importable ({})".format(exc))

    device = torch.device("cuda")
    branch = (
        DenseLSSBranch(
            in_channels=256,
            out_channels=256,
            image_size=(384, 1056),
            feature_size=(24, 66),
            xbound=[-54.0, 54.0, 0.6],
            ybound=[-54.0, 54.0, 0.6],
            zbound=[-5.0, 3.0, 8.0],
            dbound=[1.0, 60.0, 0.5],
        )
        .to(device)
        .eval()
    )

    B, N = 1, 6
    img_feats = torch.randn(B, N, 256, 24, 66, device=device)
    # one sample: [N_points, 5] = xyz + 2 dummy (intensity, time)
    points = [torch.randn(5000, 5, device=device)]

    # identity rotations + zero translations, sized [B, N, 4, 4] / [B, 4, 4]
    eye = torch.eye(4, device=device)
    lidar2img = eye.view(1, 1, 4, 4).repeat(B, N, 1, 1).contiguous()
    img_aug_matrix = eye.view(1, 1, 4, 4).repeat(B, N, 1, 1).contiguous()
    camera2lidar = eye.view(1, 1, 4, 4).repeat(B, N, 1, 1).contiguous()
    camera_intrinsics = eye.view(1, 1, 4, 4).repeat(B, N, 1, 1).contiguous()
    lidar_aug_matrix = eye.view(1, 4, 4).repeat(B, 1, 1).contiguous()

    with torch.no_grad():
        out = branch(
            img_feats,
            points,
            lidar2img=lidar2img,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
            camera2lidar=camera2lidar,
            camera_intrinsics=camera_intrinsics,
        )

    # do NOT wrap in try/except: a real shape regression must fail the gate
    assert tuple(out.shape) == (1, 256, 180, 180), (
        "unexpected DenseLSSBranch output shape: {}".format(tuple(out.shape))
    )
    print("PASS: DenseLSSBranch output shape {}".format(tuple(out.shape)))


if __name__ == "__main__":
    main()
