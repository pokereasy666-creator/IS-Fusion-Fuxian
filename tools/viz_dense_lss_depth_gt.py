#!/usr/bin/env python
"""Depth-GT calibration verifier for the dense LSS image-BEV branch.

This is the silent-failure check: a column-swap or augmentation mismatch in the
dense branch does NOT raise an exception, it just splats image features onto the
wrong BEV cells. To catch that, this tool reconstructs the dense branch's inline
depth GT for one val sample (via the branch's own ``get_depth`` method, i.e. the
exact production code path) and overlays the projected LiDAR points -- colored by
range -- back onto the 6 camera images.

  -> /tmp/dense_lss_depth_check_<sample>/cam{0..5}.png

The integrator must visually confirm the points land ON real surfaces (cars,
road, walls). If they are globally offset / transposed, the calibration wiring
is wrong and the dense branch will silently misplace features -- fix before any
training.

Run on the GPU server (needs nuScenes val data + the model env). It is tolerant
of a bare environment: if torch / matplotlib / the branch import are unavailable
it prints SKIP and exits 0.

Usage:
    python tools/viz_dense_lss_depth_gt.py [--sample 0] \\
        [--config configs/isfusion/isfusion_0075voxel_a30_4gpu_denselss.py]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CONFIG = "configs/isfusion/isfusion_0075voxel_a30_4gpu_denselss.py"
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def _skip(reason):
    print("SKIP: {}; depth-GT viz runs on the GPU server".format(reason))
    sys.exit(0)


def _unwrap(x):
    """Drill through mmcv DataContainer(.data) and single-element aug-lists
    (MultiScaleFlipAug3D wraps every key in a 1-element list) to the payload."""
    while hasattr(x, "data"):
        x = x.data
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="val sample index")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch  # noqa: F401
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mmcv import Config
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models.middle_encoders.dense_lss import DenseLSSBranch
    except ImportError as exc:
        _skip("missing dependency ({})".format(exc))

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    data = dataset[args.sample]

    # --- pull the inputs the dense branch consumes (robust to DC / aug-list) ---
    points = _unwrap(data["points"]).float()
    if points.dim() != 2:
        raise RuntimeError("expected points [N, >=3], got {}".format(tuple(points.shape)))

    def _mat(key, ndim):
        m = _unwrap(data[key]).float()
        # ensure a leading batch dim so get_depth can index [b]
        while m.dim() < ndim:
            m = m.unsqueeze(0)
        return m

    lidar2img = _mat("lidar2img", 4)            # [B, N, 4, 4]
    img_aug_matrix = _mat("img_aug_matrix", 4)  # [B, N, 4, 4]
    lidar_aug_matrix = _mat("lidar_aug_matrix", 3)  # [B, 4, 4]

    imgs = _unwrap(data["img"]).float()         # [N, 3, H, W] (or [B, N, 3, H, W])
    if imgs.dim() == 5:
        imgs = imgs[0]
    num_cam, _, img_h, img_w = imgs.shape

    # --- the production depth-GT code path (CPU is fine) ---
    branch = DenseLSSBranch(
        in_channels=256,
        out_channels=256,
        image_size=(img_h, img_w),
        feature_size=(24, 66),
        xbound=[-54.0, 54.0, 0.6],
        ybound=[-54.0, 54.0, 0.6],
        zbound=[-5.0, 3.0, 8.0],
        dbound=[1.0, 60.0, 0.5],
    ).eval()

    with torch.no_grad():
        depth = branch.get_depth([points], lidar2img, img_aug_matrix, lidar_aug_matrix)
    depth = depth[0]  # [N, 1, H, W]

    out_dir = args.out_dir or "/tmp/dense_lss_depth_check_{}".format(args.sample)
    os.makedirs(out_dir, exist_ok=True)

    mean = np.array(IMG_MEAN).reshape(3, 1, 1)
    std = np.array(IMG_STD).reshape(3, 1, 1)

    n_hit_total = 0
    for c in range(num_cam):
        # de-normalize image, then min-max scale for robust display (the exact
        # absolute brightness is irrelevant -- only point alignment matters)
        img = imgs[c].cpu().numpy() * std + mean
        img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        dmap = depth[c, 0].cpu().numpy()
        ys, xs = np.nonzero(dmap)          # row, col of scattered range values
        vals = dmap[ys, xs]
        n_hit_total += len(xs)

        fig, ax = plt.subplots(figsize=(img_w / 100.0, img_h / 100.0), dpi=100)
        ax.imshow(img)
        if len(xs) > 0:
            sc = ax.scatter(xs, ys, c=vals, s=2, cmap="jet", alpha=0.7,
                            vmin=1.0, vmax=60.0)
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="range (m)")
        ax.set_title("cam{} - {} projected pts".format(c, len(xs)))
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.axis("off")
        fig.savefig(os.path.join(out_dir, "cam{}.png".format(c)),
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print("Wrote {} overlays to {} ({} projected points total).".format(
        num_cam, out_dir, n_hit_total))
    print("VERIFY: projected points must lie ON surfaces (cars/road/walls). "
          "A global offset or transpose => calibration wiring is wrong.")


if __name__ == "__main__":
    main()
