"""Per-view 3D-to-2D GT projection utility.

Pure-NumPy helper that takes 3D ground-truth boxes in the LiDAR frame plus a
batch of ``lidar2img`` matrices (one per camera view) and returns, for every
view, the per-box clipped 2D bounding box, unclipped 2D center, depth of the
3D center in the camera frame, and a visibility score.

Existing related code in the repo (do NOT use these directly; APIs and data
structures differ, but they are useful references):
- mmdet3d/core/bbox/box_np_ops.py:346 ``points_cam2img(points_3d, proj_mat,
  with_depth)``: NumPy single-matrix projection. Confirms the convention
  ``pixel = projected[..., :2] / projected[..., 2]`` and depth in
  ``projected[..., 2]``.
- mmdet3d/core/bbox/structures/utils.py:116 ``points_cam2img``: PyTorch twin.
- mmdet3d/core/bbox/box_np_ops.py:384 ``box3d_to_bbox(box3d, P2)``: collapses
  3D box to a single-view 2D bbox; no clipping or visibility.
- mmdet3d/core/bbox/box_np_ops.py:49 ``corners_nd``: 8-corner generator with a
  different layout.
- mmdet3d/core/bbox/structures/lidar_box3d.py:50 ``LiDARInstance3DBoxes.corners``
  property: corners with z-axis yaw rotation, but uses BOTTOM-CENTER origin and
  ``(l, w, h)`` dim ordering -- both differ from this utility's conventions.
- mmdet3d/core/visualizer/image_vis.py:9 ``project_pts_on_img``: confirms the
  same ``pts_4d @ lidar2img.T`` then divide-by-z pattern used here.
- tools/data_converter/nuscenes_converter.py:540-615: full nuScenes-style
  per-camera reprojection using a convex-hull intersection with the image
  canvas. We use the simpler axis-aligned clipping the spec calls for.

Coordinate conventions (LiDAR frame):
    x = forward
    y = left
    z = up
    yaw = rotation around +z, positive = counter-clockwise viewed from above.

Box parameterization: ``(x, y, z, w, l, h, yaw)`` where ``(x, y, z)`` is the
box CENTER (not the bottom-center). Local axes assignment for the corner
template: ``+/- w/2`` along the box's local x, ``+/- l/2`` along local y,
``+/- h/2`` along local z. With ``yaw=0`` the local axes coincide with the
LiDAR axes, so width spans forward, length spans left, height spans up.

NOTE on nuScenes / mmdetection3d compatibility: ``LiDARInstance3DBoxes`` uses
the bottom-center as the origin and ``(x_size, y_size, z_size)`` =
``(length, width, height)`` ordering. To use this utility on those boxes, the
caller must (a) shift z by ``+ h/2`` so it becomes the box center, and (b) swap
the first two size columns from ``(l, w, h)`` to ``(w, l, h)`` so they match
this function's local-axis assignment.
"""
from typing import Dict, List, Tuple
import warnings

import numpy as np


def _compute_corners_3d(gt_boxes_3d: np.ndarray) -> np.ndarray:
    """Compute the 8 corners of each 3D box in the LiDAR frame.

    Args:
        gt_boxes_3d: ``(N, 7)`` array, columns are
            ``(x, y, z, w, l, h, yaw)``. ``(x, y, z)`` is the box center.

    Returns:
        ``(N, 8, 3)`` array of corner coordinates in the LiDAR frame.
    """
    N = gt_boxes_3d.shape[0]
    if N == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)

    centers = gt_boxes_3d[:, :3].astype(np.float64)
    w = gt_boxes_3d[:, 3].astype(np.float64)
    l = gt_boxes_3d[:, 4].astype(np.float64)
    h = gt_boxes_3d[:, 5].astype(np.float64)
    yaw = gt_boxes_3d[:, 6].astype(np.float64)

    # Local corners relative to box center: 8 corners, columns are (x, y, z)
    # in the box's own frame. Order matches the spec's listing.
    sx = np.array([+1, -1, -1, +1, +1, -1, -1, +1], dtype=np.float64)
    sy = np.array([+1, +1, -1, -1, +1, +1, -1, -1], dtype=np.float64)
    sz = np.array([+1, +1, +1, +1, -1, -1, -1, -1], dtype=np.float64)

    local = np.stack(
        [
            sx[None, :] * (w[:, None] / 2.0),
            sy[None, :] * (l[:, None] / 2.0),
            sz[None, :] * (h[:, None] / 2.0),
        ],
        axis=-1,
    )  # (N, 8, 3)

    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    zero = np.zeros_like(yaw)
    one = np.ones_like(yaw)
    R = np.stack(
        [
            np.stack([cos_y, -sin_y, zero], axis=-1),
            np.stack([sin_y,  cos_y, zero], axis=-1),
            np.stack([zero,   zero,  one ], axis=-1),
        ],
        axis=-2,
    )  # (N, 3, 3)

    rotated = np.einsum('nij,nkj->nki', R, local)  # (N, 8, 3)
    return rotated + centers[:, None, :]


def _empty_view_dict() -> Dict:
    """Empty result dict with the per-key dtypes the spec mandates."""
    return {
        'box_indices': [],
        'boxes_2d': np.zeros((0, 4), dtype=np.float64),
        'centers_2d': np.zeros((0, 2), dtype=np.float64),
        'depths': np.zeros((0,), dtype=np.float64),
        'labels': np.zeros((0,), dtype=int),
        'visibilities': np.zeros((0,), dtype=np.float64),
    }


def _project_to_view(corners_3d: np.ndarray,
                     centers_3d: np.ndarray,
                     gt_labels: np.ndarray,
                     lidar2img_v: np.ndarray,
                     img_shape: Tuple[int, int],
                     visibility_threshold: float,
                     depth_min: float) -> Dict:
    """Project all GT boxes into a single camera view.

    Returns the dict described in :func:`project_3d_boxes_to_views`.
    """
    N = corners_3d.shape[0]
    if N == 0:
        return _empty_view_dict()

    H, W = img_shape

    # --- Project box centers (used for depth and as the unclipped 2D center).
    centers_h = np.concatenate(
        [centers_3d, np.ones((N, 1), dtype=centers_3d.dtype)], axis=-1)
    proj_centers = centers_h @ lidar2img_v.T  # (N, 4)
    center_depths = proj_centers[:, 2]
    safe_center_depths = np.where(center_depths > 0, center_depths, 1.0)
    centers_2d = proj_centers[:, :2] / safe_center_depths[:, None]  # (N, 2)

    # --- Project the 8 corners.
    corners_h = np.concatenate(
        [corners_3d, np.ones((N, 8, 1), dtype=corners_3d.dtype)], axis=-1)
    proj_corners = corners_h @ lidar2img_v.T  # (N, 8, 4)
    depths_corners = proj_corners[:, :, 2]
    valid_corners = depths_corners > 0
    safe_depths = np.where(valid_corners, depths_corners, 1.0)
    pixel_xy = proj_corners[:, :, :2] / safe_depths[..., None]  # (N, 8, 2)

    # Mask invalid corners so they don't affect the unclipped bounding box.
    x_coords = np.where(valid_corners, pixel_xy[:, :, 0], np.nan)
    y_coords = np.where(valid_corners, pixel_xy[:, :, 1], np.nan)

    # All-NaN rows (every corner had depth <= 0) are filtered out below via the
    # ``~np.isnan(x1)`` mask, but ``np.nanmin`` still emits a RuntimeWarning
    # for those rows -- silence it explicitly. ``np.errstate`` only governs FPE
    # warnings, not the all-NaN one, so a ``warnings.catch_warnings`` block is
    # needed.
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.filterwarnings(
            'ignore', message='All-NaN slice encountered',
            category=RuntimeWarning)
        x1 = np.nanmin(x_coords, axis=1)
        y1 = np.nanmin(y_coords, axis=1)
        x2 = np.nanmax(x_coords, axis=1)
        y2 = np.nanmax(y_coords, axis=1)

    # Clip to image bounds for the returned 2D box.
    x1_c = np.clip(x1, 0.0, float(W))
    y1_c = np.clip(y1, 0.0, float(H))
    x2_c = np.clip(x2, 0.0, float(W))
    y2_c = np.clip(y2, 0.0, float(H))

    unclipped_area = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    clipped_area = np.maximum(x2_c - x1_c, 0.0) * np.maximum(y2_c - y1_c, 0.0)

    visibilities = np.where(
        unclipped_area > 1e-6,
        clipped_area / np.where(unclipped_area > 1e-6, unclipped_area, 1.0),
        0.0,
    )

    keep = (
        (center_depths >= depth_min)
        & (visibilities >= visibility_threshold)
        & ~np.isnan(x1)
    )

    if not np.any(keep):
        return _empty_view_dict()

    keep_idx = np.flatnonzero(keep)
    boxes_2d = np.stack(
        [x1_c[keep_idx], y1_c[keep_idx], x2_c[keep_idx], y2_c[keep_idx]],
        axis=-1,
    )
    return {
        'box_indices': keep_idx.tolist(),
        'boxes_2d': boxes_2d.astype(np.float64),
        'centers_2d': centers_2d[keep_idx].astype(np.float64),
        'depths': center_depths[keep_idx].astype(np.float64),
        'labels': gt_labels[keep_idx].astype(int),
        'visibilities': visibilities[keep_idx].astype(np.float64),
    }


def project_3d_boxes_to_views(
        gt_boxes_3d: np.ndarray,
        gt_labels: np.ndarray,
        lidar2img: np.ndarray,
        img_shape: Tuple[int, int],
        visibility_threshold: float = 0.5,
        depth_min: float = 0.5,
) -> List[Dict]:
    """Project 3D ground-truth boxes from LiDAR frame into each camera view.

    Args:
        gt_boxes_3d: Shape ``(N, 7)``. Each row is ``(x, y, z, w, l, h, yaw)``
            in the LiDAR frame. ``(x, y, z)`` is the box CENTER (not the
            bottom-center). ``(w, l, h)`` is width/length/height; with
            ``yaw=0`` the box's local x-axis spans ``+/- w/2`` along LiDAR x
            (forward), local y spans ``+/- l/2`` along LiDAR y (left), and
            local z spans ``+/- h/2`` along LiDAR z (up). ``yaw`` is rotation
            around the LiDAR +z axis, positive = counter-clockwise from above.

            NOTE: nuScenes' ``LiDARInstance3DBoxes`` stores boxes with a
            bottom-center reference and a ``(length, width, height)`` size
            ordering, which is *different* from this function's convention.
            Callers using that class must pre-shift z by ``+ h/2`` and swap
            the first two size columns before calling this utility.
        gt_labels: Shape ``(N,)``. Integer class indices.
        lidar2img: Shape ``(V, 4, 4)``. One projection matrix per camera view
            (``V`` is typically 6 for nuScenes). Each matrix maps homogeneous
            LiDAR coordinates ``[x, y, z, 1]`` to homogeneous image coordinates
            ``[u*d, v*d, d, 1]``, where pixel coords are
            ``projected[..., :2] / projected[..., 2:3]`` and depth is
            ``projected[..., 2]``. This matches both nuScenes / MMDetection3D
            conventions and the ``project_pts_on_img`` helper in this repo.
        img_shape: ``(H, W)`` of the image (rows, columns). nuScenes raw
            resolution is ``(900, 1600)``.
        visibility_threshold: Minimum fraction of the unclipped 2D box that
            must remain after clipping to image bounds for the box to be
            included in a view. Default 0.5.
        depth_min: Minimum positive depth (meters) for the box CENTER (not
            corners) to be included. Default 0.5.

    Returns:
        List of length ``V``. Each entry is a dict with keys:

        - ``box_indices``: ``List[int]`` of the input rows visible in the view.
        - ``boxes_2d``: ``(M, 4)`` array of ``[x1, y1, x2, y2]`` in pixels,
          CLIPPED to ``[0, W] x [0, H]``.
        - ``centers_2d``: ``(M, 2)`` array of ``[u, v]`` in pixels. NOT
          clipped (a center may legitimately fall outside the image).
        - ``depths``: ``(M,)`` array of camera-frame depths (meters) of the
          box centers.
        - ``labels``: ``(M,)`` integer class indices.
        - ``visibilities``: ``(M,)`` array of ``clipped_area / unclipped_area``
          in ``[0, 1]``.

        Views with no visible boxes return empty arrays of the appropriate
        dtypes (``box_indices=[]``, ``boxes_2d=np.zeros((0,4))``,
        ``centers_2d=np.zeros((0,2))``, ``depths=np.zeros((0,))``,
        ``labels=np.zeros((0,), dtype=int)``,
        ``visibilities=np.zeros((0,))``).
    """
    gt_boxes_3d = np.asarray(gt_boxes_3d, dtype=np.float64)
    gt_labels = np.asarray(gt_labels)
    lidar2img = np.asarray(lidar2img, dtype=np.float64)
    assert gt_boxes_3d.ndim == 2 and gt_boxes_3d.shape[1] == 7, (
        f'gt_boxes_3d must have shape (N, 7), got {gt_boxes_3d.shape}')
    assert gt_labels.shape == (gt_boxes_3d.shape[0],), (
        f'gt_labels must have shape (N,), got {gt_labels.shape}')
    assert lidar2img.ndim == 3 and lidar2img.shape[1:] == (4, 4), (
        f'lidar2img must have shape (V, 4, 4), got {lidar2img.shape}')

    V = lidar2img.shape[0]
    N = gt_boxes_3d.shape[0]

    if N == 0:
        return [_empty_view_dict() for _ in range(V)]

    corners_3d = _compute_corners_3d(gt_boxes_3d)
    centers_3d = gt_boxes_3d[:, :3]

    return [
        _project_to_view(
            corners_3d=corners_3d,
            centers_3d=centers_3d,
            gt_labels=gt_labels,
            lidar2img_v=lidar2img[v],
            img_shape=img_shape,
            visibility_threshold=visibility_threshold,
            depth_min=depth_min,
        )
        for v in range(V)
    ]


if __name__ == '__main__':
    # Run with: python -m tools.direction_1.utils.projection
    from tools.direction_1.utils.tests._fixtures import make_synthetic_lidar2img

    gt_boxes = np.array([
        [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],     # car ahead
        [5.0, 8.0, 0.0, 0.7, 0.7, 1.7, np.pi / 4],  # pedestrian on left
        [-8.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],     # car behind
    ])
    gt_labels = np.array([0, 8, 0])  # car, pedestrian, car
    lidar2img = make_synthetic_lidar2img()
    img_shape = (900, 1600)

    result = project_3d_boxes_to_views(
        gt_boxes, gt_labels, lidar2img, img_shape)
    for v, view_data in enumerate(result):
        print(f"View {v}: {len(view_data['box_indices'])} visible boxes")
        for i, box_idx in enumerate(view_data['box_indices']):
            print(
                f"  box {box_idx}: 2D box={view_data['boxes_2d'][i]}, "
                f"center={view_data['centers_2d'][i]}, "
                f"depth={view_data['depths'][i]:.2f}m, "
                f"visibility={view_data['visibilities'][i]:.2f}")
