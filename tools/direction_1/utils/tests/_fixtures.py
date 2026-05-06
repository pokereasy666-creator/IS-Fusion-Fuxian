"""Synthetic camera-rig fixtures for the projection utility.

Used by both ``test_projection.py`` and the ``__main__`` demo block in
``projection.py``. Kept out of ``projection.py`` itself so the utility module
contains only projection logic.

Coordinate conventions (LiDAR frame):
    x = forward, y = left, z = up; yaw is rotation around +z, CCW from above.

Camera convention used here (matches OpenCV / nuScenes camera frame):
    +x = right (in image), +y = down, +z = forward (into the scene).

The synthetic rig places ``num_cams`` pinhole cameras at the LiDAR origin,
spaced uniformly in azimuth, each looking radially outward. All cameras share
a single intrinsic matrix loosely modelled on nuScenes' front camera
(focal ~1252, principal point at the image center, raw 1600x900).

This is enough to exercise multi-view projection logic; it is NOT pixel-perfect
to a real nuScenes calibration.
"""
import numpy as np


def make_synthetic_lidar2img(num_cams: int = 6,
                             img_shape: tuple = (900, 1600),
                             focal: float = 1252.0) -> np.ndarray:
    """Build a (V, 4, 4) stack of ``lidar2img`` projection matrices.

    Each matrix maps homogeneous LiDAR coordinates ``[x, y, z, 1]`` to
    homogeneous image coordinates ``[u*d, v*d, d, 1]`` where ``(u, v)`` is the
    pixel location and ``d`` is the depth in the camera frame. Pixel
    coordinates are recovered as ``projected[..., :2] / projected[..., 2:3]``.

    Args:
        num_cams: Number of cameras in the rig. Default 6 (nuScenes-like).
        img_shape: ``(H, W)`` of every image. Default ``(900, 1600)``.
        focal: Shared focal length in pixels. Default 1252.

    Returns:
        ``np.ndarray`` of shape ``(num_cams, 4, 4)``. View ``v`` has azimuth
        ``2*pi*v / num_cams`` (so view 0 looks at +x_lidar, view 3 at -x_lidar
        for a 6-camera rig).
    """
    H, W = img_shape
    cx, cy = W / 2.0, H / 2.0

    K_h = np.eye(4, dtype=np.float64)
    K_h[0, 0] = focal
    K_h[1, 1] = focal
    K_h[0, 2] = cx
    K_h[1, 2] = cy

    matrices = np.zeros((num_cams, 4, 4), dtype=np.float64)
    for v in range(num_cams):
        theta = 2.0 * np.pi * v / num_cams

        # Camera basis vectors expressed in the LiDAR frame:
        #   forward (cam +z) = [cos t, sin t, 0]   (radial outward)
        #   right   (cam +x) = [sin t, -cos t, 0]  (right-of-forward in BEV)
        #   down    (cam +y) = [0, 0, -1]          (LiDAR +z is up)
        # R_lidar_to_cam has those vectors as ROWS.
        R = np.array([
            [np.sin(theta), -np.cos(theta), 0.0],
            [0.0,             0.0,         -1.0],
            [np.cos(theta),  np.sin(theta), 0.0],
        ], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R  # zero translation: cameras at LiDAR origin

        matrices[v] = K_h @ T

    return matrices
