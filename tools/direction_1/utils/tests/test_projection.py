"""Unit tests for ``project_3d_boxes_to_views``.

Runs under either ``python -m pytest`` or ``python -m unittest`` (each test is a
plain function with assertions; nothing pytest-specific is used).

The synthetic 6-camera rig is built by ``_fixtures.make_synthetic_lidar2img``.
A LiDAR point at ``(10, 0, 0)`` projects through view 0 to pixel
``(800, 450)`` at depth 10 -- this is asserted in the first test below to
guard the fixture itself.
"""
import numpy as np

from tools.direction_1.utils.projection import project_3d_boxes_to_views
from tools.direction_1.utils.tests._fixtures import make_synthetic_lidar2img


IMG_SHAPE = (900, 1600)


def _l2i():
    return make_synthetic_lidar2img(num_cams=6, img_shape=IMG_SHAPE)


def _project_point(pt_xyz, lidar2img_v):
    """Helper: project a single 3D point through one ``lidar2img`` matrix."""
    p_h = np.array([pt_xyz[0], pt_xyz[1], pt_xyz[2], 1.0])
    out = p_h @ lidar2img_v.T
    return out[:2] / out[2], out[2]


def test_synthetic_frontal_box():
    """Box at LiDAR (10, 0, 0): visible only in front camera (view 0)."""
    gt = np.array([[10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])
    result = project_3d_boxes_to_views(gt, labels, _l2i(), IMG_SHAPE)

    assert len(result) == 6
    assert len(result[0]['box_indices']) == 1
    assert result[0]['box_indices'][0] == 0
    assert abs(result[0]['depths'][0] - 10.0) < 1e-3
    # Center of front camera image (1600/2, 900/2)
    assert np.allclose(result[0]['centers_2d'][0], [800.0, 450.0], atol=0.5)

    for v in range(1, 6):
        assert result[v]['box_indices'] == [], (
            f'View {v} should be empty, got {result[v]["box_indices"]}')


def test_box_behind_camera_excluded_from_front():
    """A box at (-10, 0, 0) is behind the front camera and excluded."""
    gt = np.array([[-10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])
    result = project_3d_boxes_to_views(gt, labels, _l2i(), IMG_SHAPE)
    assert result[0]['box_indices'] == []


def test_box_behind_camera_appears_in_rear():
    """The same (-10, 0, 0) box appears in the rear camera (view 3)."""
    gt = np.array([[-10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])
    result = project_3d_boxes_to_views(gt, labels, _l2i(), IMG_SHAPE)
    assert len(result[3]['box_indices']) == 1
    assert abs(result[3]['depths'][0] - 10.0) < 1e-3


def test_box_at_fov_edge_visibility():
    """Box positioned so visibility ~0.5 in the front camera.

    With ``visibility_threshold=0.4`` the box is included; with 0.6 it is not.
    """
    gt = np.array([[10.0, -6.1, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])

    r_low = project_3d_boxes_to_views(
        gt, labels, _l2i(), IMG_SHAPE, visibility_threshold=0.4)
    r_high = project_3d_boxes_to_views(
        gt, labels, _l2i(), IMG_SHAPE, visibility_threshold=0.6)

    assert len(r_low[0]['box_indices']) == 1
    assert 0.4 <= r_low[0]['visibilities'][0] < 0.6
    assert r_high[0]['box_indices'] == []


def test_multi_view_box():
    """Box on the FOV boundary between view 0 and view 5.

    For the synthetic rig, view 5 has azimuth 300 deg (= -60 deg), so the
    geometric mid-line between views 0 and 5 is azimuth -30 deg, i.e. LiDAR
    ``(10, -5.77, 0)``. Empirically the box's right side overflows view 0 (so
    ``boxes_2d[0, 2] == W``) and its left side overflows view 5 (so
    ``boxes_2d[0, 0] == 0``). Both views report visibility just above 0.5.
    """
    gt = np.array([[10.0, -5.77, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])
    result = project_3d_boxes_to_views(gt, labels, _l2i(), IMG_SHAPE)

    assert len(result[0]['box_indices']) == 1
    assert len(result[5]['box_indices']) == 1
    assert result[0]['boxes_2d'][0, 2] == float(IMG_SHAPE[1])  # clipped right
    assert result[5]['boxes_2d'][0, 0] == 0.0                  # clipped left
    # Different unclipped extents in the two views.
    assert not np.allclose(result[0]['boxes_2d'][0], result[5]['boxes_2d'][0])
    # Other views must be empty.
    for v in (1, 2, 3, 4):
        assert result[v]['box_indices'] == []


def test_empty_input():
    """Empty input -> V dicts with empty arrays of the right dtypes."""
    gt = np.zeros((0, 7))
    labels = np.zeros((0,), dtype=int)
    result = project_3d_boxes_to_views(gt, labels, _l2i(), IMG_SHAPE)

    assert len(result) == 6
    for vd in result:
        assert vd['box_indices'] == []
        assert vd['boxes_2d'].shape == (0, 4)
        assert vd['centers_2d'].shape == (0, 2)
        assert vd['depths'].shape == (0,)
        assert vd['labels'].shape == (0,)
        assert vd['labels'].dtype == int or np.issubdtype(
            vd['labels'].dtype, np.integer)
        assert vd['visibilities'].shape == (0,)


def test_box_far_beyond_range():
    """A box at (200, 0, 0) is far away but still projects and is included."""
    gt = np.array([[200.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])
    result = project_3d_boxes_to_views(gt, labels, _l2i(), IMG_SHAPE)
    assert len(result[0]['box_indices']) == 1
    assert abs(result[0]['depths'][0] - 200.0) < 1e-2
    # Box is small and fully on the image -> visibility 1.0
    assert abs(result[0]['visibilities'][0] - 1.0) < 1e-6


def test_center_vs_corner_centroid():
    """Center 2D must be the projection of the 3D center, NOT the centroid of
    the 8 projected corners. Use an asymmetric box (yaw=pi/3, narrow in x but
    long in y, thin in z) and verify the two values differ in pixel space.
    """
    yaw = np.pi / 3
    gt = np.array([[8.0, 1.0, 0.0, 1.0, 5.0, 1.5, yaw]])
    labels = np.array([0])
    l2i = _l2i()
    result = project_3d_boxes_to_views(
        gt, labels, l2i, IMG_SHAPE, visibility_threshold=0.0)

    # The asymmetric box should be visible in at least one view; pick the
    # first view that sees it.
    seen = next(v for v, vd in enumerate(result) if len(vd['box_indices']) > 0)
    center_pix = result[seen]['centers_2d'][0]

    expected_center, _ = _project_point(gt[0, :3], l2i[seen])
    assert np.allclose(center_pix, expected_center, atol=1e-6)

    # Build the 8 corners independently, project them, and take the mean.
    cx, cy, cz, w, l, h, _yaw = gt[0]
    sx = np.array([+1, -1, -1, +1, +1, -1, -1, +1])
    sy = np.array([+1, +1, -1, -1, +1, +1, -1, -1])
    sz = np.array([+1, +1, +1, +1, -1, -1, -1, -1])
    local = np.stack([sx * w / 2, sy * l / 2, sz * h / 2], axis=-1)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0],
    ])
    world = local @ R.T + np.array([cx, cy, cz])
    pix = np.zeros((8, 2))
    for k in range(8):
        pix[k], _ = _project_point(world[k], l2i[seen])
    corner_centroid = pix.mean(axis=0)

    # Centroid of projected corners != projection of 3D center.
    assert np.linalg.norm(center_pix - corner_centroid) > 0.5


def test_depth_consistency():
    """Box at (15, 0, 0) -> depth ~ 15 m in view 0 (no LiDAR-camera offset)."""
    gt = np.array([[15.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])
    result = project_3d_boxes_to_views(gt, labels, _l2i(), IMG_SHAPE)
    assert abs(result[0]['depths'][0] - 15.0) < 1e-3


def test_visibility_threshold_filtering():
    """Lower visibility threshold strictly includes more (or equal) boxes."""
    gt = np.array([
        [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],     # fully visible front
        [10.0, -6.1, 0.0, 2.0, 4.0, 1.5, 0.0],    # ~0.5 visibility front
        [10.0, -5.77, 0.0, 2.0, 4.0, 1.5, 0.0],   # ~0.55 in views 0 and 5
        [-10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],    # rear
    ])
    labels = np.array([0, 0, 0, 0])
    l2i = _l2i()
    r_low = project_3d_boxes_to_views(
        gt, labels, l2i, IMG_SHAPE, visibility_threshold=0.3)
    r_high = project_3d_boxes_to_views(
        gt, labels, l2i, IMG_SHAPE, visibility_threshold=0.7)

    total_low = sum(len(vd['box_indices']) for vd in r_low)
    total_high = sum(len(vd['box_indices']) for vd in r_high)
    assert total_low > total_high
    for v in range(6):
        assert len(r_low[v]['box_indices']) >= len(r_high[v]['box_indices'])


def test_returns_clipped_box_unclipped_center():
    """``boxes_2d`` is clipped to image bounds; ``centers_2d`` is not.

    Two scenarios:
    (a) 3D center projects well inside the image but the 2D extent overflows
        the right edge -> ``boxes_2d[..., 2] == W``, center ``u`` is the
        unclipped projection (still inside the image).
    (b) 3D center projects very close to the image edge and the unclipped 2D
        box extends past ``W``. Assert ``centers_2d[0, 0] < W`` and
        ``boxes_2d[0, 2] == W``.
    """
    H, W = IMG_SHAPE
    l2i = _l2i()

    # (a) box overflows right edge but center is well inside.
    gt_a = np.array([[10.0, -5.77, 0.0, 2.0, 4.0, 1.5, 0.0]])
    labels = np.array([0])
    res_a = project_3d_boxes_to_views(gt_a, labels, l2i, IMG_SHAPE)
    assert len(res_a[0]['box_indices']) == 1
    assert res_a[0]['boxes_2d'][0, 2] == float(W)
    assert res_a[0]['centers_2d'][0, 0] < float(W)
    # Cross-check: center matches the direct projection of the 3D center.
    expected_center, _ = _project_point(gt_a[0, :3], l2i[0])
    assert np.allclose(res_a[0]['centers_2d'][0], expected_center, atol=1e-6)

    # (b) box whose center projects very near the right edge (u ~ W - 5),
    #     unclipped extent overflows W, yet the center remains inside.
    # Solving 1252 * (-y) / 10 + 800 = 1595 gives y = -6.351.
    gt_b = np.array([[10.0, -6.351, 0.0, 2.0, 4.0, 1.5, 0.0]])
    res_b = project_3d_boxes_to_views(
        gt_b, labels, l2i, IMG_SHAPE, visibility_threshold=0.4)
    assert len(res_b[0]['box_indices']) == 1
    assert res_b[0]['centers_2d'][0, 0] < float(W)
    assert abs(res_b[0]['centers_2d'][0, 0] - (W - 5.0)) < 1.0
    assert res_b[0]['boxes_2d'][0, 2] == float(W)
