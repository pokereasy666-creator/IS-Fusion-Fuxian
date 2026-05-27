# DONOR_REQUIREMENTS — external LSS view-transformer drop-in spec

Static-analysis spec (LOCATE-AND-REPORT ONLY; no model/config/training edits).
This document characterizes IS-Fusion's existing `bev_pool` op and the dense
image→BEV integration contract, and states exactly what an external LSS
view-transformer (DepthNet + `get_geometry` + `bev_pool`) must satisfy to drop in.

- Base: `claude/4gpu-gradient-accumulation-conf-4wxWs` @ `41d4858` (D1-free).
- Config of record: `configs/isfusion/isfusion_0075voxel.py`.

## 0. Scope / current status

- The CUDA `bev_pool` op exists at `mmdet3d/ops/bev_pool/` (`bev_pool.py`,
  `src/bev_pool.cpp`, `src/bev_pool_cuda.cu`) and is exported via
  `mmdet3d/ops/bev_pool/__init__.py`, **but it is not imported or used anywhere in
  the model** (only self-references inside the op package). It is an available,
  unused primitive.
- The current dense image→BEV branch is **`ISFusionEncoder.img_fv_to_bev`**
  (`mmdet3d/models/middle_encoders/fusion_encoder.py:1047`), a UVTR-style
  **pillar-point-sampling** projector (`F.grid_sample` of image features at
  projected LiDAR-pillar points) — **not** an LSS depth-splat. There is **no
  DepthNet** and **no `get_geometry`** in the repo; a donor must bring both.
- A donor LSS transformer would replace `img_bev_feats` (call it **B_I**) at
  `fusion_encoder.py:1162`, or be merged into B_I before `:1167`.

---

## 1. `bev_pool` call signature the donor MUST use  (Q1)

From `mmdet3d/ops/bev_pool/bev_pool.py` (verbatim):

```python
def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
```

**Arguments**

| arg      | meaning                                   | shape / type                         |
|----------|-------------------------------------------|--------------------------------------|
| `feats`  | per-point (frustum) features to scatter   | `[N, C]`, float32                    |
| `coords` | integer voxel/BEV coordinates per point   | `[N, >=4]` (only cols 0..3 are read) |
| `B`      | batch size                                | python int                           |
| `D`      | depth dim of the pooled grid              | python int                           |
| `H`      | height dim of the pooled grid             | python int                           |
| `W`      | width dim of the pooled grid              | python int                           |

**Return shape: `[B, C, D, H, W]`.** The CUDA forward allocates the raw output as
`[B, D, H, W, C]` (`torch::zeros({b, d, h, w, c})`, `src/bev_pool.cpp:40`); the
wrapper then `.permute(0, 4, 1, 2, 3)` → **`[B, C, D, H, W]`**.

**autograd `Function` signatures** (`bev_pool.py`):

```python
class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        ...
        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )
        ...
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ...
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )
        return x_grad, None, None, None, None, None, None
```

**Extension binding arg order** — `bev_pool_ext.bev_pool_forward` /
`bev_pool_backward` take **`interval_lengths` BEFORE `interval_starts`**, matching
the C++ binding (`src/bev_pool.cpp:22-28`):

```cpp
at::Tensor bev_pool_forward(
  const at::Tensor _x,
  const at::Tensor _geom_feats,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
)
```

> **Caveat (documentation bug, behavior is correct):** the `.cpp`/`.cu` docstrings
> swap the *descriptions* of `interval_lengths` ("starting position...") and
> `interval_starts` ("how many points..."). The **code** is consistent:
> `interval_starts = torch.where(kept)[0]` are offsets; `interval_lengths` are the
> per-interval counts. Trust the code/arg-order, not the comment text.

A donor must call **`from mmdet3d.ops.bev_pool import bev_pool`** and use exactly
`bev_pool(feats, coords, B, D, H, W)`. (`QuickCumsum`, the pure-torch class at
`bev_pool.py:8`, is an unused fallback and is **not** what the wrapper calls.)

---

## 2. `coords` / `ranks` layout + rank formula the donor MUST reproduce  (Q2)

**This is the compatibility-critical part. A mismatch silently corrupts the pooled
BEV (no error is raised).**

**Rank formula** (`bev_pool.py:86-91`, verbatim):

```python
ranks = (
    coords[:, 0] * (W * D * B)
    + coords[:, 1] * (D * B)
    + coords[:, 2] * B
    + coords[:, 3]
)
```

Points are then sorted **ascending** by rank (`indices = ranks.argsort()`), and
`feats`/`coords` are reordered to match. The cumsum trick relies on identical
`(col0,col1,col2,col3)` tuples landing contiguously.

**Column → output-axis mapping.** The CUDA kernel writes each pooled point into the
`[B, D, H, W, C]` output at this offset (`src/bev_pool_cuda.cu:34-36`, verbatim):

```c
float* cur_out = out + cur_geom_feats[3] * d * h * w * c +
    cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c +
    cur_geom_feats[1] * c + cur_c;
```

For a C-contiguous `[B, D, H, W, C]` tensor the strides are
`(D·H·W·C, H·W·C, W·C, C, 1)`, so the columns map **unambiguously**:

| `coords` column | multiplies stride | → output axis | valid range |
|-----------------|-------------------|---------------|-------------|
| `coords[:, 0]`  | `W·C`             | **H**         | `[0, H)`    |
| `coords[:, 1]`  | `C`               | **W**         | `[0, W)`    |
| `coords[:, 2]`  | `H·W·C`           | **D** (depth) | `[0, D)`    |
| `coords[:, 3]`  | `D·H·W·C`         | **B** (batch) | `[0, B)`    |

So the donor's `coords` must be laid out as **`[H-index, W-index, D-index, batch]`**.
The rank is a bijection over the ranges above; any value outside `[0,H)/[0,W)/[0,D)/
[0,B)` aliases another cell and silently corrupts the result (no bounds check).

**dtype:** `feats` is float32; `coords` is cast to int32 internally
(`geom_feats = geom_feats.int()`, `bev_pool.py:46`). Donor `coords` should be an
integer tensor (long/int both fine; the `.int()` cast happens inside the op).
`feats.shape[0] == coords.shape[0] == N` is asserted.

---

## 3. Calibration inputs available + frame conventions  (Q3)

**Available meta keys** — `Collect3DV2.meta_keys`
(`configs/isfusion/isfusion_0075voxel.py:290-293`):

```python
meta_keys=[
    'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
    'camera2lidar', 'lidar2img', 'img_aug_matrix', 'lidar_aug_matrix',
]
```

These reach the encoder via `**kwargs` and are per-camera 4×4 stacks
(`lidar2img`, `camera2lidar`, `lidar2camera`, `camera2ego`, `camera_intrinsics`,
`img_aug_matrix` → `[N_cam, 4, 4]`; `lidar_aug_matrix` → `[4, 4]` per sample, batched).
This is the standard BEVFusion meta set.

**How the existing code consumes them** — `ISFusionEncoder.img_point_sampling`
(`fusion_encoder.py:966-1013`) projects **augmented-LiDAR → image pixel** in this
order:

1. **Undo LiDAR aug** (`:984-987`): `cur_coords -= lidar_aug_matrix[:3,3]`; then
   `inverse(lidar_aug_matrix[:3,:3]) @ cur_coords` → raw-LiDAR.
2. **`lidar2img`** (`:990-991`): `lidar2img[:, :3,:3] @ pts + lidar2img[:, :3,3]`
   → camera-frame ray (raw-LiDAR → image, **pre-img-aug**).
3. **Perspective divide** (`:1002-1003`): clamp z, `xy /= z` → pixel coords.
4. **`img_aug_matrix`** (`:1006-1007`): `img_aug_matrix[:, :3,:3] @ px +
   img_aug_matrix[:, :3,3]` → network-input pixel (**post-img-aug**).
5. Normalize by `img_metas[0]['input_shape']` to `[-1, 1]`.

**Frame summary for a donor `get_geometry` (which runs this in reverse,
frustum→BEV):**

- `lidar2img` : raw-LiDAR → image pixel (pre-img-aug).
- `img_aug_matrix` : pre-aug pixel → network-input pixel.
- `lidar_aug_matrix` : raw-LiDAR → **augmented-LiDAR** (the frame `pc_range`/the BEV
  grid live in). The donor's pooled BEV must be expressed in **augmented-LiDAR**.
- `camera_intrinsics` / `camera2lidar` : pixel↔camera and camera→raw-LiDAR.

**Sufficient set** to map a frustum point `(u, v, depth_bin)` → BEV `(x, y)` bin in
the §2 convention (BEVFusion-style `get_geometry`):

```
camera_intrinsics⁻¹ (pixel→cam ray) → ×depth → camera2lidar (cam→raw-LiDAR)
→ lidar_aug_matrix (raw→augmented-LiDAR) → voxelize with pc_range/cell (§6)
```
plus `img_aug_matrix⁻¹` applied to the frustum grid first (undo image aug).
Equivalent alternative: invert `lidar2img` instead of `camera_intrinsics`+`camera2lidar`.
All required matrices are present in the meta set above.

---

## 4. DepthNet input + output contract  (Q4)

**DepthNet input feature:** `img_mlvl_feats[1]` = **256 channels @ 24×66**, packed
**`[B*N, 256, 24, 66]`** where `B*N = batch_size · num_cam` (num_cam = 6).

- Images enter the backbone flattened over cameras: `B, N, C, H, W = img.size();
  img = img.view(B*N, C, H, W)` (`mmdet3d/models/detectors/isfusion.py:69-70`), so
  every per-level feature map is `[B*N, C, h, w]`.
- Neck `GeneralizedLSSFPN` (`out_channels=256`) returns **2** levels
  (`used_backbone_levels = len(laterals) - 1 = 2`,
  `mmdet3d/models/necks/generalized_lss.py:90,102`). With input `img_scale=(384,1056)`:
  level 0 = stride-8 (48×132), **level 1 = stride-16 = 24×66** (384/16, 1056/16).
- ⚠️ The inline comment `feat: [24,256,32,88]` at `fusion_encoder.py:1033` is **stale**
  (from a 512×1408 input); the current config yields **24×66**. This matches the
  prior dense-locate report (no discrepancy).

A donor DepthNet must therefore be configured with `in_channels = 256` and operate
on the **24×66** grid with **`[B*N, …]`** packing.

**Output contract:** the dense BEV must be **`[bs, 256, 180, 180]`**. Integration
(`fusion_encoder.py:1162-1167`, verbatim):

```python
img_bev_feats = self.img_fv_to_bev([img_mlvl_feats[1]], bs, **kwargs)

kwargs.update(dict(img_bev_feats=img_bev_feats))
kwargs.update(dict(lidar_feats=lidar_feats))

bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats], dim=1))
```

- Replace **B_I** = `img_bev_feats` at `:1162`, **or** merge into B_I before `:1167`.
- `self.conv_fusion` has `in_channels = embed_dims*3 = 768` (`fusion_encoder.py:862`,
  `embed_dims=256`). The cat is `img_bev_feats(256) + lidar_feats(512) = 768`, so any
  merge **must keep the image side at exactly 256 channels** (do not widen the cat).
- The current B_I is built as `torch.zeros([bs, embed_dims, bev_size, bev_size])` =
  `[bs, 256, 180, 180]` (`fusion_encoder.py:1065`), confirming the required shape.

---

## 5. Required `bev_pool` dims for this grid

To produce `[bs, 256, 180, 180]`, call `bev_pool(feats, coords, B, D, H, W)` with:

- `B = bs`, `C = 256` (from `feats`' channel dim), `H = 180`, `W = 180`.
- `D = 1` (collapse Z to a single bin) → output `[bs, 256, 1, 180, 180]`, then
  `.squeeze(2)` → `[bs, 256, 180, 180]`. (If the donor keeps `D>1` Z-bins it must
  reduce/flatten D back to 1·256 channels before §4's cat.)

---

## 6. BEV metric bounds the donor MUST be configured to  (LSS terms)

From `configs/isfusion/isfusion_0075voxel.py`:
`point_cloud_range = [-54, -54, -5, 54, 54, 3]`, `voxel_size = [0.075, 0.075, 0.2]`,
`out_size_factor = 8` ⇒ `voxel_shape = 108/0.075 = 1440`, `bev_size = 1440//8 = 180`.
BEV cell = `0.075 · 8 = 0.6 m`.

| LSS bound | value `[lower, upper, cell]` | bins | notes                                  |
|-----------|------------------------------|------|----------------------------------------|
| `xbound`  | `[-54.0, 54.0, 0.6]`         | 180  | X extent of `pc_range`                 |
| `ybound`  | `[-54.0, 54.0, 0.6]`         | 180  | Y extent of `pc_range`                 |
| `zbound`  | `[-5.0, 3.0, 8.0]`           | 1    | collapse Z → single bin (2D BEV)       |
| `dbound`  | donor's choice (e.g. `[1.0, 60.0, 0.5]`) | —  | depth bins along the camera ray; not fixed by IS-Fusion, but every splatted `(x,y)` must land inside `xbound`/`ybound` |

**Axis-orientation flag (must verify):** §2 fixes `coords` col0→H, col1→W
mechanically, but *which metric axis (X vs Y) the donor puts in col0 vs col1* must
match the existing LiDAR-BEV layout so the fused map is not transposed/flipped. The
LiDAR BEV comes from `SparseEncoder` with `sparse_shape=[41, 1440, 1440]` = `[z, y, x]`
(config `pts_middle_encoder`), i.e. **BEV H ← Y, W ← X**. The donor should map
**col0 = Y-bin, col1 = X-bin** — but the integrator must confirm this against the
actual `lidar_feats` orientation, since a swap silently flips the camera BEV
relative to LiDAR.

---

## 7. Drop-in compatibility checklist

A donor `LSSViewTransformer` is **drop-in compatible IFF**:

1. **bev_pool call:** it calls `from mmdet3d.ops.bev_pool import bev_pool` as
   `bev_pool(feats, coords, B, D, H, W)` with `feats:[N,256]` float32,
   `coords:[N,>=4]` int — and consumes the return as **`[B, C, D, H, W]`**
   (channels at dim 1, after the internal `.permute(0,4,1,2,3)`).
2. **coord convention (§2):** `coords` columns are `[H-idx, W-idx, D-idx, batch]`
   with ranges `[0,180)`, `[0,180)`, `[0,D)`, `[0,bs)`, and it reproduces the rank
   formula `col0*(W*D*B) + col1*(D*B) + col2*B + col3` implicitly (i.e. it relies on
   `bev_pool` to sort — it must NOT pre-sort with a different rank).
3. **frame (§3):** its `get_geometry` outputs BEV coordinates in the
   **augmented-LiDAR** frame (applies `lidar_aug_matrix`), consistent with `pc_range`.
4. **bounds (§6):** configured to `xbound=[-54,54,0.6]`, `ybound=[-54,54,0.6]`,
   `zbound=[-5,3,8.0]` (Z collapsed to 1 bin).
5. **output (§4/§5):** emits **`[bs, 256, 180, 180]`** (call `bev_pool` with
   `B=bs, H=W=180, D=1`, then `squeeze` D).
6. **axis orientation (§6):** its X/Y→col0/col1 (H/W) assignment matches the LiDAR
   BEV (`H←Y, W←X`), verified against `lidar_feats`.
7. **input (§4):** its DepthNet ingests `[B*N, 256, 24, 66]`.

If any of 1–7 fails, integration is wrong (often silently): #1/#5 raise shape errors
at the cat/conv_fusion; **#2/#3/#6 produce a corrupted or geometrically wrong BEV
with no exception.**

---

## 8. Adaptations always needed regardless of donor

Independent of which donor is chosen, the integrator must:

- **Re-point DepthNet** to `img_mlvl_feats[1]` = **256ch @ 24×66**, `[B*N,…]`
  packing (set `in_channels=256`, feature grid 24×66; do not assume a donor's native
  resolution/channels).
- **Wire calibration** from IS-Fusion's `**kwargs` meta into the donor's
  `get_geometry`: primary `lidar2img` + `img_aug_matrix`, plus
  `camera_intrinsics` / `camera2lidar` / `lidar_aug_matrix` (rename/transpose to the
  donor's expected argument names; many donors expect `rots/trans/intrins/post_rots/
  post_trans/bda` — map these from the 4×4 stacks above).
- **Project output to `[bs, 256, 180, 180]`** (call `bev_pool` with `B=bs, D=1,
  H=W=180`, squeeze D; add a 1×1 conv if the donor's channel count ≠ 256).
- **Set BEV bounds** to match `pc_range`: `xbound/ybound = [-54,54,0.6]`,
  `zbound = [-5,3,8.0]` (collapse Z).
- **Verify X/Y → H/W orientation** against the LiDAR BEV (`H←Y, W←X`) before trusting
  the fused result.
- **Decide replace-vs-merge** at `fusion_encoder.py:1162`: either substitute B_I, or
  combine with B_I before `:1167` while keeping the image side at 256ch (conv_fusion
  expects 768 = 256 img + 512 lidar).
