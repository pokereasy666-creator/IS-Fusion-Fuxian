# Dense LSS Image-BEV Branch — Prerequisites (Locate-and-Report)

**Scope:** locate-and-report only. No model/config/training edits were made.
Static analysis (server offline). This report identifies what already exists and
can be reused versus what must be authored to add a dense, LSS-style
(Lift-Splat-Shoot / BEVDepth) image→BEV branch to IS-Fusion.

**Why:** IS-Fusion currently lifts images into BEV via a *sparse, pillar-driven*
path (`img_fv_to_bev`): image features are grid-sampled only at LiDAR-pillar pixel
locations and scattered into a 180×180 BEV grid. A *dense* branch would instead
lift every image pixel through a predicted depth distribution and splat it across
the whole BEV grid (image-only BEV, no LiDAR dependency for the image stream).

---

## Branch, base, and D1-free evidence

| Item | Value |
|------|-------|
| Base branch | `origin/claude/4gpu-gradient-accumulation-conf-4wxWs` |
| Base SHA (branched-from) | `41d485840f51c0d824dd812a0945d0d51a4e9ad1` (`41d4858`) |
| This branch | `claude/dense-locate` (branched from `41d4858`) |

**D1-free verification** — on the base, `git grep` for the D1 markers returns
**no matches**:

```
git grep -E "_probe_buffer|alpha_beta|image_classifier|loss_alpha|s_img_logits" \
    origin/claude/4gpu-gradient-accumulation-conf-4wxWs
# -> (empty) : CLEAN
```

The base is the clean, D1-free baseline. Proceeding.

---

## Q1 — Does reusable LSS / view-transform / depth code exist?  **VERDICT: NO.**

The repo contains **no** lift-splat view-transformer. The only image→BEV converter
is sparse point-sampling, not a depth-based lift.

- `img_fv_to_bev` — `mmdet3d/models/middle_encoders/fusion_encoder.py:1047-1071`.
  It calls `img_point_sampling` (`:1062`), `F.grid_sample`s image features at LiDAR
  **pillar** pixel locations (`:1033`), allocates a zero BEV tensor (`:1065`), and
  **scatters** the sampled features into the pillars' BEV cells (`:1069`). There is
  **no depth distribution, no frustum, and no depth bins** — geometry comes from
  LiDAR pillars, not from predicted depth.
- `GeneralizedLSSFPN` — `mmdet3d/models/necks/generalized_lss.py:13`. Despite the
  name, this is a **plain 2D FPN** (forward `:81-103` = upsample → concat → 1×1 conv
  → 3×3 conv). No lift/splat, no depth.
- A whole-tree search found **none** of: `LSSViewTransformer`, `DepthLSSTransform`,
  `LiftSplat`, `ViewTransformer`, `DepthNet`, `BEVDepth`, `BEVDet`, `create_frustum`,
  `get_geometry`, `voxel_pooling`, "frustum", "cumsum_trick" (as LSS machinery).

**Implication:** the depth-prediction head + frustum geometry + lift-splat call
must be **authored from scratch**. There is no existing camera view-transformer to
reuse.

---

## Q2 — Is a BEV-pooling CUDA op present/built?  **VERDICT: SOURCE PRESENT & BUILD-REGISTERED, but NOT COMPILED in this checkout.**

A standard BEVDet/BEVFusion `bev_pool` op exists in source and is wired into the
build, but no compiled artifact is present in the offline checkout.

- Op source:
  - `mmdet3d/ops/bev_pool/bev_pool.py` — `QuickCumsumCuda` autograd `Function`
    (`:37-80`, calls `bev_pool_ext.bev_pool_forward/backward`) and the public
    wrapper `bev_pool(feats, coords, B, D, H, W)` (`:83-97`, sorts ranks → cumsum
    trick → `[B, C, D, H, W]`).
  - `mmdet3d/ops/bev_pool/src/bev_pool.cpp`, `mmdet3d/ops/bev_pool/src/bev_pool_cuda.cu`
    — the CUDA kernels.
- Build registration: `setup.py:246-252`
  `make_cuda_ext(name="bev_pool_ext", module="mmdet3d.ops.bev_pool",
  sources=["src/bev_pool.cpp", "src/bev_pool_cuda.cu"])` → built as a `CUDAExtension`.
- **Not compiled here:** no `.so` artifact exists. `mmdet3d/ops/bev_pool/__init__.py:3`
  executes `from . import bev_pool_ext`, so `import mmdet3d.ops.bev_pool` **raises
  until** the extension is compiled (`python setup.py develop` / `pip install -e .`
  on a CUDA box). It is **not** re-exported from the top-level `mmdet3d/ops/__init__.py`
  (`__all__` there omits it) — import the submodule directly.

**Feasibility note:** this is a *low-risk asset*, not a blocker. A real, suitable
pooling op already exists; it only needs compilation. Its cumsum trick avoids
materializing the dense `[B, C, D, 180, 180]` volume (see Q5), which is the actual
OOM driver. **The only OOM risk is if a dense branch ships a naive Python scatter
instead of using this op.**

---

## Q3 — Camera-calibration access (reuse the existing projection)

The LiDAR→image projection IS-Fusion already uses lives in
`fusion_encoder.py::img_point_sampling`. A DepthNet/LSS frustum can reuse exactly
these tensors and geometry.

- **Calibration tensors** (read from `**kwargs`):
  - `img_aug_matrix` — `fusion_encoder.py:968`
  - `lidar_aug_matrix` — `:969`
  - `lidar2img` (`lidar2image`) — `:970`
  - image size `img_metas[0]['input_shape']` — `:971`
- **Projection computation** (where a frustum can hook the same math):
  - undo LiDAR augmentation — `:984-987`
  - **`lidar2image` rotation + translation — `:990-991`** (the core projection)
  - depth extraction **`dist = cur_coords[:, 2, :]` — `:999`**
  - perspective divide — `:1002-1003`
  - apply `img_aug_matrix` (post-aug pixel transform) — `:1006-1007`
  - normalize to `[-1, 1]` for `grid_sample` — `:1011-1013`
- **Where the tensors originate:** data pipeline `Collect3DV2` meta_keys —
  `configs/isfusion/isfusion_0075voxel.py:290-293`:
  `camera_intrinsics, camera2ego, lidar2ego, lidar2camera, camera2lidar, lidar2img,
  img_aug_matrix, lidar_aug_matrix`.

**Implication:** the dense branch can reuse `lidar2img` + `img_aug_matrix` +
`lidar_aug_matrix` and the projection block at `:984-1013` verbatim to build its
frustum→ego geometry (or its inverse for image→BEV lifting).

---

## Q4 — Sparse depth GT for BEVDepth-style supervision?  **VERDICT: does NOT exist; straightforward to add.**

No per-pixel LiDAR depth-map generation exists anywhere in the pipeline or model.

- Searches for `points2depthmap`, `depth_gt`, `gt_depth`, `depth_map`, `DepthGT`,
  `get_depth` (as a LiDAR→image depth map) returned **no** such generator. Nearby
  hits are unrelated: `mmdet3d/datasets/pipelines/formating.py` per-object `depths`,
  monocular-head object depths, and the `points_cam2img` utility.
- **Reusable projection for building the GT already exists:**
  - `points_cam2img(points_3d, proj_mat, with_depth=True)` —
    `mmdet3d/core/bbox/structures/utils.py:116-151` (returns `[u, v, z]`).
  - The per-camera pixel + depth computation in `img_point_sampling`
    (`fusion_encoder.py:984-1013`, depth at `:999`).
- **Where to add it (recommended):** a new pipeline transform inserted *after*
  `ImageAug3D` / `GlobalRotScaleTransV2`
  (`configs/isfusion/isfusion_0075voxel.py:257-271`), at the point where
  `points`, `lidar2img`, and `img_aug_matrix` all co-exist in `results` — so the
  depth map is aligned to the *augmented* image (BEVDepth convention). A natural
  home for the class is `mmdet3d/datasets/pipelines/transforms_3d.py`.
- **Alternative:** compute the depth map on-the-fly inside the detector where both
  modalities are present — `isfusion.py::extract_img_feat`/`forward` (`:54-101`).

---

## Q5 — Image-feature geometry (memory sizing)

- **Backbone:** Swin-T, `out_indices=[1, 2, 3]` → strides 8/16/32, channels
  `[192, 384, 768]` (`configs/isfusion/isfusion_0075voxel.py:33-49`).
- **Input image size:** `img_scale = (384, 1056)` (H×W) (`:8`, used as
  `ImageAug3D.final_dim` `:259`).
- **Neck:** `GeneralizedLSSFPN`, `out_channels=256`. Critically, the forward returns
  **only `num_ins - 1 = 2` levels** (`generalized_lss.py:90` sets
  `used_backbone_levels = len(laterals) - 1`; outputs built at `:102`). The config's
  `num_outs=3` is effectively ignored.

| Level | Stride | Channels | H × W (from 384×1056) |
|-------|--------|----------|------------------------|
| `img_mlvl_feats[0]` | 8 | 256 | 48 × 132 |
| `img_mlvl_feats[1]` | 16 | 256 | **24 × 66** ← consumed by `img_fv_to_bev` (`fusion_encoder.py:1162`) |

> Note: the in-code comment `# feat: [24,256,32,88]` at `fusion_encoder.py:1033`
> is **stale**. The leading `24` is the batched camera dim (B·N = 4·6), and the real
> per-camera spatial size for level 1 is **24 × 66**, not 32 × 88.

**Memory implication** (per-camera lift at level 1 = 24×66, B=4, N=6 cameras):

- Lifted frustum feature volume `[B·N, C, D, H, W] = [24, 256, D, 24, 66]`. For
  `D ≈ 80` depth bins: ≈ 7.8 × 10⁸ elements ⇒ **~1.6 GB fp16 / ~3.1 GB fp32**
  (roughly doubles with gradients/activations).
- A **naive dense splat** to `[B, C, D, 180, 180] = [4, 256, 80, 180, 180]` ⇒
  ≈ 2.65 × 10⁹ elements ⇒ **~5.3 GB fp16 / ~10.6 GB fp32** — this is the volume the
  `bev_pool` cumsum trick **avoids**, and the real 24 GB OOM driver.
- Lifting at level 0 (48×132) costs ≈ 4× the above and is OOM-prone without the
  compiled op. **Prefer level 1 (24×66) and the `bev_pool` op.**

---

## Q6 — The B_I contract and insertion site

- **B_I** = output of `img_fv_to_bev`:
  `decorated_img_feat = torch.zeros([bs, embed_dims=256, bev_size=180, bev_size=180])`
  — `fusion_encoder.py:1065`, returned at `:1071`.
  `bev_size = voxel_shape // out_size_factor = 1440 // 8 = 180`
  (`configs/isfusion/isfusion_0075voxel.py:13-15`). So **B_I = 256 ch @ 180×180**. ✔
- **Insertion site** (`fusion_encoder.py::forward`):
  - **`:1162`** — `img_bev_feats = self.img_fv_to_bev([img_mlvl_feats[1]], bs, **kwargs)`
  - **`:1167`** — `bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats], dim=1))`
  - `conv_fusion = ConvModule(embed_dims*3 = 768 → embed_dims//2 = 128)`
    (`:862-869`). `lidar_feats` is **512 ch** (pts `SECONDFPN` out `[256, 256]`,
    concatenated), so the concat is `256 + 512 = 768` → matches `conv_fusion` input.

**A dense LSS tensor of 256 ch @ 180×180 can be wired in two ways without touching
`conv_fusion`:**

1. **REPLACE** B_I — swap the `:1162` call to produce the dense tensor instead. The
   concat stays `256 + 512 = 768`; `conv_fusion` unchanged.
2. **COMBINE** — element-wise add (or otherwise merge to 256 ch) the dense tensor
   with B_I before `:1167`. Stays 256 ch; `conv_fusion` unchanged.

> Concatenating a *third* 256-ch tensor (sparse + dense + lidar) would make the
> concat `1024 ch` and require `conv_fusion` to become `1024 → 128` — a model edit,
> out of scope for this report.

---

## Feasibility summary (blunt)

Adding the dense branch is **mostly WIRING plus authoring a camera depth encoder —
it is NOT "build a CUDA op from scratch."**

**Reuse (already present):**
- `bev_pool` CUDA op — source + build-registered; just compile it.
- Full calibration + projection geometry — `lidar2img` / `img_aug_matrix` /
  `lidar_aug_matrix` and the math at `fusion_encoder.py:984-1013`.
- The clean 256 ch @ 180×180 BEV contract and a drop-in insertion point at
  `fusion_encoder.py:1162` / `:1167`.
- `points_cam2img(with_depth=True)` for building depth GT.

**Author (new code):**
- A `DepthNet` head producing a depth distribution over `D` bins from
  `img_mlvl_feats[1]` (256 ch @ 24×66).
- Frustum / `get_geometry` (reusing the Q3 calibration) + the lift-splat call into
  `bev_pool`.
- (Optional, for explicit depth supervision) a `points2depthmap` pipeline transform
  per Q4, plus a depth loss.

**Blockers / risks:**
1. `bev_pool_ext` must be **compiled** (the current server is offline / has no `.so`).
2. **OOM risk only** if the branch uses a naive dense scatter instead of `bev_pool`.
   Using the op and lifting at level 1 (24×66) keeps it well within 24 GB.

No camera encoder or CUDA op needs to be written from scratch; the depth head and
lift-splat glue are the main authoring effort.
