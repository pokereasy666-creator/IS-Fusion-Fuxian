# SHC Backbone Locate-and-Report

**Locate-and-report only — no model/config/training edits. Static analysis (server offline; nothing executed).**

This report locates IS-Fusion's LiDAR sparse-backbone structure and assesses
where an MGAF-style *"additional downsampling + sparse height compression"* (SHC)
insertion would attach, and whether the existing **B_P** channel contract into the
fusion module can be preserved.

---

## 0. Branch + base + D1-free evidence

| Item | Value |
|---|---|
| Report branch | `claude/shc-locate` |
| Branched-from | `origin/claude/4gpu-gradient-accumulation-conf-4wxWs` @ **`41d485840f51c0d824dd812a0945d0d51a4e9ad1`** ("Revert ... reduce A30 4-GPU config LR ...") |
| Why this base | The clean D1-free reproduction baseline — the SAME base used for the AFDT locate report (NOT the AFDT integration branch). SHC is measured against the clean baseline. |

**D1-free re-verification (required gate):**

```
$ git grep -nE "_probe_buffer|alpha_beta|image_classifier|loss_alpha|s_img_logits" \
    41d4858 -- mmdet3d/models/dense_heads/transfusion_head_v2.py
(no output; exit code 1)
```

No matches → baseline is **D1-free**. The file is present at base
(`git cat-file -e 41d4858:mmdet3d/models/dense_heads/transfusion_head_v2.py` → ok).
The working tree at analysis time is byte-identical to `41d4858`, so every
file:line below is as it exists on the base.

---

## 1. The full LiDAR path (config → classes → files)

Config: `configs/isfusion/isfusion_0075voxel.py`. This config has **no `_base_`** —
the entire `model` dict (`type='ISFusionDetector'`) is self-contained in this file.
Detector class: `mmdet3d/models/detectors/isfusion.py:13-14`.

Grid constants (config:6-15): `voxel_size=[0.075,0.075,0.2]`,
`point_cloud_range=[-54,-54,-5,54,54,3]`, `voxel_shape = 108/0.075 = 1440`,
`out_size_factor=8` → BEV `1440/8 = 180`.

| Stage | Config `type` | Config file:line | Class definition file:line |
|---|---|---|---|
| pts_voxel_layer | *(no `type=`)* → `Voxelization` | isfusion_0075voxel.py:58-60 | mmdet3d/ops/voxel/voxelize.py:76 |
| pts_voxel_encoder | `DynamicVFE` (in=5, feat=[64,64]) | isfusion_0075voxel.py:61-71 | mmdet3d/models/voxel_encoders/voxel_encoder.py:288 |
| **pts_middle_encoder** | **`SparseEncoder`** | isfusion_0075voxel.py:72-82 | mmdet3d/models/middle_encoders/sparse_encoder.py:18-19 |
| **pts_backbone** | **`SECONDV2`** | isfusion_0075voxel.py:97-104 | mmdet3d/models/backbones/second.py:98-99 |
| pts_neck | `SECONDFPN` | isfusion_0075voxel.py:106-113 | mmdet3d/models/necks/second_fpn.py:12 |
| (head) | `TransFusionHeadV2`, `in_channels=256*2` | isfusion_0075voxel.py:115-143 (in_ch @ line 119) | mmdet3d/models/dense_heads/transfusion_head_v2.py:507 (param), :631 (`self.in_channels`) |

`SparseEncoder` config args (isfusion_0075voxel.py:72-82): `in_channels=64`,
`sparse_shape=[41, 1440, 1440]`, `base_channels=32`, **`output_channels=256`**,
`order=('conv','norm','act')`,
`encoder_channels=((32,32,64),(64,64,128),(128,128,256),(256,256))`,
`encoder_paddings=((0,0,1),(0,0,1),(0,0,[0,1,1]),(0,0))`, `block_type='basicblock'`.

`SECONDV2` config args (isfusion_0075voxel.py:97-104): `in_channels=128`,
`out_channels=[128,256]`, `layer_nums=[5,5]`, `layer_strides=[1,2]`.

### Resolving the SparseEncoder-vs-SECONDV2 ambiguity

Both classes exist; the observation conflated two different files/classes. Evidence:

- **`SparseEncoder` is the sparse 3D middle encoder** (operates on spconv
  `SparseConvTensor`). It performs the **3D spatial downsampling** (strides 2/4/8)
  *and* the **height compression** `out.dense()` + `view(N, C*D, H, W)` at
  **`sparse_encoder.py:132-136`**. Class `@MIDDLE_ENCODERS.register_module()`,
  `sparse_encoder.py:18-19`.
- **`SECONDV2` is the 2D *dense* BEV backbone** (`nn.Conv2d` via
  `build_conv_layer`). Class `@BACKBONES.register_module()`, `second.py:98-99`.
  Its `ds_layer` (built `second.py:130-138`, applied `second.py:214` in `stage1`
  mode and `second.py:236` in the normal branch) is a **2D** stride-2 conv
  (128→256) — *not* a sparse op.
- **Is SECONDV2 actually wired by this config?** Yes — `pts_backbone=dict(
  type='SECONDV2', ...)` at isfusion_0075voxel.py:98. The plain `SECOND` class
  (`second.py:11`) is defined but **not referenced** by this config. Note SECONDV2
  is *not* called by the detector directly; it is passed into the fusion encoder
  (`isfusion.py:97`, `kwargs['pts_backbone']`) and invoked **inside**
  `fusion_encoder.py:1188` in its `stage1`/`stage2` modes.
- **Which one downsamples (the SHC-relevant 3D path)?** `SparseEncoder` performs
  the sparse 3D XY downsampling (strides 2/4/8) and the z-compression. SECONDV2's
  `ds_layer` is a separate **2D** post-fusion downsample. → **SHC ("sparse
  downsampling + sparse height compression") attaches to `SparseEncoder`.**

---

## 2. Sparse-stage structure (the SHC attach point)

Source: `mmdet3d/models/middle_encoders/sparse_encoder.py`.
Stages built by `make_encoder_layers` (`:142-216`); `conv_input` (`:80-88`);
`conv_out` (`:96-104`). With `block_type='basicblock'`, each non-final stage ends
in a stride-2 `SparseConv3d` downsample (`:182-194`, `indice_key='spconv{i+1}'`);
other blocks are `SparseBasicBlock` (stride 1, `:195-201`).

z starts at `sparse_shape[0]=41`; XY at 1440.

| Stage (F-analog) | XY stride | out ch | z | tensor var (file:line) | retained? |
|---|---|---|---|---|---|
| `conv_input` (F0) | 1 | 32 | 41 | `x` → `encode_features[0]` (:124, :127) | appended + returned, then discarded |
| `encoder_layer1` (F1) | 2 | 64 | 21 | `encode_features[1]` (:128-130) | appended + returned, then discarded |
| `encoder_layer2` (F2) | 4 | 128 | 11 | `encode_features[2]` (:128-130) | appended + returned, then discarded |
| `encoder_layer3` (F3) | 8 | 256 | 5 | `encode_features[3]` (:128-130) | appended + returned, then discarded |
| `encoder_layer4` (F4) | 8 | 256 | 5 | `encode_features[-1]` (:128-130, :132) | appended + returned, then discarded |
| `conv_out` | 8 (z 5→2) | 256 → dense | 2 | `out` / `spatial_features` (:132-136) | **becomes B_P** |

- `encoder_layer4` (last `encoder_channels` entry `(256,256)`) does **not**
  downsample (the stride-2 branch is gated by `i != len-1`, `:183-184`), so F3 and
  F4 are both at XY-stride 8 / 256ch; F4 is the deepest sparse tensor and the SHC
  attach point.
- `conv_out` (`:96-104`): `SparseConv3d`, `kernel=(3,1,1)`, `stride=(2,1,1)`,
  `padding=0`, `indice_key='spconv_down2'`, 256→`output_channels=256`. It
  compresses **z only** (5→2); XY unchanged at 180.

### Height compression + D factor + resulting channels

`sparse_encoder.py:132-136`:

```python
out = self.conv_out(encode_features[-1])         # :132  sparse, XY-stride 8, z=5
spatial_features = out.dense()                   # :133  (N, C=256, D=2, H=180, W=180)
N, C, D, H, W = spatial_features.shape           # :135
spatial_features = spatial_features.view(N, C * D, H, W)  # :136  (N, 512, 180, 180)
```

- z trace: 41 →(layer1 s2)→ 21 →(layer2 s2)→ 11 →(layer3 s2, z-pad 0 via `[0,1,1]`)→ 5 →(conv_out s2 in z)→ **2**. So **D = 2**.
- XY trace: 1440 →720→360→**180** (three stride-2 stages); layer4 & conv_out keep XY=180.
- **Resulting dense BEV channels = `output_channels(256) × D(2) = 512`, at 180×180.**

### Multi-scale retention

All five sparse tensors are appended to `encode_features` (`:127`, `:130`) and
**returned** by `forward` (`:138 return spatial_features, encode_features, kwargs`).
However the **detector discards them**: `isfusion.py:111`
`x, _, kwargs = self.pts_middle_encoder(...)` — only the dense 512-ch BEV (`x`)
propagates. So multi-scale sparse features are *produced and available* but unused.

### spconv API / version

- `sparse_encoder.py:11-15`: gated by `IS_SPCONV2_AVAILABLE`
  (`mmdet3d/ops/spconv/__init__.py:12`):
  `from spconv.pytorch import SparseConvTensor, SparseSequential` (**spconv 2.x**),
  else `from mmcv.ops import ...` (fallback exposing the same API surface). The
  in-file comments saying "spconv v1" are stale/misleading.
- 4-arg constructor `SparseConvTensor(voxel_features, coors, self.sparse_shape,
  batch_size)` (`:122-123`); `indice_key=...` used throughout
  (`:77,:88,:103,:180,:193,:210`); `.dense()` (`:133`). `SparseBasicBlock` and
  `make_sparse_convmodule` imported from `mmdet3d.ops` (`:5`).
- **Verdict:** the code targets the **spconv 2.x** API (with an mmcv.ops fallback).
  Adding new stride-2 `SparseConv3d` downsampling stages is the existing pattern
  (`make_sparse_convmodule(..., stride=2, conv_type='SparseConv3d',
  indice_key=...)`, as already used at `:96-104` and `:185-194`). **No version
  blocker** to in-place new sparse stages.

---

## 3. The B_P contract SHC must preserve

Fusion entry: `fusion_encoder.py` `ISFusionEncoder.forward(img_mlvl_feats,
lidar_feats, bs, **kwargs)` (`:1154-1159`). `embed_dims=256` (config:88).

- **B_P = `lidar_feats` = 512 ch @ 180×180.** It is the dense BEV from
  `SparseEncoder` passed through the detector: `isfusion.py:111` (`x`) → `:113`
  `self.isfusion(pts, x, ...)` → `:99` `self.fusion_encoder(img_feats, pts_feats,
  ...)`.
- **Derivation:** `SparseEncoder.output_channels = 256` (config:77) × `D = 2`
  (§2) = **512**. Spatial 180×180 = `voxel_shape/out_size_factor = 1440/8`.
- **Fusion point** (`fusion_encoder.py:1167`):
  `bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats], dim=1))`
  — `img_bev_feats` is 256 ch (`embed_dims`, from `img_fv_to_bev`, `:1162`),
  `lidar_feats` is 512 ch → concat = **768** @ 180×180.
- **`conv_fusion`** (`fusion_encoder.py:862-869`):
  `ConvModule(in = embed_dims*3 = 768, out = embed_dims//2 = 128, k=3, p=1)`.
  Its input width `embed_dims*3` *implicitly assumes* `lidar_feats = 2*embed_dims =
  512`. Its **output is a fixed 128** regardless of the split.

### Downstream consumers of these channel counts

| Module | Arg | Value | file:line |
|---|---|---|---|
| `conv_fusion` | `in_channels` (`embed_dims*3`) | **768** — the *only direct consumer of B_P's width* | fusion_encoder.py:863 |
| `conv_fusion` | `out_channels` (`embed_dims//2`) | 128 | fusion_encoder.py:864 |
| `SECONDV2` | `in_channels` (= conv_fusion out) | 128 | config:99 (invoked fusion_encoder.py:1188) |
| `SECONDV2` | `out_channels` | [128, 256] | config:100 |
| `SECONDFPN` | `in_channels` / `out_channels` | [128,256] / [256,256] | config:108-109 |
| head `TransFusionHeadV2` | `in_channels` (= Σ neck outs) | 256*2 = 512 | config:119; transfusion_head_v2.py:507,631 |

### If SHC changes B_P's channel count (X instead of 512)

- The **only** module that consumes B_P's raw width is `conv_fusion`'s input.
  Because the `torch.cat` at `:1167` is dynamic and `conv_fusion`'s **output stays
  128**, everything downstream (`SECONDV2` in=128, `SECONDFPN`, head=512) is
  **insulated** and needs **no** change.
- **Minimal edit set** (do NOT apply — enumerated only):
  - `fusion_encoder.py:863` — change `conv_fusion` input from `self.embed_dims*3`
    to `self.embed_dims + X` (i.e. `256 + X`).
  - (No other edits, *provided* `conv_fusion` output and `embed_dims` are unchanged
    and B_P stays 180×180.)
- This corrects an over-broad "cascading dependency" list (one exploration pass
  flagged `pts_backbone.out_channels`, `pts_neck`, head, `embed_dims`, etc.); those
  matter only if one changes `conv_fusion`'s **output** or `embed_dims`, which a
  B_P-*input* width change does not require.
- **Spatial constraint:** B_P must remain **180×180** to concat with the image BEV
  at `:1167`.

---

## 4. Feasibility assessment

**Can MGAF-style SHC (add stride-16/32 sparse stages + sparse-height-compress
F4/F5/F6 to BEV) be inserted into this backbone?**

**Yes.** Attach **inside `SparseEncoder`**, after `encoder_layer4` /
`encode_features[-1]` (XY-stride 8, 256 ch, z=5):

- Add new stride-2 `SparseConv3d` stages **F5** (XY-stride 16 → 90×90) and **F6**
  (XY-stride 32 → 45×45), either by extending `encoder_channels` / the
  `make_encoder_layers` loop (`sparse_encoder.py:142-216`) or as new modules after
  `__init__` `:104`.
- Add per-scale sparse-height-compression mirroring the existing
  `conv_out`+`dense`+`view` block (`:96-104`, `:132-136`) for F4/F5/F6.
- Fuse the multi-scale BEVs and return as the dense `spatial_features`.
- **Cleanest placement is inside `SparseEncoder`** so the multi-scale fold lands in
  the returned dense `x`; otherwise the detector's discarded `_` at `isfusion.py:111`
  would have to be captured (a detector edit).

**Does it require new spconv downsampling layers, and is the present spconv capable?**
Yes (stride-16/32 `SparseConv3d` + new `indice_key`s). The present **spconv 2.x**
API (with mmcv.ops fallback) **supports them via the existing
`make_sparse_convmodule` helper** — already used for stride-2 sparse convs at
`:96-104` and `:185-194`. **No version blocker.**

**Can B_P's 512 ch / 180×180 contract be preserved?**

- **Yes, preservable with zero downstream edits**, *if* SHC upsamples the coarser
  compressed scales (90×90, 45×45) back to 180×180 and projects/sums the
  multi-scale BEV to **exactly 512 channels** before returning. Then `conv_fusion`
  (in `embed_dims*3=768`, out 128) and all downstream modules are untouched.
- **If not preserved** (e.g. scales are concatenated → channels ≠ 512), the change
  is **unavoidable** but minimal: the single edit listed in §3 —
  **`fusion_encoder.py:863`** (`embed_dims*3` → `embed_dims + X`). B_P must still be
  180×180.

**Blockers / watch-items**
- No architectural blocker. It is a standard mmdet3d `SparseEncoder` on the spconv
  2.x API; adding downsampling stages + extra height-compression is an in-place,
  fully-supported modification.
- Watch-items: (a) coarse F5/F6 (90×90 / 45×45) must be upsampled to 180×180 for
  the fusion concat — a *dense* op after each sparse height-compression; (b)
  `encode_features` is discarded by the detector (`isfusion.py:111`), so multi-scale
  SHC is cleanest *inside* `SparseEncoder`; (c) modest extra compute/memory from the
  deeper sparse stages.

---

## 5. Ambiguities / blockers

- **Branch-name note:** the standing default "develop on `claude/keen-euler-1ebNT`"
  is overridden by this task's explicit instruction to create/push
  `claude/shc-locate`; both point at the same base SHA `41d4858`.
- No analysis blockers. All numeric claims were re-derived from source
  (`output_channels=256` is the *config* value, not the class default 128; D=2 from
  the z-trace; 256×2=512; concat 256+512=768→128; neck 256+256=512=head in).
