# AFDT Fusion-Slot Locate-and-Report

**Task:** Locate the HSF multimodal-BEV fusion point in IS-Fusion (paper Eq. 6,
`B_F = f_conv([B_I, B_P])`) and report tensor shapes there. **Locate-and-report only —
no model/config/training code was modified.** Shapes were inferred **statically** from
code + config arithmetic (server offline; no CUDA/spconv/data; nothing was run).

---

## STEP 0 — Branch & D1-free evidence

- **New branch:** `claude/afdt-locate`
- **BASE branch:** `claude/4gpu-gradient-accumulation-conf-4wxWs`
- **Branched-from SHA:** `41d485840f51c0d824dd812a0945d0d51a4e9ad1`

The base is the clean reproduction baseline. After `git fetch --all`, the full remote
set was enumerated; the forbidden D1 line (names containing `d1`/`f2`/`f3`/`zerosum`/
`confmask`/`alpha`) — `origin/claude/d1-f3-zerosum-confmask`,
`origin/claude/fix-alpha-gradient-f2-2qLMv`, `origin/claude/fixed-alpha-experiment`,
and the related `origin/claude/image-classifier-head-yRrFR` — were **excluded**. The
base was verified D1-free on the freshly-fetched ref:

```
$ B=origin/claude/4gpu-gradient-accumulation-conf-4wxWs
$ git rev-parse $B
41d485840f51c0d824dd812a0945d0d51a4e9ad1

# (a) reproduction config present:
$ git ls-tree -r --name-only $B -- configs/isfusion/isfusion_0075voxel_a30_4gpu.py
configs/isfusion/isfusion_0075voxel_a30_4gpu.py        # FOUND

# (b1) head free of all D1 artifacts (no matches => clean):
$ git grep -nE "_probe_buffer|alpha_beta|image_classifier|loss_alpha|s_img_logits" \
      $B -- mmdet3d/models/dense_heads/transfusion_head_v2.py
(no matches)                                            # CLEAN

# (b2) image_classifier_head.py absent (blank => absent):
$ git ls-tree -r --name-only $B -- mmdet3d/models/dense_heads/image_classifier_head.py
(blank)                                                 # ABSENT
$ git ls-tree -r --name-only $B -- mmdet3d/models/dense_heads/ | grep -iE "classifier|probe|alpha"
(none matching)
```

All three checks pass: config present, head has none of the five D1 tokens, and there
is no `image_classifier_head.py`.

---

## STEP 1 — Fusion point (single, unambiguous)

**File:** `mmdet3d/models/middle_encoders/fusion_encoder.py`
**Class:** `ISFusionEncoder` (line 834) — registered `@FUSION_LAYERS.register_module()`,
instantiated as the detector's `fusion_encoder` (config `isfusion_0075voxel.py:85`).

**The fusion line is `fusion_encoder.py:1167`:**

```python
bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats], dim=1))
```

`dim=1` is the channel axis, and the cat order is `[B_I, B_P]` (image BEV first, point
BEV second) — exactly Eq. 6 `B_F = f_conv([B_I, B_P])`. `self.conv_fusion` is the 3×3
`f_conv`, defined at `fusion_encoder.py:861-869`:

```python
        self.embed_dims = embed_dims                       # 861
        self.conv_fusion = ConvModule(                     # 862
            self.embed_dims*3,                             # in_channels  = 256*3 = 768
            self.embed_dims//2,                            # out_channels = 256//2 = 128
            kernel_size=3,                                 # 3x3 conv
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )                                              # 869
```

### Verbatim `forward()` snippet (`fusion_encoder.py:1154-1191`)

```python
    @auto_fp16()                                                                    # 1154
    def forward(self,                                                               # 1155
                img_mlvl_feats,                                                     # 1156
                lidar_feats,                                                        # 1157
                bs,                                                                 # 1158
                **kwargs):                                                          # 1159


        img_bev_feats = self.img_fv_to_bev([img_mlvl_feats[1]], bs, **kwargs)       # 1162  B_I

        kwargs.update(dict(img_bev_feats=img_bev_feats))                            # 1164
        kwargs.update(dict(lidar_feats=lidar_feats))                                # 1165

        bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats], dim=1))# 1167  <-- FUSION (Eq.6)

        grid_features = bev_feats.flatten(2, 3).permute(0, 2, 1).reshape(-1, bev_feats.shape[1])  # 1169
        bev_coords = self.create_dense_coord(self.bev_size, self.bev_size, bs).type_as(grid_features).int()  # 1170
        this_coords = []                                                            # 1171
        for k in range(bs):                                                         # 1172
            this_coord = bev_coords[k].reshape(4, -1).transpose(1, 0)               # 1173
            this_coords.append(this_coord)                                          # 1174
        grid_coords = torch.cat(this_coords, dim=0)                                 # 1175

        pts_backbone = kwargs.get('pts_backbone', None)                             # 1177

        ins_hm = None                                                               # 1179
        return_feats = []                                                           # 1180
        for i in range(len(self.get_regions)):                                      # 1181
            x = self.get_regions[i](grid_features, grid_coords, bs)                 # 1182
            x = self.grid2region_att[i](x)                                          # 1183

            if i == 0:                                                              # 1185
                x[0], ins_hm = self.instance_fusion(bev_feats, x[0], bs, **kwargs)  # 1186  (IGF / instance stage)

            grid_features, grid_coords, this_feat = pts_backbone(x, 'stage{}'.format(i+1))  # 1188
            return_feats.append(this_feat)                                          # 1189

        return return_feats, ins_hm                                                 # 1191
```

The fused `bev_feats` (B_F) feeds the region/grid attention (`get_regions`/
`grid2region_att`) and the instance stage `instance_fusion` (line 1186), then onward to
`pts_backbone`/`pts_neck` and the detection head — i.e. B_F is the BEV tensor consumed
by the IGF / instance-selection stage and ultimately the head.

---

## STEP 2 — Shapes at the fusion point (static)

### How each tensor is constructed

- **B_I = `img_bev_feats`** — built in `img_fv_to_bev` (`fusion_encoder.py:1047-1071`),
  called at line 1162. Its channel count is fixed at `self.embed_dims` by the
  allocation at **`fusion_encoder.py:1065`**:
  ```python
  decorated_img_feat = torch.zeros([bs, self.embed_dims, self.bev_size, self.bev_size]).type_as(mlvl_feats[0])
  ```
  ⇒ B_I = **256** channels, **180×180**. (Source image features come from
  `img_backbone`+`img_neck`=`GeneralizedLSSFPN out_channels=256`,
  config `isfusion_0075voxel.py:50-55`; `img_mlvl_feats[1]` is the level used.)

- **B_P = `lidar_feats`** — the `forward` argument, passed from the detector:
  `isfusion.py:111` `x, _, kwargs = self.pts_middle_encoder(...)` → `isfusion.py:113`
  `self.isfusion(pts, x, ...)` → `isfusion.py:99`
  `self.fusion_encoder(img_feats, pts_feats, batch_size, ...)`. The middle encoder is
  `SparseEncoder` (config `isfusion_0075voxel.py:72-82`, `output_channels=256`), whose
  dense BEV is produced by folding depth into channels at
  **`sparse_encoder.py:132-136`**:
  ```python
  out = self.conv_out(encode_features[-1])
  spatial_features = out.dense()
  N, C, D, H, W = spatial_features.shape
  spatial_features = spatial_features.view(N, C * D, H, W)   # C=256, D=2 -> 512
  ```
  ⇒ B_P = **512** channels (`output_channels` 256 × depth-fold D=2), **180×180**.

### Shape table

| tensor | channels | spatial | where constructed (file:line) |
|---|---|---|---|
| **B_I** `img_bev_feats` | **256** (`embed_dims`) | 180×180 | `fusion_encoder.py:1065` (in `img_fv_to_bev`); called `:1162` |
| **B_P** `lidar_feats` | **512** (`output_channels` 256 × D=2) | 180×180 | `isfusion.py:111`→`:113`→`:99`; reshape `sparse_encoder.py:132-136`; cfg `isfusion_0075voxel.py:72-82` |
| `cat([B_I, B_P], dim=1)` | **768** | 180×180 | `fusion_encoder.py:1167` |
| **B_F** `bev_feats` (fused) | **128** (`embed_dims//2`) | 180×180 | `conv_fusion`, `fusion_encoder.py:1167` (def `:862-869`) |

### Do B_I and B_P have equal channels?

**NO.** **B_I = 256, B_P = 512.** They differ by 2×.

This is forced by the architecture, not assumed: `conv_fusion.in_channels = embed_dims*3
= 768` (`fusion_encoder.py:863`) and B_I = `embed_dims` = 256 (`:1065`), so
B_P = 768 − 256 = **512** = `output_channels`(256) × D(2). Downstream wiring
corroborates: `pts_backbone.in_channels = 128` (= `conv_fusion` out, cfg
`isfusion_0075voxel.py:99`) and `pts_bbox_head.in_channels = 256*2 = 512` (cfg `:119`).

> **AFDT flag:** AFDT needs matched channel counts for B_I and B_P. Because they are
> **unequal (256 vs 512)**, integrating AFDT at this slot **must add a projection** —
> e.g. a 1×1 conv to lift B_I 256→512, or to reduce B_P 512→256, or to map both to a
> common C. The fused output B_F is **128** channels, so anything operating on B_F (vs.
> the pre-fusion inputs) sees 128.

### Confirmed BEV resolution: 180×180

Derived from config `isfusion_0075voxel.py:6-15`:
```python
voxel_size        = [0.075, 0.075, 0.2]                                   # :6
point_cloud_range = [-54, -54, -5, 54, 54, 3]                             # :7
out_size_factor   = 8                                                     # :13
voxel_shape = int((point_cloud_range[3]-point_cloud_range[0])//voxel_size[0])  # :14  = int(108//0.075) = 1440
bev_size    = voxel_shape // out_size_factor                              # :15  = 1440 // 8 = 180
```
`bev_size=180` is passed to `ISFusionEncoder` (cfg `:89`) and used directly to allocate
the BEV grid (`fusion_encoder.py:851`, `:1065`, `create_2D_grid`/`create_dense_coord`).
So the fusion point operates at **180×180**.

### dtype / device

`forward` is decorated `@auto_fp16()` (`fusion_encoder.py:1154`) and
`SparseEncoder.forward` is `@auto_fp16(apply_to=('voxel_features',))`, so under AMP the
fusion-point tensors are **fp16** (fp32 master weights). `img_bev_feats` is created with
`.type_as(mlvl_feats[0])` (`:1065`), inheriting the image-feature dtype/device. Device =
**CUDA** (model uses spconv + GPU voxelization). Layout is contiguous NCHW.

---

## Ambiguity / candidate fusion points

- **Single multimodal-BEV fusion point.** `fusion_encoder.py:1167` is the only place a
  channel-axis `torch.cat` feeds a 3×3 conv to fuse image-BEV with point-BEV. Every
  other `torch.cat` in the file is unrelated: MultiheadAttention internals (lines
  386-436), positional embeddings (692, 702), deformable-attn flattening (818-821),
  coordinate bases (910, 1080, 1085), multi-camera feature sampling
  (`img_point_sampling`, 1039), and grid-coord assembly (1175).
- **Non-candidates (not wired in this config):** `fusion_layers/PointFusion`,
  `fusion_layers/VoteFusion`, and `voxel_encoders/DynamicFusionVFE` exist in the repo
  but are **not used** by `isfusion_0075voxel*.py` — the config sets
  `pts_voxel_encoder=DynamicVFE` and defines no `pts_fusion_layer`. The only
  multimodal BEV fusion is `ISFusionEncoder.conv_fusion`.
- `instance_fusion` (`fusion_encoder.py:1092`, called `:1186`) is the **IGF/instance**
  stage that consumes the already-fused B_F; it is downstream of the HSF fusion point,
  not the Eq. 6 fusion itself.
