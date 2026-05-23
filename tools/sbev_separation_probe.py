"""S_bev separation probe for ``alpha_conf_threshold`` (D1 F3, fix #2)
plus recall-at-proposal mechanism gate for D2.

Diagnostic question A (F3 fix #2): does any ``alpha_conf_threshold`` cleanly
separate the BEV-confident-FP population (the precision targets fix #2 wants
to KEEP) from the low-``s_bev`` missed-positive population (the recall cases
fix #2 wants to DROP)? If no threshold separates them, fix #2 is a no-op and
F3 should not launch.

Diagnostic question B (D2 mechanism gate): per class x per range bin, are
the GT objects PRESENT in the proposal set but suppressed (high recall +
low matched s_bev), or ABSENT entirely (low recall)? This is computed from
the Hungarian assigner's GT->proposal output captured under the same
``_probe_buffer`` guard inside ``get_targets_single``. Focused on
``barrier`` and ``traffic_cone`` at 30-50m because that's where D2 must
choose its mechanism. Recall is computed from the actual assigner result,
not approximated from proposal-side data; if gt_records are missing the
analyzer raises.

Mechanics:
    - Build the model from --config and load --checkpoint (same as
      ``tools/test.py``).
    - Build a dataloader over the val split using the **train** pipeline so
      GT annotations are loaded and the head's target assignment fires under
      ``forward_train``. The val .pkl (``nuscenes_infos_val.pkl``) is
      substituted for the train .pkl in the existing train dataset config.
    - Set ``model.pts_bbox_head._probe_buffer = []`` to enable the in-head
      dump (it's None by default; the dump branch is a strict no-op when
      None, so this script is the only path that activates it).
    - Run ``--num-batches`` batches through the model with
      ``return_loss=True`` under ``torch.no_grad()``. Target assignment runs
      and the dump fires inside the loss_alpha block. No backward.
    - Concatenate the dumps; compute per-candidate ``bev_conf``; label
      POSITIVE (matched to GT) / NEGATIVE (assigned-bg) / ignored; report
      quantiles, per-class quantiles, and a threshold sweep with a per-tau
      viability verdict.

Run on the GPU server with the A1 config and the trained A1 checkpoint, e.g.::

    python tools/sbev_separation_probe.py \
        --config configs/isfusion/isfusion_0075voxel_a30_4gpu_d1_A1.py \
        --checkpoint work_dirs/.../d1_A1/run_001/epoch_10.pth \
        --num-batches 200 \
        --out-dir work_dirs/probe_sbev_separation
"""
import argparse
import copy
import csv
import os
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
# Classes the F3 fix #2 hypothesis is most concerned with.
OVERFIRED_CLASSES = ['pedestrian', 'motorcycle', 'bicycle']   # FPs to KEEP
RECALL_CLASSES = ['barrier', 'traffic_cone']                  # missed pos to DROP
SWEEP_TAUS = [0.1, 0.2, 0.3, 0.4, 0.5]
QUANTILES = [10, 25, 50, 75, 90]

# Recall-at-proposal analysis (D2 mechanism gate).
RANGE_BINS = [(0.0, 30.0), (30.0, 50.0), (50.0, 100.0)]
RANGE_LABELS = ['0-30m', '30-50m', '50-100m']
FOCUS_BIN_LO, FOCUS_BIN_HI = 30.0, 50.0       # the bin we report on barrier+cone
RECALL_HIGH = 0.7        # recall_at_proposal >= => "GTs are mostly present"
RECALL_LOW = 0.3         # recall_at_proposal <= => "GTs are mostly absent"
SUPPRESS_MEDIAN = 0.3    # matched s_bev median <= => "suppressed at proposal"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', required=True,
                        help='test config file path (use the A1 config)')
    parser.add_argument('--checkpoint', required=True,
                        help='trained checkpoint (.pth) to probe')
    parser.add_argument('--num-batches', type=int, default=200,
                        help='val batches to run forward_train on')
    parser.add_argument('--out-dir', required=True,
                        help='output dir for CSV + optional PNGs')
    parser.add_argument('--samples-per-gpu', type=int, default=1,
                        help='val batch size (default 1 keeps memory low)')
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Dataloader: val data + train pipeline (so GT loads and forward_train runs)
# --------------------------------------------------------------------------

def _train_pipeline_on_val_dataset_cfg(cfg):
    """Return a dataset-config dict that uses the train pipeline on val data.

    The val dataset uses ``test_pipeline``, which does not load
    ``LoadAnnotations3D`` -- so ``forward_train`` cannot run on val batches
    directly. The simplest correct fix is to take the inner train dataset
    config (unwrapping CBGSDataset if present) and swap its ``ann_file`` to
    the val .pkl. Pipeline and ``test_mode=False`` are preserved, so GT
    loads and the head's target assignment fires.
    """
    train_cfg = copy.deepcopy(cfg.data.train)
    val_ann_file = cfg.data.val['ann_file']
    if isinstance(train_cfg, dict) and train_cfg.get('type') == 'CBGSDataset':
        inner = train_cfg['dataset']
        inner['ann_file'] = val_ann_file
        inner['test_mode'] = False
        if 'load_interval' in inner:
            inner['load_interval'] = 1
        return inner
    train_cfg['ann_file'] = val_ann_file
    train_cfg['test_mode'] = False
    if 'load_interval' in train_cfg:
        train_cfg['load_interval'] = 1
    return train_cfg


# --------------------------------------------------------------------------
# Forward loop + buffer concatenation
# --------------------------------------------------------------------------

def _get_head(model):
    """Unwrap MMDataParallel to reach pts_bbox_head."""
    m = model.module if hasattr(model, 'module') else model
    return m.pts_bbox_head


def collect_probe_records(model, data_loader, num_batches, device):
    head = _get_head(model)
    head._probe_buffer = []
    model.eval()
    seen = 0
    with torch.no_grad():
        for data in data_loader:
            if seen >= num_batches:
                break
            try:
                model(return_loss=True, **data)
            except StopIteration:
                break
            except Exception as e:  # noqa: BLE001
                print(f'[probe] batch {seen} forward failed: {e}',
                      file=sys.stderr)
                seen += 1
                continue
            seen += 1
    buffer = head._probe_buffer
    head._probe_buffer = None  # disable dump immediately after collection
    print(f'[probe] collected {len(buffer)} batches (target {num_batches})')
    return buffer


def flatten_buffer(buffer, num_classes):
    """Return per-candidate arrays concatenated across batches."""
    bev_conf_all = []
    labels_all = []
    weights_all = []
    in_view_all = []
    xy_all = []
    arg_cls_all = []  # argmax BEV class per candidate (for neg labeling)
    for rec in buffer:
        s_bev_pre = rec['s_bev_pre']           # [B, C, K]
        prob = torch.sigmoid(s_bev_pre)        # [B, C, K]
        bev_conf, arg_cls = prob.max(dim=1)    # [B, K], [B, K]
        bev_conf_all.append(bev_conf.reshape(-1).numpy())
        arg_cls_all.append(arg_cls.reshape(-1).numpy())
        labels_all.append(rec['final_labels'].numpy())                 # [B*K]
        weights_all.append(rec['final_label_weights'].numpy())         # [B*K]
        in_view_all.append(rec['in_any_view'].reshape(-1).numpy())     # [B*K]
        xy = rec.get('candidate_xy', None)
        if xy is not None:
            xy_all.append(xy.reshape(-1, 2).numpy())                   # [B*K, 2]
    out = dict(
        bev_conf=np.concatenate(bev_conf_all),
        labels=np.concatenate(labels_all),
        weights=np.concatenate(weights_all),
        in_view=np.concatenate(in_view_all),
        arg_cls=np.concatenate(arg_cls_all),
    )
    if xy_all:
        out['xy'] = np.concatenate(xy_all, axis=0)
    return out


def collect_gt_entries(buffer):
    """Flatten per-batch gt_records into a per-GT entry list.

    Each entry: dict(cls=int, range_m=float, matched=bool, matched_sbev=float).
    ``matched_sbev`` is sigmoid(s_bev_pre[b, :, prop_idx]).max() for the
    matched proposal, or NaN when unmatched. Raises if any batch lacks
    gt_records -- per the task spec we do NOT approximate recall from
    proposal-side data.
    """
    entries = []
    missing_batches = 0
    for batch_rec in buffer:
        gt_records = batch_rec.get('gt_records', None)
        if gt_records is None:
            missing_batches += 1
            continue
        s_bev_pre = batch_rec['s_bev_pre']  # [B, C, K]
        K = s_bev_pre.shape[-1]
        for sample_rec in gt_records:
            b = int(sample_rec['sample_idx'])
            labels = sample_rec['gt_labels_3d']           # [num_gt]
            centers = sample_rec['gt_centers_xy']          # [num_gt, 2]
            gt_inds = sample_rec['final_layer_gt_inds']    # [num_proposals_final]
            num_gt = int(sample_rec['num_gts'])
            if num_gt == 0:
                continue
            if int(gt_inds.numel()) != K:
                raise RuntimeError(
                    f'gt_inds length {int(gt_inds.numel())} != s_bev_pre K={K}; '
                    'final-layer assignment / proposal-count mismatch -- the '
                    'capture point in get_targets_single is misaligned with '
                    'the loss_alpha dump. Refusing to compute recall.'
                )
            sigmoid_b = torch.sigmoid(s_bev_pre[b])        # [C, K]
            max_per_prop = sigmoid_b.max(dim=0).values     # [K]
            for j in range(num_gt):
                matched_idx_t = (gt_inds == (j + 1)).nonzero(as_tuple=False)
                matched = matched_idx_t.numel() > 0
                if matched:
                    # Hungarian assigner produces at most one proposal per GT.
                    prop_idx = int(matched_idx_t[0].item())
                    matched_sbev = float(max_per_prop[prop_idx].item())
                else:
                    matched_sbev = float('nan')
                rng = float(
                    torch.sqrt(centers[j, 0] ** 2 + centers[j, 1] ** 2).item()
                )
                entries.append(dict(
                    cls=int(labels[j].item()),
                    range_m=rng,
                    matched=bool(matched),
                    matched_sbev=matched_sbev,
                ))
    if missing_batches > 0:
        raise RuntimeError(
            f'{missing_batches}/{len(buffer)} batches lack gt_records: the '
            "head's get_targets_single capture did not fire. Recall-at-"
            'proposal cannot be reliably computed from proposal-side data.'
        )
    return entries


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _quantile_str(x, qs=QUANTILES):
    if x.size == 0:
        return '  '.join([f'q{q}=NA' for q in qs])
    vals = np.percentile(x, qs)
    return '  '.join([f'q{q}={v:.3f}' for q, v in zip(qs, vals)])


def analyze(records, num_classes, out_dir):
    bev_conf = records['bev_conf']
    labels = records['labels']
    weights = records['weights']
    arg_cls = records['arg_cls']

    supervised = weights > 0
    is_pos = supervised & (labels < num_classes)
    is_neg = supervised & (labels >= num_classes)

    print()
    print('=' * 72)
    print(f'Total candidates: {bev_conf.size}  '
          f'supervised: {supervised.sum()}  '
          f'pos: {is_pos.sum()}  neg: {is_neg.sum()}')
    print('=' * 72)
    print()

    # ---- (1) overall pos vs neg quantiles ---------------------------------
    print('--- bev_conf quantiles (overall) ---')
    print(f'POSITIVES (matched-GT, recall pool incl. low s_bev cases)')
    print(f'  {_quantile_str(bev_conf[is_pos])}')
    print(f'NEGATIVES (assigned-bg, precision pool incl. confident FPs)')
    print(f'  {_quantile_str(bev_conf[is_neg])}')
    print()

    # ---- (2) per-class quantiles, with attention to over-fired & recall ---
    print('--- per-class bev_conf quantiles (POS: by matched class; '
          'NEG: by argmax BEV class) ---')
    print(f'{"class":<22} {"role":<5} {"n":>7}  '
          + '  '.join([f'q{q}'.ljust(8) for q in QUANTILES]))
    per_class_rows = []
    for c, name in enumerate(CLASS_NAMES[:num_classes]):
        pos_mask = is_pos & (labels == c)
        neg_mask = is_neg & (arg_cls == c)
        for role, mask in [('POS', pos_mask), ('NEG', neg_mask)]:
            n = int(mask.sum())
            if n == 0:
                row_vals = [None] * len(QUANTILES)
                print(f'{name:<22} {role:<5} {n:>7}  '
                      + '  '.join(['NA      '] * len(QUANTILES)))
            else:
                row_vals = np.percentile(bev_conf[mask], QUANTILES).tolist()
                print(f'{name:<22} {role:<5} {n:>7}  '
                      + '  '.join([f'{v:.4f}  ' for v in row_vals]))
            per_class_rows.append(dict(
                cls=name, role=role, n=n,
                **{f'q{q}': v for q, v in zip(QUANTILES, row_vals)}
            ))
    print()

    print('Special attention (fix #2 hypothesis):')
    for cls_name in OVERFIRED_CLASSES:
        c = CLASS_NAMES.index(cls_name)
        m = is_neg & (arg_cls == c)
        print(f'  {cls_name:<14} NEG (FPs to KEEP):       n={int(m.sum())}  '
              f'{_quantile_str(bev_conf[m])}')
    for cls_name in RECALL_CLASSES:
        c = CLASS_NAMES.index(cls_name)
        m = is_pos & (labels == c)
        print(f'  {cls_name:<14} POS (missed pos to DROP): n={int(m.sum())}  '
              f'{_quantile_str(bev_conf[m])}')
    print()

    # ---- (3) threshold sweep ---------------------------------------------
    overfired_neg_mask = is_neg & np.isin(arg_cls, [
        CLASS_NAMES.index(n) for n in OVERFIRED_CLASSES])
    recall_pos_mask = is_pos & np.isin(labels, [
        CLASS_NAMES.index(n) for n in RECALL_CLASSES])

    print('--- threshold sweep ---')
    print(f'{"tau":>5}  {"%sup_kept":>10}  {"%pos_kept":>10}  '
          f'{"%neg_kept":>10}  {"%recall_pos_kept":>17}  '
          f'{"%overfired_neg_kept":>20}  verdict')
    sweep_rows = []
    any_viable = False
    for tau in SWEEP_TAUS:
        kept = bev_conf >= tau
        pct_sup = _safe_pct(kept & supervised, supervised)
        pct_pos = _safe_pct(kept & is_pos, is_pos)
        pct_neg = _safe_pct(kept & is_neg, is_neg)
        pct_recall_pos = _safe_pct(kept & recall_pos_mask, recall_pos_mask)
        pct_overfired_neg = _safe_pct(
            kept & overfired_neg_mask, overfired_neg_mask)
        viable = (pct_overfired_neg >= 50.0) and (pct_recall_pos <= 20.0)
        any_viable = any_viable or viable
        verdict = 'viable' if viable else 'no'
        print(f'{tau:>5.2f}  {pct_sup:>9.1f}%  {pct_pos:>9.1f}%  '
              f'{pct_neg:>9.1f}%  {pct_recall_pos:>16.1f}%  '
              f'{pct_overfired_neg:>19.1f}%  {verdict}')
        sweep_rows.append(dict(
            tau=tau, pct_sup=pct_sup, pct_pos=pct_pos, pct_neg=pct_neg,
            pct_recall_pos=pct_recall_pos,
            pct_overfired_neg=pct_overfired_neg, viable=viable,
        ))

    print()
    print('=' * 72)
    if any_viable:
        print('CONCLUSION: at least one tau in {0.1..0.5} separates '
              'confident FPs (KEEP) from missed positives (DROP).')
        print('  => fix #2 is well-posed; F3 can launch (consider also '
              'probing a finer-grained sweep around the viable tau).')
    else:
        print('CONCLUSION: NO tau in {0.1..0.5} achieves the precision/'
              'recall split that fix #2 assumes.')
        print('  => fix #2 may be a no-op; do NOT launch F3 without '
              'revisiting the hypothesis.')
    print('=' * 72)
    print()

    # ---- save CSVs --------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, 'per_class_quantiles.csv'),
               per_class_rows)
    _write_csv(os.path.join(out_dir, 'threshold_sweep.csv'), sweep_rows)
    print(f'[probe] CSVs written under {out_dir}')

    # ---- optional PNG -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, 1, 51)
        ax.hist(bev_conf[is_pos], bins=bins, alpha=0.5, label='POS', density=True)
        ax.hist(bev_conf[is_neg], bins=bins, alpha=0.5, label='NEG', density=True)
        for tau in SWEEP_TAUS:
            ax.axvline(tau, ls='--', lw=0.6, color='gray')
        ax.set_xlabel('bev_conf = sigmoid(s_bev_pre).max(class)')
        ax.set_ylabel('density')
        ax.set_title('s_bev separation: POS vs NEG')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'sbev_pos_vs_neg.png'), dpi=120)
        plt.close(fig)
        print(f'[probe] PNG written under {out_dir}')
    except Exception as e:  # noqa: BLE001
        print(f'[probe] matplotlib unavailable / failed ({e}); skip PNG')


def _safe_pct(numer_mask, denom_mask):
    d = int(denom_mask.sum())
    if d == 0:
        return float('nan')
    return 100.0 * int((numer_mask & denom_mask).sum()) / d


def _range_bin_label(r):
    for (lo, hi), label in zip(RANGE_BINS, RANGE_LABELS):
        if lo <= r < hi:
            return label
    return None


def _matched_sbev_quantiles(matched_sbev_list):
    arr = np.array([v for v in matched_sbev_list if not np.isnan(v)],
                   dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan')
    q10, q50, q90 = np.percentile(arr, [10, 50, 90])
    return float(q10), float(q50), float(q90)


def _fmt(x, w=6, p=3):
    if isinstance(x, float) and (x != x):  # NaN
        return ' ' * (w - 2) + 'NA'
    return f'{x:>{w}.{p}f}'


def analyze_recall(entries, num_classes, out_dir):
    """Per-class x per-range recall-at-proposal + matched s_bev quantiles.

    Decides D2's mechanism: for barrier/traffic_cone at 30-50m, are the GT
    objects present in the candidate set but suppressed (high recall +
    low matched s_bev median) or absent entirely (low recall)?
    """
    print()
    print('=' * 72)
    print('RECALL-AT-PROPOSAL (per class x per range bin) -- D2 mechanism gate')
    print('=' * 72)
    print(f'Total GT instances across probed batches: {len(entries)}')
    print()
    print(f'{"class":<22} {"range":<8} {"n_gt":>7} {"matched":>8} '
          f'{"recall":>8}   matched_sbev (q10/q50/q90)')
    rows = []
    for c, name in enumerate(CLASS_NAMES[:num_classes]):
        for (lo, hi), label in zip(RANGE_BINS, RANGE_LABELS):
            sel = [e for e in entries if e['cls'] == c
                   and lo <= e['range_m'] < hi]
            n_gt = len(sel)
            matched_sel = [e for e in sel if e['matched']]
            n_matched = len(matched_sel)
            recall = (n_matched / n_gt) if n_gt > 0 else float('nan')
            q10, q50, q90 = _matched_sbev_quantiles(
                [e['matched_sbev'] for e in matched_sel])
            print(f'{name:<22} {label:<8} {n_gt:>7} {n_matched:>8} '
                  f'{_fmt(recall, 8, 3)}   '
                  f'{_fmt(q10)}  {_fmt(q50)}  {_fmt(q90)}')
            rows.append(dict(
                cls=name, range=label, n_gt=n_gt, n_matched=n_matched,
                recall=recall,
                matched_sbev_q10=q10,
                matched_sbev_q50=q50,
                matched_sbev_q90=q90,
            ))

    # ---- focused summary: barrier + traffic_cone at 30-50m ---------------
    print()
    print('--- FOCUSED: barrier + traffic_cone at '
          f'{FOCUS_BIN_LO:.0f}-{FOCUS_BIN_HI:.0f}m ---')
    print(f'(verdict thresholds, auditable: recall_high={RECALL_HIGH}, '
          f'recall_low={RECALL_LOW}, suppress_median={SUPPRESS_MEDIAN})')
    focus_rows = []
    n_gt_focus_total = 0
    for cls_name in RECALL_CLASSES:
        c = CLASS_NAMES.index(cls_name)
        sel = [e for e in entries
               if e['cls'] == c
               and FOCUS_BIN_LO <= e['range_m'] < FOCUS_BIN_HI]
        n_gt = len(sel)
        n_gt_focus_total += n_gt
        matched_sel = [e for e in sel if e['matched']]
        n_matched = len(matched_sel)
        recall = (n_matched / n_gt) if n_gt > 0 else float('nan')
        q10, q50, q90 = _matched_sbev_quantiles(
            [e['matched_sbev'] for e in matched_sel])

        if n_gt == 0:
            verdict = 'no_GTs_in_bin'
        elif recall >= RECALL_HIGH and (q50 == q50) and q50 <= SUPPRESS_MEDIAN:
            verdict = 'present-but-suppressed'
        elif recall <= RECALL_LOW:
            verdict = 'absent'
        else:
            verdict = 'mixed'

        print(f'  {cls_name:<14} n_gt={n_gt:>5}  matched={n_matched:>5}  '
              f'recall={_fmt(recall, 5, 3)}  '
              f'sbev q10/q50/q90={_fmt(q10)}/{_fmt(q50)}/{_fmt(q90)}  '
              f'-> {verdict}')
        focus_rows.append(dict(
            cls=cls_name,
            range_lo=FOCUS_BIN_LO,
            range_hi=FOCUS_BIN_HI,
            n_gt=n_gt,
            n_matched=n_matched,
            recall=recall,
            matched_sbev_q10=q10,
            matched_sbev_q50=q50,
            matched_sbev_q90=q90,
            recall_high=RECALL_HIGH,
            recall_low=RECALL_LOW,
            suppress_median=SUPPRESS_MEDIAN,
            verdict=verdict,
        ))

    print()
    print('*' * 72)
    print(f'AP-headroom denominator: n_gt(barrier + traffic_cone @ '
          f'{FOCUS_BIN_LO:.0f}-{FOCUS_BIN_HI:.0f}m) = {n_gt_focus_total}')
    print('*' * 72)
    print()

    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, 'recall_at_proposal.csv'), rows)
    _write_csv(os.path.join(out_dir, 'recall_focused_bcone_3050.csv'),
               focus_rows)
    print(f'[probe] recall_at_proposal.csv + '
          'recall_focused_bcone_3050.csv written under '
          f'{out_dir}')


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# Inline guard sanity-check (does NOT need torch/mmdet3d at import time):
# verify the head's dump branch is skipped when _probe_buffer is None.
# --------------------------------------------------------------------------

def _verify_guard_default_off():
    """Static check: read the head source and confirm the dump is gated by
    ``if self._probe_buffer is not None:``. This runs in __main__ before any
    heavy imports so a broken guard fails loudly even without a checkpoint.
    """
    import pathlib
    head_path = pathlib.Path(__file__).resolve().parent.parent / (
        'mmdet3d/models/dense_heads/transfusion_head_v2.py')
    src = head_path.read_text()
    assert 'self._probe_buffer = None' in src, (
        'head missing default-off init: self._probe_buffer = None')
    assert 'if self._probe_buffer is not None:' in src, (
        'head missing guard: if self._probe_buffer is not None:')
    print('[probe] guard check OK: dump is default-off')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    _verify_guard_default_off()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = cfg.get('train_cfg', cfg.model.get('train_cfg', None))

    # val data + train pipeline (so GT loads and forward_train runs)
    probe_dataset_cfg = _train_pipeline_on_val_dataset_cfg(cfg)
    dataset = build_dataset(probe_dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])
        device = torch.device('cuda')
    else:
        model = MMDataParallel(model, device_ids=[0])
        device = torch.device('cpu')

    buffer = collect_probe_records(
        model, data_loader, args.num_batches, device)
    if not buffer:
        print('[probe] ERROR: no records collected -- the dump never fired. '
              'Verify the model has an image_classifier head and the loss '
              'reached the loss_alpha branch.', file=sys.stderr)
        sys.exit(1)

    num_classes = len(getattr(dataset, 'CLASSES', CLASS_NAMES))
    records = flatten_buffer(buffer, num_classes)
    analyze(records, num_classes, args.out_dir)

    # GT-side: per-class per-range recall-at-proposal (D2 mechanism gate).
    # Raises if gt_records are missing -- we do NOT approximate recall.
    gt_entries = collect_gt_entries(buffer)
    analyze_recall(gt_entries, num_classes, args.out_dir)


if __name__ == '__main__':
    main()
