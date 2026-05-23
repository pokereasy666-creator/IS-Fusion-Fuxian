#!/usr/bin/env python
"""Per-class alpha tracker for D1 runs. F3-aware: reconstructs the effective
fusion log-weight beta + (alpha - alpha.mean()) when the zero-sum reparam is
present; falls back to raw alpha for F2 checkpoints. diff_raw and spread are
invariant to the centering, so they stay comparable across F2 and F3; only the
absolute log-weights and softplus columns change (now correct under reparam)."""
import argparse, csv, glob, os, re, sys
import torch
import torch.nn.functional as F

CLASS_NAMES = ["car","truck","construction_vehicle","bus","trailer",
               "barrier","motorcycle","bicycle","pedestrian","traffic_cone"]
OVERFIRED = ["pedestrian","motorcycle","bicycle"]
CONFIDENT = ["car","truck","bus"]
ALPHA_KEY = "pts_bbox_head.image_classifier.alpha"
BETA_KEY  = "pts_bbox_head.image_classifier.alpha_beta"
EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")

def effective_log_weight(sd):
    """Return (logw[10], beta, raw_mean) reconstructing the F3 reparam if present."""
    if ALPHA_KEY not in sd:
        cands = [k for k in sd if k.endswith(".alpha") and "image_classifier" in k]
        if not cands: return None, None, None
        key = cands[0]
    else:
        key = ALPHA_KEY
    alpha = sd[key].detach().float().flatten()
    raw_mean = float(alpha.mean())
    if BETA_KEY in sd:                          # F3: reparam present
        beta = float(sd[BETA_KEY].detach().float().flatten()[0])
        logw = beta + (alpha - alpha.mean())
    else:                                       # F2: raw alpha is the log-weight
        beta = float("nan")
        logw = alpha
    return logw, beta, raw_mean

def group_mean(d, names):
    v = [d[n] for n in names if n in d]
    return sum(v)/len(v) if v else float("nan")

def process(path):
    m = EPOCH_RE.search(os.path.basename(path))
    if not m: return None
    try: ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        sys.stderr.write("WARN load %s: %s\n" % (path, e)); return None
    sd = ckpt.get("state_dict", ckpt)
    logw, beta, raw_mean = effective_log_weight(sd)
    if logw is None:
        sys.stderr.write("WARN no alpha in %s\n" % path); return None
    a = logw.tolist(); sp = F.softplus(logw).tolist(); n = len(a)
    row = {"epoch": int(m.group(1)), "beta": round(beta,6) if beta==beta else "nan",
           "alpha_raw_mean": round(raw_mean,6), "mtime": int(os.path.getmtime(path))}
    if n == len(CLASS_NAMES):
        ad = dict(zip(CLASS_NAMES,a)); spd = dict(zip(CLASS_NAMES,sp))
        for c in CLASS_NAMES: row["a_"+c] = round(ad[c],6)
        row["overfired_mean"] = round(group_mean(ad,OVERFIRED),6)
        row["confident_mean"] = round(group_mean(ad,CONFIDENT),6)
        row["diff_raw"] = round(row["overfired_mean"]-row["confident_mean"],6)
        row["sp_overfired_mean"] = round(group_mean(spd,OVERFIRED),6)
        row["sp_confident_mean"] = round(group_mean(spd,CONFIDENT),6)
    else:
        for i,v in enumerate(a): row["a_%d"%i]=round(v,6)
        for f in ("overfired_mean","confident_mean","diff_raw",
                  "sp_overfired_mean","sp_confident_mean"): row[f]=float("nan")
    row["alpha_min"]=round(min(a),6); row["alpha_max"]=round(max(a),6)
    row["spread"]=round(max(a)-min(a),6); row["alpha_mean"]=round(sum(a)/n,6)
    return row

def main():
    ck = sorted(glob.glob(os.path.join(ARGS.run_dir,"epoch_*.pth")),
                key=lambda p:int(EPOCH_RE.search(os.path.basename(p)).group(1))
                if EPOCH_RE.search(os.path.basename(p)) else -1)
    if not ck: sys.exit("no epoch_*.pth in %s"%ARGS.run_dir)
    out = ARGS.out or os.path.join(ARGS.run_dir,"alpha_tracking.csv")
    existing = {}
    if os.path.exists(out) and not ARGS.force:
        with open(out,newline="") as f:
            existing = {int(r["epoch"]):r for r in csv.DictReader(f)}
    rows = dict(existing); new=[]
    for p in ck:
        e=int(EPOCH_RE.search(os.path.basename(p)).group(1))
        if e in rows and not ARGS.force: continue
        r=process(p)
        if r: rows[e]=r; new.append(e)
    if not rows: sys.exit("no alpha extracted")
    ordered=[rows[e] for e in sorted(rows,key=int)]
    cols=(["epoch"]+["a_"+c for c in CLASS_NAMES]+
          ["alpha_min","alpha_max","spread","alpha_mean","overfired_mean",
           "confident_mean","diff_raw","sp_overfired_mean","sp_confident_mean",
           "beta","alpha_raw_mean","mtime"])
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); w.writeheader()
        for r in ordered: w.writerow(r)
    print("wrote %d epoch(s) -> %s (new: %s)"%(len(ordered),out,sorted(new) or "none"))
    print("\nepoch  beta     spread   diff_raw   sp_img(over/conf)        raw_mean")
    for r in ordered:
        print("  %2d   %7s  %7.4f  %+8.4f   %s / %s     %s"%(
            int(r["epoch"]), r["beta"], float(r["spread"]), float(r["diff_raw"]),
            r["sp_overfired_mean"], r["sp_confident_mean"], r["alpha_raw_mean"]))
    print("\nF3 read: with beta frozen, the scale is locked -> spread/diff_raw should")
    print("grow in the SOFTPLUS columns now (sp_over should pull above sp_conf),")
    print("unlike F2 where the common-mode collapse crushed both. alpha_raw_mean is")
    print("a diagnostic only (re-centering at use makes its drift irrelevant).")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",required=True)
    p.add_argument("--out",default=None)
    p.add_argument("--force",action="store_true")
    ARGS=p.parse_args(); main()
