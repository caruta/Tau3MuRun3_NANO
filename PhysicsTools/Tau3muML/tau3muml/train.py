#!/usr/bin/env python3
"""Train one fold of the tau -> 3mu transformer (command line or from the SWAN notebook).

Fold k: test = k, validation = (k+1) % 5, training = the other three. Train all five folds
to get an out-of-fold score for every candidate (evaluate.py).

Loss = weighted BCE + lambda * DisCo(score, mass) on the background (sideband) candidates of
each batch, which penalises a score that depends on the 3mu mass.

Checkpoints go to <out>/fold<k>/last.pt after every epoch (and best.pt by validation loss);
rerunning the same command resumes, so a SWAN session that disconnects loses at most one epoch.

  python3 -m tau3muml.train --data /eos/user/<u>/<you>/train_Transformer_v1 --out runs/v1 --fold 0
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from . import data as D
from .model import build

DEFAULTS = dict(epochs=20, batch_size=256, lr=5e-4, weight_decay=0.01, warmup_frac=0.03, disco=2.0, disco_on="score", disco_run=0.0,
                drop=None, drop_tokens=None,
                d=128, heads=8, layers=6, ff=4, dropout=0.1, pair_bias=True, amp=True,
                prod_frac=None, npv_reweight=True, val_rows=200_000, max_train_rows=None, seed=1,
                num_threads=None, grad_clip=1.0)


def distance_corr(x, y, w):
    """Weighted distance correlation (DisCo, arXiv:2001.05310); w normalised inside."""
    w = w / w.mean()
    a = (x[:, None] - x[None, :]).abs()
    b = (y[:, None] - y[None, :]).abs()

    def center(m):
        r = (m * w[None, :]).mean(1)
        return m - r[:, None] - r[None, :] + (r * w).mean()

    A, B = center(a), center(b)
    ww = w[:, None] * w[None, :]
    ab, aa, bb = (A * B * ww).mean(), (A * A * ww).mean(), (B * B * ww).mean()
    return ab / torch.sqrt(aa * bb + 1e-12)


def weighted_auc(score, label, w):
    o = np.argsort(-score)
    s, l, w = score[o], label[o], w[o]
    tp = np.cumsum(w * (l == 1))
    fp = np.cumsum(w * (l == 0))
    if tp[-1] <= 0 or fp[-1] <= 0:
        return float("nan")
    return float(np.trapezoid(tp / tp[-1], fp / fp[-1]))


def bkg_eff_at(score, label, w, sig_eff=0.5):
    s, b = label == 1, label == 0
    o = np.argsort(-score[s])
    c = np.cumsum(w[s][o]) / w[s].sum()
    cut = score[s][o][min(np.searchsorted(c, sig_eff), len(o) - 1)]
    return float(w[b][score[b] >= cut].sum() / max(w[b].sum(), 1e-12))


def device_of():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_loader(model, loader, device, amp=True):
    model.eval()
    scores, metas, ws = [], [], []
    for x, meta, w in loader:
        with torch.autocast(device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            s = torch.sigmoid(model(D.to_torch(x, device)).float())
        scores.append(s.cpu().numpy())
        metas.append(meta)
        ws.append(w)
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]} if metas else {}
    return (np.concatenate(scores) if scores else np.zeros(0)), meta, (np.concatenate(ws) if ws else np.zeros(0))


def train(data_dir, out_dir, fold, **cfg):
    c = dict(DEFAULTS)
    c.update({k: v for k, v in cfg.items() if v is not None})
    if c["num_threads"]:
        torch.set_num_threads(c["num_threads"])
    torch.manual_seed(c["seed"] + fold)
    device = device_of()
    run_dir = os.path.join(out_dir, f"fold{fold}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({**c, "data_dir": data_dir, "fold": fold}, fh, indent=1)

    stats = D.load_stats(data_dir)
    meta = D.load_meta(data_dir)
    tr_f, va_f, te_f = D.fold_split(fold)
    rows_tr = np.isin(meta["fold"], tr_f) & np.isin(meta["role"], [0, 1])
    prod = {int(k): float(v) for k, v in c["prod_frac"].items()} if c["prod_frac"] else None
    weighter = D.Weighter(meta, rows_tr, prod_frac=prod, npv_reweight=c["npv_reweight"])
    n_tr = int(rows_tr.sum()) if not c["max_train_rows"] else min(int(rows_tr.sum()), c["max_train_rows"])
    print(f"fold {fold}: train folds {tr_f} ({n_tr} rows: {int((rows_tr & (meta['role'] == 0)).sum())} signal), "
          f"val fold {va_f}, test fold {te_f}; device {device}")

    model = build(stats, d=c["d"], heads=c["heads"], layers=c["layers"], ff=c["ff"], dropout=c["dropout"],
                  pair_bias=c["pair_bias"], drop=c["drop"] or (), drop_tokens=c["drop_tokens"] or ()).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "type_embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "type_embed" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=c["lr"])
    steps_per_epoch = max(1, n_tr // c["batch_size"])
    total = steps_per_epoch * c["epochs"]
    warm = max(1, int(c["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_amp = c["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, float("inf")
    last = os.path.join(run_dir, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']} (best val loss {best:.4f})")
    if start_epoch >= c["epochs"]:
        print("training already finished")
        return run_dir

    val_loader = D.ShardLoader(data_dir, va_f, roles=(0, 1), batch_size=1024, shuffle=False, weighter=weighter,
                               max_rows=c["val_rows"])
    metrics_path = os.path.join(run_dir, "metrics.csv")
    new_metrics = not os.path.exists(metrics_path)
    for epoch in range(start_epoch, c["epochs"]):
        model.train()
        loader = D.ShardLoader(data_dir, tr_f, roles=(0, 1), batch_size=c["batch_size"], shuffle=True,
                               weighter=weighter, max_rows=c["max_train_rows"], seed=c["seed"] * 1000 + epoch,
                               drop_last=True)
        t0, sums, nb = time.time(), np.zeros(3), 0
        for x, meta_b, w in loader:
            xb = D.to_torch(x, device)
            y = torch.from_numpy(meta_b["label"].astype(np.float32)).to(device)
            wt = torch.from_numpy(w).to(device)
            mass = torch.from_numpy(meta_b["mass"].astype(np.float32)).to(device)
            runv = torch.from_numpy(meta_b["run"].astype(np.float32)).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logit = model(xb)
            logit = logit.float()
            bce = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * wt).sum() / wt.sum()
            bk = (y == 0) & (wt > 0)
            dvar = logit[bk] if c["disco_on"] == "logit" else torch.sigmoid(logit[bk])
            dc = distance_corr(dvar, mass[bk], wt[bk]) if bk.sum() > 8 else torch.zeros((), device=device)
            loss = bce + c["disco"] * dc
            if c["disco_run"] > 0 and bk.sum() > 8:
                # background score independent of the data-taking period (detector conditions change during the year)
                loss = loss + c["disco_run"] * distance_corr(dvar, runv[bk], wt[bk])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [loss.item(), bce.item(), dc.item()]
            nb += 1
            if nb % 200 == 0:
                print(f"  epoch {epoch} batch {nb}/{steps_per_epoch}: loss {sums[0] / nb:.4f} "
                      f"(bce {sums[1] / nb:.4f}, disco {sums[2] / nb:.4f}), {time.time() - t0:.0f} s", flush=True)
        tr = sums / max(nb, 1)

        s, m, w = predict_loader(model, val_loader, device, c["amp"])
        y = m["label"]
        eps = 1e-6
        vbce = float(-(w * (y * np.log(s + eps) + (1 - y) * np.log(1 - s + eps))).sum() / w.sum())
        b = y == 0
        nb_dc = min(int(b.sum()), 5000)
        sel = np.nonzero(b)[0][:nb_dc]
        sv = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7))) if c["disco_on"] == "logit" else s
        vdc = float(distance_corr(torch.from_numpy(sv[sel]).double(), torch.from_numpy(m["mass"][sel]).double(),
                                  torch.from_numpy(w[sel]).double())) if nb_dc > 8 else float("nan")
        # sculpting: (1.70-1.74, 1.82-1.86) / (1.50-1.62, 1.94-2.06) GeV for the top 1 % sidebands vs all
        mb = m["mass"]
        near = b & (((mb >= 1.70) & (mb < 1.74)) | ((mb >= 1.82) & (mb < 1.86)))
        far = b & (((mb >= 1.50) & (mb < 1.62)) | ((mb >= 1.94) & (mb < 2.06)))
        top = s >= np.quantile(s[b], 0.99) if b.any() else np.zeros_like(b)
        v_nf = ((near & top).sum() / max((far & top).sum(), 1)) / max(near.sum() / max(far.sum(), 1), 1e-9)
        early, late = b & (m["run"] < 383000), b & (m["run"] >= 383000)
        v_el = ((top & early).sum() / early.sum()) / max((top & late).sum() / late.sum(), 1e-9) \
            if early.any() and late.any() else float("nan")
        vloss = vbce + c["disco"] * vdc
        row = dict(epoch=epoch, lr=sched.get_last_lr()[0], train_loss=tr[0], train_bce=tr[1], train_disco=tr[2],
                   val_loss=vloss, val_bce=vbce, val_disco=vdc, val_auc=weighted_auc(s, y, w),
                   val_bkgeff_sig50=bkg_eff_at(s, y, w, 0.5), val_bkgeff_sig20=bkg_eff_at(s, y, w, 0.2), val_near_far=float(v_nf), val_early_late=float(v_el),
                   time_s=time.time() - t0)
        print("  " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        with open(metrics_path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new_metrics:
                wr.writeheader()
                new_metrics = False
            wr.writerow(row)
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "config": model.config, "stats": stats, "epoch": epoch,
                        "val_loss": vloss}, os.path.join(run_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "config": model.config,
                    "stats": stats}, last + ".tmp")
        os.replace(last + ".tmp", last)
    return run_dir


def load_model(path, device=None):
    device = device or device_of()
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build(ck["stats"], **ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training set directory (make_training_set.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, nargs="+", default=[0])
    for k, v in DEFAULTS.items():
        if k in ("drop", "drop_tokens"):
            ap.add_argument(f"--{k.replace('_', '-')}", nargs="+", default=None)
        elif k == "disco_on":
            ap.add_argument("--disco-on", choices=["score", "logit"], default=v)
        elif k == "prod_frac":
            ap.add_argument("--prod-frac", default=None,
                            help='JSON, e.g. \'{"1": 0.7, "2": 0.1, "3": 0.2}\' (1 Ds, 2 B->Ds, 3 B->tau, 4 D+, 5 W)')
        elif isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if v is not None else int, default=v)
    a = vars(ap.parse_args())
    if a["prod_frac"]:
        a["prod_frac"] = json.loads(a["prod_frac"])
    data, out, folds = a.pop("data"), a.pop("out"), a.pop("fold")
    for f in folds:
        train(data, out, f, **a)


if __name__ == "__main__":
    main()
