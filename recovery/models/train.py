"""Train the detector (ConvNeXt-V2, timm) and segmenter (SegFormer, transformers) on synthetic
(+ optionally real) layer images. Input = previous-layer frame + current frame + expected render.

  python -m recovery.models.train --data data/synth.npz --epochs 15 --pretrained
  python -m recovery.models.train --data data/synth.npz --init runs --real data/real.npz --epochs 8 --lr-det 1e-4 --lr-seg 3e-5

Real data: an .npz with the same arrays as synth_data (img, exp, tgt, label), built from your
own prints with induced failures (label them with tools/label_real.md workflow in README).
Reports detection accuracy / false-alarm rate and contour IoU on a held-out split.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .classical import LABELS
from .nets import (DEFAULT_BACKBONE, DEFAULT_SEG, N_FRAMES, build_detector, build_segmenter, device,
                   seg_logits)


def batches(n, bs, rng, shuffle=True):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def make_x(d, idx):
    """[previous frame, current frame, expected render] -> (B, 7, H, W)"""
    im = torch.from_numpy(d["img"][idx]).float().permute(0, 3, 1, 2) / 255.0
    pv = torch.from_numpy(d["prev"][idx]).float().permute(0, 3, 1, 2) / 255.0
    ex = torch.from_numpy(d["exp"][idx]).float()[:, None] / 255.0
    return torch.cat([pv, im, ex], 1)


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


def iou(logits, y):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * y).sum((1, 2, 3))
    union = ((p + y) > 0).float().sum((1, 2, 3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--real", default=None, help="optional real-image .npz for fine-tuning")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-det", type=float, default=3e-4)
    ap.add_argument("--lr-seg", type=float, default=1e-4)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, help="any timm model name")
    ap.add_argument("--seg-arch", default=DEFAULT_SEG, help="SegFormer checkpoint (nvidia/mit-b0..b5)")
    ap.add_argument("--pretrained", action="store_true", help="start from ImageNet / ADE20k weights")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--init", default=None, help="directory of checkpoints to fine-tune from")
    a = ap.parse_args(argv)
    dev, rng = device(), np.random.default_rng(0)
    d = dict(np.load(a.data))
    if a.real:
        r = dict(np.load(a.real))
        d = {k: np.concatenate([d[k], r[k]]) for k in d}
    n = len(d["label"])
    perm = rng.permutation(n)
    cut = int(0.85 * n)
    tr, va = perm[:cut], perm[cut:]
    Path(a.out).mkdir(exist_ok=True)

    clf = build_detector(a.backbone, N_FRAMES, a.pretrained).to(dev)
    seg = build_segmenter(a.seg_arch, N_FRAMES, a.pretrained).to(dev)
    if a.init:
        clf.load_state_dict(torch.load(Path(a.init) / "det.pt", map_location=dev))
        seg.load_state_dict(torch.load(Path(a.init) / "seg.pt", map_location=dev))
    opt_c = torch.optim.AdamW(clf.parameters(), a.lr_det, weight_decay=0.05)
    opt_s = torch.optim.AdamW(seg.parameters(), a.lr_seg, weight_decay=0.01)
    for ep in range(a.epochs):
        clf.train(), seg.train()
        tl = ts = 0.0
        for b in batches(len(tr), a.bs, rng):
            idx = tr[b]
            x = make_x(d, idx).to(dev)
            y = torch.from_numpy(d["label"][idx]).long().to(dev)
            m = (torch.from_numpy(d["tgt"][idx]).float()[:, None] / 255.0).to(dev)
            loss_c = Fn.cross_entropy(clf(x), y)
            opt_c.zero_grad(), loss_c.backward(), opt_c.step()
            lo = seg_logits(seg, x)
            loss_s = Fn.binary_cross_entropy_with_logits(lo, m) + dice_loss(lo, m)
            opt_s.zero_grad(), loss_s.backward(), opt_s.step()
            tl, ts = tl + loss_c.item(), ts + loss_s.item()
        clf.eval(), seg.eval()
        correct = fa = n_ok = 0
        ious = []
        with torch.no_grad():
            for b in batches(len(va), 32, rng, shuffle=False):
                idx = va[b]
                x = make_x(d, idx).to(dev)
                y = torch.from_numpy(d["label"][idx]).long().to(dev)
                pred = clf(x).argmax(1)
                correct += (pred == y).sum().item()
                n_ok += (y == 0).sum().item()
                fa += ((y == 0) & (pred != 0)).sum().item()
                m = (torch.from_numpy(d["tgt"][idx]).float()[:, None] / 255.0).to(dev)
                ious.append(iou(seg_logits(seg, x), m))
        print(f"epoch {ep+1}/{a.epochs} loss det {tl:.2f} seg {ts:.2f} | val acc {correct/len(va):.3f} "
              f"false-alarm {fa/max(n_ok,1):.3f} contour IoU {np.mean(ious):.3f}")
    torch.save(clf.state_dict(), Path(a.out) / "det.pt")
    torch.save(seg.state_dict(), Path(a.out) / "seg.pt")
    import json
    (Path(a.out) / "config.json").write_text(json.dumps(
        dict(backbone=a.backbone, seg_arch=a.seg_arch, n_frames=N_FRAMES)))
    print("saved", Path(a.out) / "det.pt", Path(a.out) / "seg.pt")


if __name__ == "__main__":
    main()
