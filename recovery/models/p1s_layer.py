"""Print-failure detector for the P1S chamber camera, every layer, in the camera's own view.

Input (7 x 256 x 256, like the dent model and nets.py): the frame after the previous layer, the frame
after this layer (RGB each) and the plan's outline so far, rendered from the calibrated camera. The plan
outline replaces the top-down "expected layer" of the old pipeline: the P1S camera has no top-down view.
The frames are cropped to a square around that outline.

One SegFormer (MiT-b2) marks, per pixel, where the print departs from the plan and which failure it
looks like (collapse, warp, shift, under_extrusion, detach). The image-level answer comes from that
region: "ok" when it is smaller than a threshold tuned on the val split.

  python -m recovery.models.p1s_layer build --data data/p1s_layer_v1 --out data/p1s_layer_v1/crops.npz
  python -m recovery.models.p1s_layer train --crops data/p1s_layer_v1/crops.npz --out runs/p1s_layer --pretrained
  python -m recovery.models.p1s_layer eval  --crops data/p1s_layer_test_s1/crops.npz --ckpt runs/p1s_layer
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

from .. import p1s_view as V
from ..p1s_layer_data import FAILURES, KINDS, plan_silhouette
from .dent import MARGIN, SIZE, STORE, crop, dice_loss, to_tensor
from .nets import build_segmenter, device, seg_logits

N_SEG = 1 + len(FAILURES)


# ---------------------------------------------------------------- input
def crop_box_from_guide(guide, margin=MARGIN):
    ys, xs = np.nonzero(guide)
    if not len(xs):
        raise ValueError("the plan's outline is not in the image: wrong calibration or part position")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(x1 - x0, y1 - y0, 24) * (1 + 2 * margin) * STORE / SIZE
    return (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side


def make_input(prev, cur, guide):
    box = crop_box_from_guide(guide)
    x = np.concatenate([crop(prev, box, STORE), crop(cur, box, STORE), crop(guide, box, STORE)[..., None]], -1)
    return x, box


def build(data_dir, out_path):
    data_dir = Path(data_dir)
    labels = [json.loads(l) for l in open(data_dir / "labels.jsonl")]
    X = np.zeros((len(labels), STORE, STORE, 7), np.uint8)
    Y = np.zeros((len(labels), STORE, STORE), np.uint8)
    for i, l in enumerate(labels):
        f = l["files"]
        prev = cv2.imread(str(data_dir / f["prev"]))[..., ::-1]
        cur = cv2.imread(str(data_dir / f["cur"]))[..., ::-1]
        guide = cv2.imread(str(data_dir / f["guide"]), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(data_dir / f["mask"]), cv2.IMREAD_GRAYSCALE)
        X[i], box = make_input(prev, cur, guide)
        Y[i] = (crop(m, box, STORE, cv2.INTER_LINEAR) > 64).astype(np.uint8) * 255
    meta = dict(ids=[l["id"] for l in labels], split=[l["split"] for l in labels], part=[l["part"] for l in labels],
                change_px=[l["change_px"] for l in labels], visible=[l["visible"] for l in labels],
                since=[None if l["failure_started_layer"] is None else l["layer"] - l["failure_started_layer"]
                       for l in labels])
    np.savez(out_path, x=X, mask=Y, kind=np.array([KINDS.index(l["kind"]) for l in labels], np.int64),
             meta=json.dumps(meta))
    return len(labels)


def load_crops(path):
    d = dict(np.load(path))
    d["meta"] = json.loads(str(d["meta"]))
    return d


# ---------------------------------------------------------------- model
def build_model(seg_arch="nvidia/mit-b2", pretrained=False, config=None):
    return build_segmenter(seg_arch, n_frames=2, pretrained=pretrained, config=config, num_labels=N_SEG)


def model_loss(lo, m, kind, pos_weight=5.0):
    ch = lo[:, :1]
    loss = Fn.binary_cross_entropy_with_logits(ch, m, pos_weight=torch.tensor(pos_weight, device=lo.device)) + dice_loss(ch, m)
    on = (m[:, 0] > 0.5) & (kind[:, None, None] > 0)            # failure type: supervised where it shows
    if on.any():
        k_pix = (kind - 1).clamp(min=0)[:, None, None].expand_as(on)[on]
        loss = loss + Fn.cross_entropy(lo[:, 1:].permute(0, 2, 3, 1)[on], k_pix)
    return loss


@torch.no_grad()
def raw_outputs(model, x, dev, bs=64):
    model.eval()
    area, pk, masks = [], [], []
    for i in range(0, len(x), bs):
        t = to_tensor(x[i:i + bs]).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            lo = seg_logits(model, t).float()
        p = torch.sigmoid(lo[:, 0])
        m = p > 0.5
        wgt = p * m + 1e-6 * p
        wgt = wgt / wgt.sum((1, 2), keepdim=True)
        pk.append((torch.softmax(lo[:, 1:], 1) * wgt[:, None]).sum((2, 3)).cpu().numpy())
        area.append(m.sum((1, 2)).cpu().numpy())
        masks.append(m.cpu().numpy())
    return np.concatenate(area), np.concatenate(pk), np.concatenate(masks)


def decide(area, pk, min_area):
    return np.where(area >= min_area, pk.argmax(1) + 1, 0)


def tune_min_area(kind_true, area, pk, grid=(2, 3, 5, 8, 12, 16, 20, 30, 40, 60, 80, 120)):
    """Smallest flagged region (px in the 256 crop) that counts as a failure. Picked for the best accuracy
    among thresholds that keep false alarms on good prints at or below 2 %."""
    ok = kind_true == 0
    best, best_acc = grid[-1], -1.0
    for a in grid:
        pred = decide(area, pk, a)
        fa = np.mean(pred[ok] != 0) if ok.any() else 0.0
        acc = np.mean(pred == kind_true)
        if fa <= 0.02 and acc > best_acc:
            best, best_acc = a, acc
    return int(best)


def metrics(d, idx, pred, mask_p):
    kt = d["kind"][idx]
    off = (STORE - SIZE) // 2
    m_true = d["mask"][idx][:, off:off + SIZE, off:off + SIZE] > 0
    vis = np.array(d["meta"]["visible"])[idx]
    parts = np.array(d["meta"]["part"])[idx]
    since = np.array([-1 if s is None else s for s in np.array(d["meta"]["since"], dtype=object)[idx]])
    fail_t, fail_p = kt != 0, pred != 0

    def rate(a):
        return round(float(np.mean(a)), 4) if len(a) else None

    conf = np.zeros((len(KINDS), len(KINDS)), int)
    for t, p in zip(kt, pred):
        conf[t, p] += 1
    ious = [(t & p).sum() / max((t | p).sum(), 1) for t, p in zip(m_true[fail_t], mask_p[fail_t])]
    out = dict(
        n=int(len(idx)),
        exact_accuracy=rate(pred == kt),                       # ok / which failure, all correct
        failure_caught=rate(fail_p[fail_t]),                    # a failing print flagged as failing (any type)
        failure_caught_visible=rate(fail_p[fail_t & vis]),
        false_alarm_rate=rate(fail_p[~fail_t]),                 # a good print flagged
        failure_type_correct_when_caught=rate(pred[fail_t & fail_p] == kt[fail_t & fail_p]),
        caught_by_type={k: rate(fail_p[kt == i]) for i, k in enumerate(KINDS) if i > 0},
        caught_by_part={p: rate(fail_p[fail_t & (parts == p)]) for p in sorted(set(parts))},
        caught_by_layers_since_start={b: rate(fail_p[fail_t & (since >= lo) & (since <= hi)])
                                      for b, (lo, hi) in (("1-3", (1, 3)), ("4-7", (4, 7)), ("8+", (8, 999)))},
        mask_iou=rate(ious),
        confusion=dict(order=list(KINDS), rows_true_cols_pred=conf.tolist()),
    )
    return out


def train(crops, out_dir, epochs=30, bs=32, lr=2e-4, seg_arch="nvidia/mit-b2", pretrained=False, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    d = load_crops(crops)
    split = np.array(d["meta"]["split"])
    tr, va = np.nonzero(split == "train")[0], np.nonzero(split == "val")[0]
    model = build_model(seg_arch, pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * int(np.ceil(len(tr) / bs)), pct_start=0.1)
    kind_t = torch.from_numpy(d["kind"])
    best, history, out_dir = -1.0, [], Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl = 0.0
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, m = to_tensor(d["x"][idx], d["mask"][idx], train=True, rng=rng)
            x, m = x.to(dev), m.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                lo = seg_logits(model, x).float()
            loss = model_loss(lo, m, kind_t[idx].to(dev))
            opt.zero_grad(), loss.backward(), opt.step(), sch.step()
            tl += loss.item() * len(idx)
        area, pk, mk = raw_outputs(model, d["x"][va], dev)
        min_area = tune_min_area(d["kind"][va], area, pk)
        r = metrics(d, va, decide(area, pk, min_area), mk)
        history.append(dict(epoch=ep + 1, loss=tl / len(tr), min_area=min_area, **{
            k: r[k] for k in ("exact_accuracy", "failure_caught", "false_alarm_rate", "mask_iou")}))
        log(f"epoch {ep + 1}/{epochs} loss {tl / len(tr):.3f} | val exact {r['exact_accuracy']:.3f} caught "
            f"{r['failure_caught']:.3f} false-alarm {r['false_alarm_rate']:.3f} type-ok "
            f"{r['failure_type_correct_when_caught'] or 0:.3f} IoU {r['mask_iou']:.3f} (min area {min_area}) "
            f"({time.time() - t0:.0f}s)")
        score = r["exact_accuracy"] + r["failure_caught"] - 2 * r["false_alarm_rate"]
        if score > best:
            best = score
            torch.save(model.state_dict(), out_dir / "model.pt")
            best_val = dict(epoch=ep + 1, min_area=min_area, **r)
    (out_dir / "config.json").write_text(json.dumps(dict(
        seg_arch=seg_arch, segmenter_config=model.config.to_dict(), size=SIZE, store=STORE, margin=MARGIN,
        kinds=list(KINDS), min_area=best_val["min_area"], crops=str(crops), epochs=epochs, bs=bs, lr=lr,
        pretrained=pretrained, seed=seed)))
    (out_dir / "val.json").write_text(json.dumps(dict(best=best_val, history=history), indent=1))
    return best_val


def load_model(ckpt_dir, dev=None):
    cfg = json.loads((Path(ckpt_dir) / "config.json").read_text())
    dev = dev or device()
    model = build_model(cfg["seg_arch"], config=cfg["segmenter_config"])
    model.load_state_dict(torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu"))
    return model.to(dev).eval(), cfg, dev


def evaluate(crops, ckpt_dir):
    model, cfg, dev = load_model(ckpt_dir)
    d = load_crops(crops)
    area, pk, mk = raw_outputs(model, d["x"], dev)
    return metrics(d, np.arange(len(d["kind"])), decide(area, pk, cfg["min_area"]), mk)


class LayerWatcher:
    """Every layer on the printer: compare the new frame with the previous one and with the plan.

        w = LayerWatcher("runs/p1s_layer", P1SCamera.load("calib.npz"))
        r = w.check(prev_rgb, cur_rgb, plan_layers_so_far, part_xy=(128, 128))
        r["kind"]           # "ok" or the failure it looks like
    """

    def __init__(self, ckpt_dir, cam: V.P1SCamera = None, dev=None):
        self.model, self.cfg, self.dev = load_model(ckpt_dir, dev)
        self.cam = cam or V.P1SCamera.default()

    def check(self, prev, cur, plan, part_xy, yaw_deg=0.0):
        guide = plan_silhouette(self.cam, V.Look(), plan, part_xy, yaw_deg)
        x, box = make_input(prev, cur, guide)
        area, pk, mk = raw_outputs(self.model, x[None], self.dev)
        kind = KINDS[int(decide(area, pk, self.cfg["min_area"])[0])]
        return dict(kind=kind, flagged_px=int(area[0]), min_px=int(self.cfg["min_area"]),
                    type_prob=dict(zip(FAILURES, pk[0].round(4).tolist())), mask_crop=mk[0], box=box, guide=guide)

    @staticmethod
    def flagged_points(r):
        """Full-frame (x, y) pixel coordinates of what check() flagged."""
        x0, y0, side = r["box"]
        v, u = np.nonzero(r["mask_crop"])
        off = (STORE - SIZE) / 2
        k = side / STORE
        return np.stack([x0 + (u + off + 0.5) * k, y0 + (v + off + 0.5) * k], 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1S per-layer failure detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--crops", required=True)
    t.add_argument("--out", default="runs/p1s_layer")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--seg-arch", default="nvidia/mit-b2")
    t.add_argument("--pretrained", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--crops", required=True)
    e.add_argument("--ckpt", default="runs/p1s_layer")
    e.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(f"{build(a.data, a.out)} samples -> {a.out}")
    elif a.cmd == "train":
        print(json.dumps(train(a.crops, a.out, a.epochs, a.bs, a.lr, a.seg_arch, a.pretrained, a.seed,
                               log=lambda s: print(s, flush=True)), indent=1))
    else:
        r = evaluate(a.crops, a.ckpt)
        print(json.dumps(r, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
