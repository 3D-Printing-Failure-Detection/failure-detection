"""Dent / poke detector for the P1S chamber camera (the real, near-horizontal fisheye view).

Same encoders as the layer models in nets.py, on the same kind of input: a short temporal stack plus a
guide channel, 3 + 3 + 1 = 7 channels at 256 x 256.
  - frames: BEFORE (at the pause) and AFTER (after the person touched the print), as RGB;
  - guide: where the part should be in the image. It is projected from the calibration and the part's
    bed position, just as the layer models get the expected layer.
The frames are cropped to a square around the guide, because the tray is only 150 to 300 px wide in a
1280 x 720 frame and a poke changes about 40 px. A downscaled full frame would lose it.

  Detector (ConvNeXt-V2, timm)       -> kind (none / dent / poke) and wall (front / back / left / right)
  Segmenter (SegFormer / MiT)        -> mask of the surface that changed

Build crops from recovery/p1s_dent_data.py output, train, evaluate:
  python -m recovery.models.dent build --data data/p1s_dent_v1 --out data/p1s_dent_v1/crops.npz
  python -m recovery.models.dent train --crops data/p1s_dent_v1/crops.npz --out runs/dent --pretrained
  python -m recovery.models.dent eval  --crops data/p1s_dent_test_s1/crops.npz --ckpt runs/dent

At inference (`DentDetector.predict`) only the base calibration and the part's place on the bed are
used: the per-sample camera jitter of the synthetic data plays the role of calibration error.
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
from ..p1s_dent_data import KINDS, WALLS, _corners_world
from .nets import build_segmenter, device, seg_logits

SIZE = 256
STORE = 288                 # crops are stored a little larger and randomly cropped to SIZE in training
MARGIN = 0.35               # crop = guide bbox grown by this fraction on each side, made square
N_OUT = len(KINDS) + len(WALLS)
TRAY_BOUNDS = (-15.0, -10.0, 15.0, 10.0)


# ---------------------------------------------------------------- input
def guide_polygon(cam, part_xy, yaw_deg, top_z, bounds=TRAY_BOUNDS):
    """Image outline (convex hull, px) of the part's bounding box, plate to top, with the top at the
    nozzle plane (the P1S bed moves down, the camera does not)."""
    plate_z = V.plate_z_mm(cam, top_z)
    px = cam.project(_corners_world(bounds, part_xy, yaw_deg, plate_z, plate_z + top_z))
    return cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)


def crop_box(poly, size_px=STORE / SIZE, margin=MARGIN):
    """Square crop (x0, y0, side) around the guide polygon."""
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin) * size_px
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - side / 2, cy - side / 2, side


def crop(img, box, out, interp=cv2.INTER_AREA):
    """Crop a square (x0, y0, side) of `img` to out x out, padding with edge pixels outside the frame."""
    x0, y0, side = box
    s = out / side
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(img, M, (out, out), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def guide_mask(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return m


def make_input(before, after, poly, out=STORE):
    """RGB before/after frames + guide polygon -> (out, out, 7) uint8 and the crop box."""
    box = crop_box(poly, out / SIZE)
    g = guide_mask(poly, before.shape)
    x = np.concatenate([crop(before, box, out), crop(after, box, out), crop(g, box, out)[..., None]], -1)
    return x, box


# ---------------------------------------------------------------- dataset
def build(data_dir, out_path, base_camera=None):
    """Crops for every pair in a p1s_dent_data output directory -> .npz (x, mask, kind, wall, meta)."""
    data_dir = Path(data_dir)
    man = json.loads((data_dir / "manifest.json").read_text())
    cam = base_camera or _camera_from_json(man["base_camera"])
    labels = [json.loads(l) for l in open(data_dir / "labels.jsonl")]
    X = np.zeros((len(labels), STORE, STORE, 7), np.uint8)
    Y = np.zeros((len(labels), STORE, STORE), np.uint8)
    kind = np.array([KINDS.index(l["kind"]) for l in labels], np.int64)
    wall = np.array([WALLS.index(l["wall"]) if l["wall"] else -1 for l in labels], np.int64)
    for i, l in enumerate(labels):
        b = cv2.imread(str(data_dir / l["files"]["before"]))[..., ::-1]
        a = cv2.imread(str(data_dir / l["files"]["after"]))[..., ::-1]
        m = cv2.imread(str(data_dir / l["files"]["mask"]), cv2.IMREAD_GRAYSCALE)
        poly = guide_polygon(cam, l["part_xy_mm"], l["part_yaw_deg"], l["top_z_mm"])
        X[i], box = make_input(b, a, poly)
        Y[i] = (crop(m, box, STORE, cv2.INTER_LINEAR) > 64).astype(np.uint8) * 255
    meta = dict(ids=[l["id"] for l in labels], split=[l["split"] for l in labels],
                change_px=[l["change_px"] for l in labels], visible=[l["visible"] for l in labels],
                facing=[bool(l.get("outer_face_toward_camera")) for l in labels])
    np.savez(out_path, x=X, mask=Y, kind=kind, wall=wall, meta=json.dumps(meta))
    return len(labels)


def _camera_from_json(d):
    return V.P1SCamera(np.array(d["K"], float), np.array(d["dist"], float), np.array(d["rvec"], float).reshape(3, 1),
                       np.array(d["tvec"], float).reshape(3, 1), float(d["calib_z_mm"]))


def load_crops(path):
    d = dict(np.load(path))
    d["meta"] = json.loads(str(d["meta"]))
    return d


def to_tensor(x, mask=None, train=False, rng=None):
    """(B, STORE, STORE, 7) uint8 -> (B, 7, SIZE, SIZE) float; random crop in training, centre crop otherwise."""
    B = len(x)
    if train:
        offs = rng.integers(0, STORE - SIZE + 1, (B, 2))
    else:
        offs = np.full((B, 2), (STORE - SIZE) // 2)
    xs = np.stack([x[i, oy:oy + SIZE, ox:ox + SIZE] for i, (ox, oy) in enumerate(offs)])
    t = torch.from_numpy(xs).permute(0, 3, 1, 2).float() / 255.0
    if mask is None:
        return t
    ms = np.stack([mask[i, oy:oy + SIZE, ox:ox + SIZE] for i, (ox, oy) in enumerate(offs)])
    return t, torch.from_numpy(ms).float()[:, None] / 255.0


# ---------------------------------------------------------------- model
# One SegFormer, dense outputs per pixel: [changed] + [dent, poke] + [front, back, left, right].
# A whole-image classifier (ConvNeXt-V2 with kind/wall heads) was tried first and stayed at chance on
# 2700 pairs (it overfits 64 pairs, so it trains; it just cannot find a 40 to 400 px change among
# everything else that varies). Telling the network WHERE the change is, per pixel, is what works.
N_SEG = 1 + 2 + len(WALLS)
TOUCH = ("dent", "poke")


def build_model(seg_arch="nvidia/mit-b2", pretrained=False, config=None):
    return build_segmenter(seg_arch, n_frames=2, pretrained=pretrained, config=config, num_labels=N_SEG)


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


def model_loss(lo, m, kind, wall, pos_weight=5.0):
    """lo (B, N_SEG, S, S) logits, m (B, 1, S, S) changed mask, kind/wall per sample (KINDS / WALLS index)."""
    ch = lo[:, :1]
    loss = Fn.binary_cross_entropy_with_logits(ch, m, pos_weight=torch.tensor(pos_weight, device=lo.device)) + dice_loss(ch, m)
    on = m[:, 0] > 0.5                                         # supervise kind / wall on changed pixels only
    if on.any():
        kt = torch.tensor([TOUCH.index(KINDS[k]) if KINDS[k] in TOUCH else 0 for k in kind.tolist()], device=lo.device)
        k_pix = kt[:, None, None].expand_as(on)[on]
        w_pix = wall[:, None, None].expand_as(on)[on].clamp(min=0)
        loss = loss + Fn.cross_entropy(lo[:, 1:3].permute(0, 2, 3, 1)[on], k_pix)
        loss = loss + 0.5 * Fn.cross_entropy(lo[:, 3:].permute(0, 2, 3, 1)[on], w_pix)
    return loss


@torch.no_grad()
def raw_outputs(model, x, dev, bs=64):
    """Per sample: changed-pixel count, p(dent) and wall probabilities averaged over the predicted change
    (weighted by the change probability), and the change mask."""
    model.eval()
    area, pk, pw, masks = [], [], [], []
    for i in range(0, len(x), bs):
        t = to_tensor(x[i:i + bs]).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            lo = seg_logits(model, t).float()
        p = torch.sigmoid(lo[:, 0])
        m = p > 0.5
        wgt = p * m + 1e-6 * p                                 # mostly the predicted change; fallback: all pixels
        wgt = wgt / wgt.sum((1, 2), keepdim=True)
        pk.append((torch.softmax(lo[:, 1:3], 1) * wgt[:, None]).sum((2, 3)).cpu().numpy())
        pw.append((torch.softmax(lo[:, 3:], 1) * wgt[:, None]).sum((2, 3)).cpu().numpy())
        area.append(m.sum((1, 2)).cpu().numpy())
        masks.append(m.cpu().numpy())
    return np.concatenate(area), np.concatenate(pk), np.concatenate(pw), np.concatenate(masks)


def decide(area, pk, min_area):
    """Image-level kind index (KINDS) from the change area and the dent/poke split."""
    out = np.where(pk[:, 0] >= pk[:, 1], KINDS.index("dent"), KINDS.index("poke"))
    return np.where(area >= min_area, out, KINDS.index("none"))


def tune_min_area(d, idx, area, pk, grid=(1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 60)):
    """Smallest change (px in the 256 crop) that counts as touched: the best kind accuracy on these samples."""
    acc = [np.mean(decide(area, pk, a) == d["kind"][idx]) for a in grid]
    return int(grid[int(np.argmax(acc))])


# ---------------------------------------------------------------- evaluation
def metrics(d, idx, k_pred, wall_p, mask_p):
    """Success rates on samples idx (predictions are aligned with idx)."""
    k_true, w_true = d["kind"][idx], d["wall"][idx]
    off = (STORE - SIZE) // 2
    m_true = d["mask"][idx][:, off:off + SIZE, off:off + SIZE] > 0
    vis = np.array(d["meta"]["visible"])[idx]
    face = np.array(d["meta"]["facing"])[idx]
    none, dent, poke = KINDS.index("none"), KINDS.index("dent"), KINDS.index("poke")
    changed_t, changed_p = k_true != none, k_pred != none
    conf = np.zeros((3, 3), int)
    for t, p in zip(k_true, k_pred):
        conf[t, p] += 1

    def rate(a):
        return round(float(np.mean(a)), 4) if len(a) else None

    ious = [(t & p).sum() / max((t | p).sum(), 1) for t, p in zip(m_true[changed_t], mask_p[changed_t])]
    walls_ok = wall_p.argmax(1)[changed_t] == w_true[changed_t]
    return dict(
        n=int(len(idx)),
        kind_accuracy=rate(k_pred == k_true),                          # dent / poke / none all correct
        touched_vs_untouched_accuracy=rate(changed_p == changed_t),
        touched_detected=rate(changed_p[changed_t]),                   # dent or poke flagged as either
        false_alarm_rate=rate(changed_p[k_true == none]),              # untouched flagged as touched
        dent_recall=rate(k_pred[k_true == dent] == dent),
        poke_recall=rate(k_pred[k_true == poke] == poke),
        poke_recall_visible=rate(k_pred[(k_true == poke) & vis] == poke),
        dent_precision=rate(k_true[k_pred == dent] == dent),
        poke_precision=rate(k_true[k_pred == poke] == poke),
        wall_accuracy=rate(walls_ok),
        wall_accuracy_facing_camera=rate(walls_ok[face[changed_t]]),
        wall_accuracy_not_facing=rate(walls_ok[~face[changed_t]]),
        change_mask_iou=rate(ious),
        confusion_true_rows_pred_cols=dict(order=list(KINDS), matrix=conf.tolist()),
    )


# ---------------------------------------------------------------- training
def train(crops, out_dir, epochs=30, bs=32, lr=2e-4, seg_arch="nvidia/mit-b2", pretrained=False, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    d = load_crops(crops)
    split = np.array(d["meta"]["split"])
    tr, va = np.nonzero(split == "train")[0], np.nonzero(split == "val")[0]
    model = build_model(seg_arch, pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    kind_t, wall_t = torch.from_numpy(d["kind"]), torch.from_numpy(d["wall"])
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
            loss = model_loss(lo, m, kind_t[idx].to(dev), wall_t[idx].to(dev))
            opt.zero_grad(), loss.backward(), opt.step(), sch.step()
            tl += loss.item() * len(idx)
        area, pk, pw, mk = raw_outputs(model, d["x"][va], dev)
        min_area = tune_min_area(d, va, area, pk)
        r = metrics(d, va, decide(area, pk, min_area), pw, mk)
        history.append(dict(epoch=ep + 1, loss=tl / len(tr), min_area=min_area, **{
            k: r[k] for k in ("kind_accuracy", "false_alarm_rate", "dent_recall", "poke_recall", "wall_accuracy",
                              "change_mask_iou")}))
        log(f"epoch {ep + 1}/{epochs} loss {tl / len(tr):.3f} | val kind acc {r['kind_accuracy']:.3f} false-alarm "
            f"{r['false_alarm_rate']:.3f} dent {r['dent_recall']:.3f} poke {r['poke_recall']:.3f} wall "
            f"{r['wall_accuracy']:.3f} IoU {r['change_mask_iou']:.3f} (min area {min_area}) ({time.time() - t0:.0f}s)")
        score = r["kind_accuracy"] + 0.25 * r["wall_accuracy"] + 0.25 * r["change_mask_iou"]
        if score > best:
            best = score
            torch.save(model.state_dict(), out_dir / "model.pt")
            best_val = dict(epoch=ep + 1, min_area=min_area, **r)
    (out_dir / "config.json").write_text(json.dumps(dict(
        seg_arch=seg_arch, segmenter_config=model.config.to_dict(), size=SIZE, store=STORE, margin=MARGIN,
        kinds=list(KINDS), walls=list(WALLS), min_area=best_val["min_area"], crops=str(crops), epochs=epochs,
        bs=bs, lr=lr, pretrained=pretrained, seed=seed)))
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
    idx = np.arange(len(d["kind"]))
    area, pk, pw, mk = raw_outputs(model, d["x"], dev)
    r = metrics(d, idx, decide(area, pk, cfg["min_area"]), pw, mk)
    # by how visible the change is (changed pixels in the full frame)
    px = np.array(d["meta"]["change_px"])
    k_pred = decide(area, pk, cfg["min_area"])
    r["touched_detected_by_change_px"] = {
        name: round(float(np.mean(k_pred[sel] != KINDS.index("none"))), 4) if sel.any() else None
        for name, sel in (("<15", (px < 15) & (d["kind"] != KINDS.index("none"))), ("15-50", (px >= 15) & (px < 50)),
                          ("50-200", (px >= 50) & (px < 200)), (">=200", px >= 200))}
    return r


class DentDetector:
    """Real-time use: before/after frames from the P1S camera at a pause.

        det = DentDetector("runs/dent", P1SCamera.load("calib.npz"))
        r = det.predict(before_rgb, after_rgb, part_xy=(128, 128), yaw_deg=0, top_z=6.0)
        r["kind"], r["wall"], r["prob"], r["mask"]      # mask in full-frame pixels
    """

    def __init__(self, ckpt_dir, cam: V.P1SCamera = None, dev=None):
        self.model, self.cfg, self.dev = load_model(ckpt_dir, dev)
        self.cam = cam or V.P1SCamera.default()

    def predict(self, before, after, part_xy, yaw_deg, top_z, bounds=TRAY_BOUNDS):
        poly = guide_polygon(self.cam, part_xy, yaw_deg, top_z, bounds)
        x, box = make_input(before, after, poly)
        area, pk, pw, mk = raw_outputs(self.model, x[None], self.dev)
        kind = KINDS[int(decide(area, pk, self.cfg["min_area"])[0])]
        off = (STORE - SIZE) // 2
        m = np.zeros((STORE, STORE), np.uint8)
        m[off:off + SIZE, off:off + SIZE] = mk[0].astype(np.uint8) * 255
        x0, y0, side = box
        s = side / STORE
        full = cv2.warpAffine(m, np.array([[s, 0, x0], [0, s, y0]], np.float32), (before.shape[1], before.shape[0]),
                              flags=cv2.INTER_NEAREST)
        return dict(kind=kind, change_px_crop=int(area[0]), p_dent_vs_poke=pk[0].round(4).tolist(),
                    wall=WALLS[int(pw[0].argmax())] if kind != "none" else None, mask=full)


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1S dent / poke detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--crops", required=True)
    t.add_argument("--out", default="runs/dent")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--seg-arch", default="nvidia/mit-b2")
    t.add_argument("--pretrained", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--crops", required=True)
    e.add_argument("--ckpt", default="runs/dent")
    e.add_argument("--out", help="write the metrics JSON here")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(f"{build(a.data, a.out)} pairs -> {a.out}")
    elif a.cmd == "train":
        r = train(a.crops, a.out, a.epochs, a.bs, a.lr, a.seg_arch, a.pretrained, a.seed,
                  log=lambda s: print(s, flush=True))
        print(json.dumps(r, indent=1))
    else:
        r = evaluate(a.crops, a.ckpt)
        print(json.dumps(r, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
