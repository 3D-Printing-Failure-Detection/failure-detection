"""Synthetic training data: (camera image, expected-layer mask) -> (label, true mask).

Every sample slices a random part, optionally injects a failure that began up to 10 layers
earlier, and renders the camera view of the current layer. Labels come from the exact
ground-truth contour, so they are free and unlimited. Fine-tune on real layer images
afterwards (models/train.py --real-dir) and report real-image results separately.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .. import camera
from ..deform import KINDS, random_failure
from ..parts import PART_NAMES, get_part
from .classical import LABELS, boundary_deviation

SIZE = 256


def _layers_cache():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = get_part(name).layers()
        return cache[name]
    return get


def make_sample(rng, get, p_fail=0.6, dev_thresh=1.0):
    name = PART_NAMES[rng.integers(len(PART_NAMES))]
    P = get(name)
    k = int(rng.integers(6, len(P)))
    f = None
    if rng.random() < p_fail:
        kind = KINDS[rng.integers(len(KINDS))]
        k0 = max(2, k - int(rng.integers(0, 10)))
        f = random_failure(P, rng, kind=kind, k0=k0)
    actual = []
    for j in range(max(1, k - 3), k + 1):
        actual.append(f.apply(P[j], j) if f is not None and f.active(j) else P[j])
    img = camera.render([P[0]] + actual, rng)
    cur, exp = actual[-1], P[k]
    label = "ok"
    if f is not None:
        dev = 99.0 if cur.region.area < 0.35 * exp.region.area else boundary_deviation(cur.region, exp.region)
        if dev >= dev_thresh:
            label = {"collapse": "collapse", "warp": "warp", "shift": "shift",
                     "under_extrusion": "under_extrusion", "detach": "detach"}[f.kind]
    big = lambda m: cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    img_s = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    exp_s = big(camera.expected_mask(exp) * 255)
    tgt_s = big(camera.rasterise(cur.region))
    return img_s, exp_s, tgt_s, LABELS.index(label)


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    get = _layers_cache()
    imgs, exps, tgts, labs = [], [], [], []
    for _ in range(n):
        a, b, c, d = make_sample(rng, get)
        imgs.append(a), exps.append(b), tgts.append(c), labs.append(d)
    return dict(img=np.stack(imgs), exp=np.stack(exps), tgt=np.stack(tgts), label=np.array(labs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synth.npz")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d = make_dataset(a.n, a.seed)
    np.savez_compressed(a.out, **d)
    print("saved", a.out, {k: v.shape for k, v in d.items()}, np.bincount(d["label"], minlength=len(LABELS)))
