"""Render the sim-in-the-loop rehearsal (results/live_sim_check_v3) as one demo video.

Each print: zoomed camera view with the detector's marks, the full camera view, the layer status and a
timeline of flagged area per layer with the injected failure start and the first alarm.

    ~/.venvs/fd/bin/python videos/make_sim_video.py [--run results/live_sim_check_v3] [--out videos/...mp4] [--hq]
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 15
LAYER_S, ALARM_HOLD_S, TITLE_S, END_S = 0.2, 1.5, 2.0, 2.2
BG, PANEL, TEXT, MUTED = (14, 17, 23), (26, 31, 40), (235, 238, 243), (140, 148, 160)
OK, WARN, ALARM, START = (64, 196, 120), (240, 176, 60), (232, 64, 64), (120, 160, 255)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ORDER = [4, 3, 5, 6, 1, 2]          # shift, collapse, under-extrusion, detach, clean, warp (missed)
NAMES = dict(none="no failure (clean print)", under_extrusion="under-extrusion")


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def nice(kind):
    return NAMES.get(kind, kind)


def crop_box(run, layers):
    """One fixed 16:9 box around everything the detector drew (plan outline + flagged area) in this run."""
    xs, ys = [], []
    for r in layers:
        p = run / "marked" / f"{r['layer']:04d}.jpg"
        if not p.exists():
            continue
        m = np.asarray(Image.open(p).convert("RGB")).astype(int)
        f = np.asarray(Image.open(run / "frames" / p.name).convert("RGB")).astype(int)
        diff = np.abs(m - f).sum(2) > 60
        diff[:60] = False                       # the status banner at the top of marked frames
        yy, xx = np.nonzero(diff)
        if len(xx):
            xs += [xx.min(), xx.max()]
            ys += [yy.min(), yy.max()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * 2.2, (y1 - y0) * 2.2 * 16 / 9, 360)
    w = min(w, 1280)
    h = w * 9 / 16
    cx = min(max(cx, w / 2), 1280 - w / 2)
    cy = min(max(cy - h * 0.08, h / 2), 720 - h / 2)
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def status(r):
    if r.get("label") is None:
        return "waiting for first layer", MUTED
    if r.get("warmup"):
        return "warming up (learning the baseline)", MUTED
    if r.get("alarm"):
        return f"ALARM: {nice(r['label'])}", ALARM
    if r["label"] != "ok":
        return f"suspect: {nice(r['label'])}", WARN
    return "OK, matches the plan", OK


def draw_frame(res, meta, layers, i, box, first_alarm):
    run = Path(res["run"])
    r = layers[i]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # header
    d.text((40, 26), "3D-print failure detector", font=font(34, True), fill=TEXT)
    d.text((40, 70), "simulated Bambu P1S (MuJoCo), checked live, layer by layer, through the printer camera",
           font=font(20), fill=MUTED)
    tag = f"{res['part']}  ·  injected: {nice(res['failure'])}"
    if res["failure_starts"] is not None:
        tag += f" from layer {res['failure_starts']}"
    tw = d.textlength(tag, font=font(24, True))
    d.rounded_rectangle((W - 60 - tw - 28, 34, W - 40, 82), 12, fill=PANEL)
    d.text((W - 54 - tw, 44), tag, font=font(24, True), fill=TEXT)

    label, col = status(r)
    marked = run / "marked" / f"{r['layer']:04d}.jpg"
    src = Image.open(marked if marked.exists() else run / "frames" / f"{r['layer']:04d}.jpg").convert("RGB")
    raw = Image.open(run / "frames" / f"{r['layer']:04d}.jpg").convert("RGB")
    # zoomed view
    zx, zy, zw, zh = 40, 120, 1216, 684
    d.rounded_rectangle((zx - 6, zy - 6, zx + zw + 6, zy + zh + 6), 10, fill=col)
    img.paste(src.crop(box).resize((zw, zh), Image.LANCZOS), (zx, zy))
    d.rounded_rectangle((zx + 14, zy + 14, zx + 300, zy + 50), 8, fill=(0, 0, 0))
    d.text((zx + 26, zy + 20), "zoomed printer camera", font=font(20), fill=TEXT)
    # right column
    rx = 1290
    full = raw.resize((590, 332), Image.LANCZOS)
    fd = ImageDraw.Draw(full)
    s = 590 / 1280
    fd.rectangle([v * s for v in box], outline=(255, 255, 255), width=2)
    img.paste(full, (rx, 120))
    d.text((rx, 460), "full camera view (box = zoom)", font=font(18), fill=MUTED)
    d.rounded_rectangle((rx, 500, rx + 590, 804), 14, fill=PANEL)
    d.text((rx + 24, 516), "LAYER", font=font(20), fill=MUTED)
    d.text((rx + 24, 540), f"{r['layer']} / {meta['n_layers']}", font=font(56, True), fill=TEXT)
    d.text((rx + 300, 516), "HEIGHT", font=font(20), fill=MUTED)
    d.text((rx + 300, 540), f"{r['z']:.2f} mm", font=font(56, True), fill=TEXT)
    d.rounded_rectangle((rx + 24, 626, rx + 566, 680), 10, fill=col)
    d.text((rx + 40, 636), label, font=font(28, True), fill=(10, 10, 10))
    if r.get("flagged_px") is not None:
        d.text((rx + 24, 700), f"area different from the plan: {r['flagged_px']} px", font=font(22), fill=TEXT)
    d.text((rx + 24, 736), "alarm rule: 3 failing layers in a row", font=font(20), fill=MUTED)
    if first_alarm and r["layer"] >= first_alarm["layer"]:
        late = res.get("layers_late")
        d.text((rx + 24, 766), f"alarm at layer {first_alarm['layer']}"
               + (f" ({late} layers after the failure began)" if late is not None else ""),
               font=font(20, True), fill=ALARM)
    # timeline
    tx0, tx1, ty0, ty1 = 40, W - 40, 860, 1030
    d.rounded_rectangle((tx0, ty0 - 20, tx1, ty1 + 30), 14, fill=PANEL)
    d.text((tx0 + 20, ty0 - 10), "area flagged per layer", font=font(18), fill=MUTED)
    n = len(layers)
    peak = max([l.get("flagged_px") or 0 for l in layers] + [1])
    bw = (tx1 - tx0 - 40) / n
    base = ty1 - 8
    for j, l in enumerate(layers):
        x = tx0 + 20 + j * bw
        fp = l.get("flagged_px") or 0
        c = ALARM if l.get("alarm") else WARN if l.get("label") not in (None, "ok") else (70, 80, 95)
        if j > i:
            c = tuple(int(v * 0.35 + 26 * 0.65) for v in c)
        hgt = 6 + (base - ty0 - 30) * fp / peak
        d.rectangle((x + 1, base - hgt, x + bw - 2, base), fill=c)
        if res["failure_starts"] == l["layer"]:
            d.line((x, ty0 + 14, x, base), fill=START, width=3)
            d.text((x + 6, ty0 + 12), "failure starts", font=font(18, True), fill=START)
        if first_alarm and first_alarm["layer"] == l["layer"]:
            d.polygon([(x + bw / 2, ty0 + 62), (x + bw / 2 - 10, ty0 + 44), (x + bw / 2 + 10, ty0 + 44)], fill=ALARM)
            d.text((x + bw / 2 + 14, ty0 + 42), "ALARM", font=font(18, True), fill=ALARM)
    px = tx0 + 20 + (i + 0.5) * bw
    d.line((px, ty0 + 8, px, base + 6), fill=TEXT, width=2)
    d.text((tx0 + 20, base + 6), f"layer {layers[0]['layer']}", font=font(16), fill=MUTED)
    end = f"layer {layers[-1]['layer']}"
    d.text((tx1 - 20 - d.textlength(end, font=font(16)), base + 6), end, font=font(16), fill=MUTED)
    return np.asarray(img)


def card(lines, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H / 2 - 40 * len(lines)
    for k, (t, c, sz) in enumerate(lines):
        f = font(sz, True)
        d.text(((W - d.textlength(t, font=f)) / 2, y), t, font=f, fill=c)
        y += sz + 26
    if sub:
        for t in sub:
            f = font(26)
            d.text(((W - d.textlength(t, font=f)) / 2, y + 20), t, font=f, fill=MUTED)
            y += 40
    return np.asarray(img)


def result_line(res):
    if res["failure"] == "none":
        return ("no alarm on a clean print: correct", OK) if not res["first_alarm"] else ("false alarm", ALARM)
    if res["first_alarm"]:
        return f"caught: {nice(res['first_alarm']['kind'])}, {res['layers_late']} layers after it began", OK
    return "missed", ALARM


def summary(results):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((120, 110), "Rehearsal summary: 7 simulated prints, model v3", font=font(44, True), fill=TEXT)
    d.text((120, 176), "Trained only on rendered frames. Real-printer test still to come.", font=font(24), fill=MUTED)
    y = 260
    for res in results:
        t, c = result_line(res)
        d.rounded_rectangle((120, y, W - 120, y + 76), 12, fill=PANEL)
        d.text((150, y + 20), f"{res['part']}", font=font(28, True), fill=TEXT)
        d.text((480, y + 20), nice(res["failure"]), font=font(28), fill=TEXT)
        d.text((960, y + 20), t, font=font(28, True), fill=c)
        y += 96
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/live_sim_check_v3")
    ap.add_argument("--out", default="videos/sim_failure_detection.mp4")
    ap.add_argument("--hq", action="store_true", help="full 1080p, ~45 MB (default: 720p, ~2 MB, small enough for git)")
    ap.add_argument("--only", type=int, nargs="*", help="rehearsal indices to include (default: all, in ORDER)")
    ap.add_argument("--no-summary", action="store_true")
    a = ap.parse_args()
    results = json.loads(Path(a.run, "results.json").read_text())
    small = [] if a.hq else ["-vf", "scale=1280:-2"]
    w = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                           ffmpeg_params=small + ["-crf", "18" if a.hq else "32", "-preset", "slow",
                                                  "-movflags", "+faststart"])

    def put(frame, seconds):
        for _ in range(max(1, round(seconds * FPS))):
            w.append_data(frame)

    put(card([("3D-print failure detector", TEXT, 64), ("sim-in-the-loop rehearsal", START, 40)],
             ["a simulated Bambu P1S prints; the trained model watches its camera every layer",
              "and raises an alarm after 3 failing layers in a row"]), 3.0)
    for k in (a.only or ORDER):
        res = results[k]
        run = Path(res["run"])
        meta = json.loads((run / "meta.json").read_text())
        layers = [json.loads(l) for l in (run / "layers.jsonl").read_text().splitlines() if l.strip()]
        box = crop_box(run, layers)
        sub = [f"failure injected at layer {res['failure_starts']}"] if res["failure_starts"] is not None else \
            ["nothing goes wrong: any alarm would be a false alarm"]
        put(card([(res["part"], TEXT, 60), (nice(res["failure"]), START, 40)], sub), TITLE_S)
        fa = res["first_alarm"]
        for i, r in enumerate(layers):
            fr = draw_frame(res, meta, layers, i, box, fa)
            put(fr, ALARM_HOLD_S if fa and r["layer"] == fa["layer"] else LAYER_S)
        t, c = result_line(res)
        put(np.asarray(Image.blend(Image.fromarray(fr), Image.fromarray(card([(t, c, 48)])), 0.85)), END_S)
        print(res["part"], res["failure"], "->", t, flush=True)
    if not a.no_summary:
        put(summary(results), 6.0)
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
