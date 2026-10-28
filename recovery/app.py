"""Gradio app: show the deviation, the decision and a preview of the adapted part.

  python -m recovery.app            # simulated printer
Pick a part and a failure, press Run. For a real printer, build a BambuPrinter
(printer/bambu.py) and feed its events through the same render functions.
"""
import tempfile

import cv2
import gradio as gr
import numpy as np

from . import camera
from .deform import KINDS
from .parts import PART_NAMES
from .run import run_scenario
from .viz import event_figure


def _overlay(event):
    """Camera frame with expected (grey) and measured (red) contours drawn on it."""
    img = event.image.copy()
    for region, col in ((event.expected.region, (200, 200, 200)), (event.measured.region, (40, 40, 255))):
        for p in getattr(region, "geoms", [region]):
            for ring in [p.exterior, *p.interiors]:
                pts = np.round(camera.mm_to_px(np.array(ring.coords))).astype(np.int32)
                cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(part, kind, seed, severity, policy, models):
    logs = []
    sc = run_scenario(part, kind, int(seed), severity if severity > 0 else None, policy, models,
                      log=logs.append)
    res = sc.result
    ev = res.events[0] if res.events else None
    status = [f"**Injected:** {sc.failure.kind} at layer {sc.failure.k0} "
              f"(severity {sc.failure.p.get('severity', 0):.1f} mm)",
              f"**If ignored:** {sc.ignore_verdict}"]
    if ev is None:
        status.append("**Loop:** no alarm raised; print finished as planned.")
    else:
        status.append(f"**Alarm** at layer {ev.layer}: {ev.label}, {ev.deviation_mm:.1f} mm off plan")
        status.append(f"**Decision:** {ev.action} — {ev.reason}")
    status.append(f"**Outcome:** {res.outcome}" + (f"; final check: {res.final}" if res.final else ""))
    table = [[c["name"], "pass" if c["ok"] else "fail", f"{c['fidelity']:.3f}",
              "; ".join(c["violations"])[:90]] for c in (ev.considered if ev else [])]
    img = _overlay(ev) if ev is not None else None
    fig = event_figure(sc.nominal, ev, sc.printed, f"{part}: {kind}") if ev and sc.printed else None
    path = tempfile.NamedTemporaryFile(suffix=".gcode", delete=False).name
    open(path, "w").write(sc.gcode)
    return "\n\n".join(status), img, fig, table, "\n".join(logs), path


def build():
    with gr.Blocks(title="Functional recovery from print failures") as demo:
        gr.Markdown("## Detect → measure → assess → decide → adapt\nSimulated printer; "
                    "inject a failure and watch the loop respond.")
        with gr.Row():
            part = gr.Dropdown(PART_NAMES, value="mug", label="Part")
            kind = gr.Dropdown(KINDS, value="collapse", label="Failure")
            seed = gr.Number(0, label="Seed", precision=0)
            sev = gr.Slider(0, 8, value=0, step=0.1, label="Severity mm (0 = random)")
            pol = gr.Radio(["ours", "alert"], value="ours", label="Policy")
            mdl = gr.Radio(["classical", "torch"], value="classical", label="Models")
        btn = gr.Button("Run", variant="primary")
        out_txt = gr.Markdown()
        with gr.Row():
            out_img = gr.Image(label="Camera at alarm (grey = expected, red = measured)")
            out_fig = gr.Plot(label="Deviation and adapted part")
        out_tab = gr.Dataframe(headers=["option", "spec", "fidelity", "violations"],
                               label="Options considered")
        out_log = gr.Textbox(label="Controller log", lines=6)
        out_file = gr.File(label="Final G-code")
        btn.click(run, [part, kind, seed, sev, pol, mdl],
                  [out_txt, out_img, out_fig, out_tab, out_log, out_file])
    return demo


if __name__ == "__main__":
    build().launch()
