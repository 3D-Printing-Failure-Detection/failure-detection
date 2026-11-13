"""Use a real Bambu P1S: check the connection, calibrate its camera, watch a print.

Everything is seen through the printer's own camera (front-left, low, wide lens). There is no
top-down view, so nothing here re-projects the frame; the trained models work on it as it is.

First runs on a printer, safest first:
  python -m recovery.cli check-gcode part.gcode.3mf                 # offline: does the file parse?
  python -m recovery.cli probe --ip ... --serial ... --code ...     # read-only: status + one frame
  python -m recovery.cli calibrate --ip ... --code ... --calib-z 10 --out calib.npz
  python -m recovery.cli watch --gcode part.gcode.3mf --ip ... --serial ... --code ... [--calib calib.npz]
      # watch only: checks every layer, logs it, saves the frames; never sends anything
  python -m recovery.cli agent ...                                  # acts, with a person approving

The sliced file (`--gcode part.gcode.3mf`, or `gcode:` in a part YAML) gives the plan: exactly the
layers the printer is printing, and where the part sits on the plate.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import calibrate_bed_camera
from .parts import Part, get_part
from .printer.bambu import BambuCamera, BambuLink, BambuPrinter, PrintEnded

STATUS_KEYS = ("gcode_state", "layer_num", "total_layer_num", "mc_percent", "mc_remaining_time",
               "nozzle_temper", "bed_temper", "subtask_name", "gcode_file", "print_error")


def calibrate(ip, code, out, calib_z, pattern=(9, 6), square_mm=10.0, board_origin=(100.0, 100.0), n_frames=12):
    """Checkerboard calibration of the printer camera. calib_z: the bed Z shown on the printer screen for
    the flat shot (the bed moves down as a print grows; the camera does not, so the models need it)."""
    cam = BambuCamera(ip, code)
    frames = []
    print("Hold the printed checkerboard in view, different places/tilts for each shot.")
    for i in range(n_frames):
        input(f"[{i+1}/{n_frames}] position the board, press Enter to capture...")
        frames.append(cam.grab())
    input(f"Lay the board FLAT on the bed, first inner corner at bed X={board_origin[0]} "
          f"Y={board_origin[1]} mm, then press Enter...")
    pose = cam.grab()
    calib = calibrate_bed_camera(frames, pattern, square_mm, board_origin, pose)
    np.savez(out, K=calib.K, dist=calib.dist, rvec=calib.rvec, tvec=calib.tvec, calib_z_mm=float(calib_z))
    cv2.imwrite(str(out).replace(".npz", "_pose.png"), pose)
    print("saved", out)
    cam.close()


def probe(ip, serial, code, out="results/probe", wait_s=15.0, log=print):
    """Read-only connection check: printer status over MQTT and one camera frame. Sends only
    'pushall' (a status request), never a print command."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    link = BambuLink(ip, serial, code)
    cam = BambuCamera(ip, code)
    try:
        t0 = time.time()
        while time.time() - t0 < wait_s and link.get("gcode_state") is None:
            time.sleep(0.5)
        with link.lock:
            state = dict(link.state)
        if not state:
            log(f"no MQTT status within {wait_s:.0f}s: check IP, serial, access code, LAN mode (port 8883)")
        for k in STATUS_KEYS:
            log(f"  {k:18s} {state.get(k)}")
        (out / "state.json").write_text(json.dumps(state, indent=1, default=str))
        try:
            frame = cam.grab(timeout=20)
        except TimeoutError as e:
            log(f"camera: {e}. On newer firmware also enable LAN-mode liveview on the printer.")
            return state
        cv2.imwrite(str(out / "camera_raw.jpg"), frame)
        log(f"camera frame {frame.shape[1]}x{frame.shape[0]} -> {out / 'camera_raw.jpg'}")
        return state
    finally:
        cam.close()
        link.cli.loop_stop()
        link.cli.disconnect()


def _load_watch_part(part_path, gcode_path, plate=1):
    if part_path:
        part = get_part(part_path)
        if part.gcode_text is None:
            raise ValueError("watch needs `gcode:` (Bambu Studio sliced file) in the part YAML")
        return part
    return Part.from_gcode(gcode_path, plate=plate)


def run_watch(part_path, gcode_path, ip, serial, code, calib_path=None, layer_model="runs/p1s_layer",
              out="results/watch", plate=1, log=print):
    """Watch a print started from Bambu Studio or the printer screen; never sends anything. Every layer:
    grab a fresh frame and, if the failure detector is there, compare it with the previous frame and the
    plan (recovery/models/p1s_layer.py). Saves the raw frames and one JSON line per layer under
    <out>/<timestamp>/: real images for testing the models, and a record of what they said. Without the
    detector only the frames and the layer timing are saved."""
    from . import p1s_view as V
    part = _load_watch_part(part_path, gcode_path, plate)
    nom = part.layers()
    run = Path(out) / time.strftime("%Y%m%d-%H%M%S")
    frames = run / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    cam = None
    if calib_path and Path(calib_path).exists():
        cam = V.P1SCamera.load(calib_path)
    watcher = None
    if layer_model and (Path(layer_model) / "model.pt").exists():
        from .models.p1s_layer import LayerWatcher
        watcher = LayerWatcher(layer_model, cam)
    printer = BambuPrinter(ip, serial, code, nom, part.layer_h, origin=part.origin,
                           original_gcode=part.gcode_text, observe_only=True)
    meta = dict(part=part.name, n_layers=len(nom), layer_h=part.layer_h, origin=list(part.origin),
                dropped_layers=part.dropped_layers, layer_model=layer_model if watcher else None,
                calibration=str(calib_path) if cam else "approximate default camera", started=time.time())
    t0 = time.time()
    while time.time() - t0 < 15 and printer.link.get("total_layer_num") is None:
        time.sleep(0.5)
    meta["printer_total_layer_num"] = printer.link.get("total_layer_num")
    log(f"watching {len(nom)} parsed layers (printer reports total_layer_num="
        f"{meta['printer_total_layer_num']}); writing to {run}")
    if watcher is None:
        log(f"no failure detector at {layer_model}: saving frames and layer timing only")
    if part.dropped_layers:
        log(f"WARNING: {part.dropped_layers} empty layers were dropped while parsing, so layer "
            "indices may not match the printer's layer numbers")
    if meta["printer_total_layer_num"] not in (None, 0, len(nom)):
        log("WARNING: parsed layer count differs from the printer's; frames may be matched to the wrong layer")
    alarms, prev = 0, None
    try:
        with open(run / "layers.jsonl", "a") as f:
            while printer.n_done < printer.n_total:
                printer.print_layer()
                i = printer.n_done - 1
                frame = printer.snapshot()
                cv2.imwrite(str(frames / f"{i:04d}.jpg"), frame)
                rec = dict(layer=i, z=nom[i].z, t=time.time(), printer_layer_num=printer.link.get("layer_num"),
                           state=printer.link.get("gcode_state"), skipped_so_far=printer.skipped_layers)
                if watcher is not None:
                    r = watcher.check((frame if prev is None else prev)[..., ::-1], frame[..., ::-1],
                                      nom[:i + 1], part.origin)
                    rec.update(label=r["kind"], flagged_px=r["flagged_px"], type_prob=r["type_prob"])
                    alarms += r["kind"] != "ok"
                prev = frame
                f.write(json.dumps(rec, default=float) + "\n")
                f.flush()
                log(f"layer {i:4d} z={rec['z']:6.2f}  " + (f"{rec['label']:16s} {rec['flagged_px']} px differ"
                                                         if "label" in rec else "frame saved"))
    except PrintEnded as e:
        log(f"print ended: {e}")
    except KeyboardInterrupt:
        log("stopped by user (the print itself is not affected)")
    finally:
        printer.close()
        meta.update(finished=time.time(), layers_observed=printer.n_done - printer.skipped_layers,
                    layers_skipped=printer.skipped_layers, alarms=alarms)
        (run / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
        log(f"observed {meta['layers_observed']} layers ({printer.skipped_layers} skipped), {alarms} alarms; "
            f"see {run}")
    return run
