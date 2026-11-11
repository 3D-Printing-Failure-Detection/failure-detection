"""Run the loop on a real Bambu P1S, and calibrate its chamber camera.

First runs on a printer, safest first:
  python -m recovery.cli check-gcode part.gcode.3mf                 # offline: does the file parse?
  python -m recovery.cli probe --ip ... --serial ... --code ...     # read-only: status + one frame
  python -m recovery.cli calibrate --ip 192.168.1.50 --code 12345678 --out calib.npz
  python -m recovery.cli watch --gcode part.gcode.3mf --ip ... --serial ... --code ... --calib calib.npz
      # watch-only: measures every layer, logs what it WOULD do, saves frames; never commands
  python -m recovery.cli live --part parts/mypart.yaml --ip ... --serial ... --code ... --calib calib.npz
      # acts: pauses, stops and uploads re-planned G-code. Only after watch runs look right.

The part YAML must point at Bambu Studio's sliced file (`gcode: part.gcode.3mf`) so the
nominal layers are exactly what the printer is printing. Start the print from Bambu Studio /
the printer screen as usual, then run `live`; the loop watches layer by layer.
"""
import json
import time
from pathlib import Path

import cv2

from . import camera
from . import config as C
from .camera import BedCamera, calibrate_bed_camera
from .loop.controller import Controller
from .loop.decide import RulePolicy
from .parts import Part, get_part
from .printer.bambu import BambuCamera, BambuLink, BambuPrinter, PrintEnded
from .run import make_models
from .types import Layer

STATUS_KEYS = ("gcode_state", "layer_num", "total_layer_num", "mc_percent", "mc_remaining_time",
               "nozzle_temper", "bed_temper", "subtask_name", "gcode_file", "print_error")


def calibrate(ip, code, out, pattern=(9, 6), square_mm=10.0, board_origin=(100.0, 100.0), n_frames=12):
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
    calib.save(out)
    cv2.imwrite(str(out).replace(".npz", "_pose.png"), pose)
    print("saved", out)
    cam.close()


def run_live(part_path, ip, serial, code, calib_path, models="classical", mode="restart",
             rehome_gcode="", skip_stress=False, log=print):
    part = get_part(part_path)
    nom = part.layers()
    if not skip_stress and part.spec.get("loads"):
        from .fea import stress_test
        rep = stress_test(nom, part.spec, part.layer_h, 1.0)
        log("pre-print stress test of the sliced part:\n" + rep.table())
        if not rep.ok:
            raise SystemExit("the part as sliced fails its stress tests; fix the design or the spec "
                             "(or pass --skip-stress)")
    if part.gcode_text is None:
        raise ValueError("live runs need `gcode:` (Bambu Studio sliced file) in the part YAML")
    printer = BambuPrinter(ip, serial, code, nom, part.layer_h, BedCamera.load(calib_path),
                           origin=part.origin, original_gcode=part.gcode_text, mode=mode,
                           rehome_gcode=rehome_gcode)
    seg, det = make_models(models)
    pol = RulePolicy(part.spec, nom, part.layer_h)
    ctl = Controller(printer, seg, det, pol, part.spec, nom, part.layer_h, log=log)
    t0 = time.time()
    try:
        res = ctl.run()
        log(f"done in {time.time()-t0:.0f}s: {res.outcome}; measured final check: {res.final}")
        return res
    finally:
        printer.close()


def probe(ip, serial, code, out="results/probe", calib_path=None, wait_s=15.0, log=print):
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
        if calib_path:
            top = BedCamera.load(calib_path).top_down(frame, 0.0, C.BED_CENTER)
            cv2.imwrite(str(out / "camera_topdown_bed.png"), top)
            log(f"top-down view of the bed centre -> {out / 'camera_topdown_bed.png'}")
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


def run_watch(part_path, gcode_path, ip, serial, code, calib_path, models="classical", out="results/watch",
              decide=True, plate=1, log=print):
    """Watch-only monitoring of a print started from Bambu Studio / the printer screen. Every
    layer: grab a fresh frame, measure, detect, and (on an alarm) ask the policy what it WOULD
    do. Nothing is sent to the printer. Saves raw + top-down frames, the expected mask and one
    JSON line per layer under <out>/<timestamp>/: real images for the dataset and evidence for
    tuning thresholds before `live` is allowed to act. Without a calibration file only the raw
    frames and the layer log are saved (still useful as real data); nothing is measured."""
    part = _load_watch_part(part_path, gcode_path, plate)
    calibrated = bool(calib_path) and Path(calib_path).exists()
    nom = part.layers()
    run = Path(out) / time.strftime("%Y%m%d-%H%M%S")
    frames = run / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    printer = BambuPrinter(ip, serial, code, nom, part.layer_h,
                           BedCamera.load(calib_path) if calibrated else None,
                           origin=part.origin, original_gcode=part.gcode_text, observe_only=True)
    seg, det = make_models(models)
    pol = RulePolicy(part.spec, nom, part.layer_h) if decide else None
    meta = dict(part=part.name, n_layers=len(nom), layer_h=part.layer_h, origin=list(part.origin),
                dropped_layers=part.dropped_layers, models=models, decide=decide, calibrated=calibrated,
                started=time.time())
    t0 = time.time()
    while time.time() - t0 < 15 and printer.link.get("total_layer_num") is None:
        time.sleep(0.5)
    meta["printer_total_layer_num"] = printer.link.get("total_layer_num")
    log(f"watching {len(nom)} parsed layers (printer reports total_layer_num="
        f"{meta['printer_total_layer_num']}); writing to {run}")
    if not calibrated:
        log(f"no calibration file ({calib_path}): saving raw frames and layer timing only, no measurement")
    if part.dropped_layers:
        log(f"WARNING: {part.dropped_layers} empty layers were dropped while parsing, so layer "
            "indices may not match the printer's layer numbers")
    if meta["printer_total_layer_num"] not in (None, 0, len(nom)):
        log("WARNING: parsed layer count differs from the printer's; frames may be matched to the wrong layer")
    alarms = 0
    try:
        with open(run / "layers.jsonl", "a") as f:
            while printer.n_done < printer.n_total:
                printer.print_layer()
                i = printer.n_done - 1
                if not calibrated:
                    frame = printer.cam.grab(newer_than=printer._layer_done_t)
                    cv2.imwrite(str(frames / f"{i:04d}_raw.jpg"), frame)
                    rec = dict(layer=i, z=printer.expected_layer().z, t=time.time(),
                               printer_layer_num=printer.link.get("layer_num"),
                               state=printer.link.get("gcode_state"), label="unmeasured",
                               skipped_so_far=printer.skipped_layers)
                    f.write(json.dumps(rec, default=float) + "\n")
                    f.flush()
                    log(f"layer {i:4d} z={rec['z']:6.2f}  frame saved (uncalibrated)")
                    continue
                img, exp = printer.camera_image(), printer.expected_layer()
                mask = camera.expected_mask(exp)
                meas = seg.measure(img, mask)
                d = det.detect(img, exp.region, meas)
                cv2.imwrite(str(frames / f"{i:04d}_raw.jpg"), printer.last_frame)
                cv2.imwrite(str(frames / f"{i:04d}_top.png"), img)
                cv2.imwrite(str(frames / f"{i:04d}_expected.png"), mask * 255)
                rec = dict(layer=i, z=exp.z, t=time.time(), printer_layer_num=printer.link.get("layer_num"),
                           state=printer.link.get("gcode_state"), label=d.label,
                           deviation_mm=d.deviation_mm, info=d.info, skipped_so_far=printer.skipped_layers)
                if d.label != "ok":
                    alarms += 1
                    if pol is not None:
                        # same belief as the live controller: nominal below, measured at this layer
                        dec = pol.decide(list(nom[:i]) + [Layer(exp.z, meas)], printer.plan, d)
                        rec.update(would_action=dec.action, would_reason=dec.reason)
                f.write(json.dumps(rec, default=float) + "\n")
                f.flush()
                log(f"layer {i:4d} z={exp.z:6.2f}  {d.label:16s} dev={d.deviation_mm:5.2f} mm"
                    + (f"  -> would {rec['would_action']}: {rec['would_reason']}" if "would_action" in rec else ""))
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
