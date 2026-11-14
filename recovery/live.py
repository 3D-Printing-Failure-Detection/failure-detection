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


def _load_watch_part(part_path, gcode_path, plate=1, log=print):
    if part_path:
        part = get_part(part_path)
        if part.gcode_text is None:
            log(f"note: {part.name} has no `gcode:`, so the plan comes from its YAML shape. That is right for the "
                "simulated printer; for a real print use the sliced file, so the layers match exactly.")
        return part
    return Part.from_gcode(gcode_path, plate=plate)


def annotate(frame, r=None, banner="", alarm=False):
    """The camera frame with the plan's outline (green) and what the checker flagged (red)."""
    img = frame.copy()
    if r is not None and "guide" in r:
        cs, _ = cv2.findContours((r["guide"] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (60, 220, 60), 2)
        from .models.p1s_layer import LayerWatcher
        pts = np.round(LayerWatcher.flagged_points(r)).astype(int)
        k = max(1, int(round(r["box"][2] / 288)))
        h, w = img.shape[:2]
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        over = np.zeros(img.shape[:2], np.uint8)
        over[pts[ok, 1], pts[ok, 0]] = 255
        over = cv2.dilate(over, np.ones((k + 1, k + 1), np.uint8))
        img[over > 0] = (0.4 * img[over > 0] + 0.6 * np.array([40, 40, 255])).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 200) if alarm else (30, 30, 30), -1)
    cv2.putText(img, banner, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>P1S watch</title><style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}
img{max-width:100%;border:1px solid #333}table{border-collapse:collapse;margin-top:12px}
td,th{padding:2px 10px;border-bottom:1px solid #333;text-align:left}.alarm{color:#ff6b6b;font-weight:bold}</style></head>
<body><h3 id="h">P1S watch</h3><img id="f" src="latest.jpg"><table id="t"></table><script>
async function tick(){try{const s=await (await fetch('status.json?'+Date.now())).json();
document.getElementById('h').textContent=s.headline;document.getElementById('h').className=s.alarm?'alarm':'';
document.getElementById('f').src='latest.jpg?'+Date.now();
document.getElementById('t').innerHTML='<tr><th>layer</th><th>z mm</th><th>check</th><th>flagged px</th><th></th></tr>'+
s.recent.map(r=>`<tr class="${r.alarm?'alarm':''}"><td>${r.layer}</td><td>${r.z.toFixed(2)}</td><td>${r.label||'-'}</td>
<td>${r.flagged_px??''}</td><td>${r.alarm?'ALARM':(r.warmup?'learning baseline':'')}</td></tr>`).join('');}catch(e){}}
setInterval(tick,2000);tick();</script></body></html>"""


class LiveView:
    """A small web page with the latest marked-up frame and the recent layers (http://host:port)."""
    def __init__(self, port, host="127.0.0.1"):
        import http.server
        import socketserver
        view = self
        self.jpg, self.status = b"", dict(headline="waiting for the first layer", alarm=False, recent=[])

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/latest.jpg":
                    body, ctype = view.jpg, "image/jpeg"
                elif path == "/status.json":
                    body, ctype = json.dumps(view.status, default=float).encode(), "application/json"
                else:
                    body, ctype = _PAGE.encode(), "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.ThreadingTCPServer((host, port), H)
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}"
        import threading
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def update(self, img, status):
        self.jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.status = status

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def run_watch(part_path, gcode_path, ip, serial, code, calib_path=None, layer_model="runs/p1s_layer",
              out="results/watch", plate=1, ports=None, view_port=None, view_host="127.0.0.1", warmup=3, confirm=3,
              log=print, stop_after=None):
    """Watch a print started from Bambu Studio or the printer screen; never sends anything. Every layer:
    grab a fresh frame from the printer camera and, if the failure detector is there, compare it with the
    previous frame and the plan (recovery/models/p1s_layer.py). The alarm rule is the agent's
    (recovery/alarm.py): the first `warmup` layers set a baseline, then `confirm` failing layers in a row
    raise an alarm. Saves under <out>/<timestamp>/: frames/ (raw), marked/ (outline + what was flagged),
    layers.jsonl (one line per layer) and meta.json. view_port: also serve a live page on that port.
    ports: {"mqtt": ..., "camera": ...} for a printer on non-standard ports (the simulated printer).
    stop_after: stop watching once that many layers are done (rehearsals that watch part of a print)."""
    from . import p1s_view as V
    from .alarm import AlarmRule
    part = _load_watch_part(part_path, gcode_path, plate, log)
    nom = part.layers()
    run = Path(out) / time.strftime("%Y%m%d-%H%M%S")
    frames, marked = run / "frames", run / "marked"
    frames.mkdir(parents=True, exist_ok=True)
    marked.mkdir(exist_ok=True)
    cam = None
    if calib_path and Path(calib_path).exists():
        cam = V.P1SCamera.load(calib_path)
    watcher = None
    if layer_model and (Path(layer_model) / "model.pt").exists():
        from .models.p1s_layer import LayerWatcher
        watcher = LayerWatcher(layer_model, cam)
    view = LiveView(view_port, view_host) if view_port is not None else None
    if view:
        log(f"live view: {view.url}")
    printer = BambuPrinter(ip, serial, code, nom, part.layer_h, origin=part.origin,
                           original_gcode=part.gcode_text, observe_only=True, ports=ports)
    meta = dict(part=part.name, n_layers=len(nom), layer_h=part.layer_h, origin=list(part.origin),
                dropped_layers=part.dropped_layers, layer_model=layer_model if watcher else None,
                calibration=str(calib_path) if cam else "approximate default camera", started=time.time(),
                alarm_rule=dict(warmup=warmup, confirm=confirm))
    t0 = time.time()
    while time.time() - t0 < 15 and printer.link.get("total_layer_num") is None:
        time.sleep(0.5)
    meta["printer_total_layer_num"] = printer.link.get("total_layer_num")
    log(f"watching {len(nom)} parsed layers (printer reports total_layer_num="
        f"{meta['printer_total_layer_num']}); writing to {run}")
    if watcher is None:
        log(f"no failure detector at {layer_model}: saving frames and layer timing only")
    if cam is None:
        log("no calibration file: using the approximate default camera position")
    if part.dropped_layers:
        log(f"WARNING: {part.dropped_layers} empty layers were dropped while parsing, so layer "
            "indices may not match the printer's layer numbers")
    if meta["printer_total_layer_num"] not in (None, 0, len(nom)):
        log("WARNING: parsed layer count differs from the printer's; frames may be matched to the wrong layer")
    rule = AlarmRule(warmup, confirm)
    flagged, unchecked, first_alarm, recent, prev = 0, 0, None, [], None
    try:
        with open(run / "layers.jsonl", "a") as f:
            while printer.n_done < min(printer.n_total, stop_after or printer.n_total):
                printer.print_layer()
                i = printer.n_done - 1
                rec = dict(layer=i, z=nom[i].z, t=time.time(), printer_layer_num=printer.link.get("layer_num"),
                           state=printer.link.get("gcode_state"), skipped_so_far=printer.skipped_layers)
                try:
                    frame = printer.snapshot()
                except TimeoutError as e:                        # the camera stalled: skip this layer, go on
                    unchecked += 1
                    rec.update(label="unchecked", error=str(e))
                    f.write(json.dumps(rec, default=float) + "\n")
                    f.flush()
                    log(f"layer {i:4d} z={rec['z']:6.2f}  unchecked: no new camera frame; watching goes on")
                    prev = None                                  # don't compare across the gap
                    continue
                cv2.imwrite(str(frames / f"{i:04d}.jpg"), frame)
                r, alarm = None, False
                if watcher is not None and prev is not None:
                    try:
                        r = watcher.check(prev[..., ::-1], frame[..., ::-1], nom[:i + 1], part.origin)
                    except Exception as e:                       # never let a check stop the watch
                        rec.update(label="unchecked", error=f"{type(e).__name__}: {e}")
                        unchecked += 1
                    else:
                        warm = rule.warming_up
                        alarm = rule.update(r["kind"], r["flagged_px"])
                        rec.update(label=r["kind"], flagged_px=r["flagged_px"], type_prob=r["type_prob"],
                                   baseline_px=rule.baseline, warmup=warm, alarm=alarm)
                        flagged += r["kind"] != "ok"
                        if alarm and first_alarm is None:
                            first_alarm = dict(layer=i, z=nom[i].z, kind=r["kind"], flagged_px=r["flagged_px"])
                prev = frame
                f.write(json.dumps(rec, default=float) + "\n")
                f.flush()
                if "flagged_px" in rec:
                    line = f"{rec['label']:16s} {rec['flagged_px']:5d} px flagged"
                    line += "  (learning this print's baseline)" if rec["warmup"] else ""
                    line += f"  ALARM: {confirm} layers in a row look like {r['kind']}" if alarm else ""
                else:
                    line = rec.get("label", "frame saved") + (f": {rec['error']}" if "error" in rec else "")
                log(f"layer {i:4d} z={rec['z']:6.2f}  {line}")
                head = (f"layer {i + 1}/{len(nom)}  " + (rec.get("label") or "")
                        + ("  ALARM" if alarm else "") + (f"  (first alarm at layer {first_alarm['layer']})"
                                                         if first_alarm and not alarm else ""))
                img = annotate(frame, r, head, alarm)
                cv2.imwrite(str(marked / f"{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if view:
                    recent = ([dict(layer=i, z=rec["z"], label=rec.get("label"), flagged_px=rec.get("flagged_px"),
                                    alarm=alarm, warmup=rec.get("warmup", False))] + recent)[:25]
                    view.update(img, dict(headline=head, alarm=bool(first_alarm), recent=recent))
    except PrintEnded as e:
        log(f"print ended: {e}")
    except KeyboardInterrupt:
        log("stopped by user (the print itself is not affected)")
    finally:
        printer.close()
        meta.update(finished=time.time(), layers_observed=printer.n_done - printer.skipped_layers,
                    layers_skipped=printer.skipped_layers, layers_flagged=flagged, layers_unchecked=unchecked,
                    first_alarm=first_alarm)
        (run / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
        log(f"observed {meta['layers_observed']} layers ({printer.skipped_layers} skipped, {unchecked} unchecked); "
            + (f"first alarm at layer {first_alarm['layer']} ({first_alarm['kind']})" if first_alarm else "no alarm")
            + f"; see {run}")
        if view:
            view.close()
    return run
