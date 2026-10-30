"""Run the loop on a real Bambu P1S, and calibrate its chamber camera.

  python -m recovery.cli calibrate --ip 192.168.1.50 --code 12345678 --out calib.npz
  python -m recovery.cli live --part parts/mypart.yaml --ip ... --serial ... --code ... --calib calib.npz

The part YAML must point at Bambu Studio's sliced file (`gcode: part.gcode.3mf`) so the
nominal layers are exactly what the printer is printing. Start the print from Bambu Studio /
the printer screen as usual, then run `live`; the loop watches layer by layer.
"""
import time

import cv2

from .camera import BedCamera, calibrate_bed_camera
from .loop.controller import Controller
from .loop.decide import RulePolicy
from .parts import get_part
from .printer.bambu import BambuCamera, BambuPrinter
from .run import make_models


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
