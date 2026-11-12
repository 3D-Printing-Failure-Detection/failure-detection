"""Can this laptop talk to the P1S? A step-by-step check with a plain pass/fail per step.

  python -m recovery.cli connection-test --ip IP --serial SERIAL --code CODE
  python -m recovery.cli connection-test --mock          # same checklist against a fake printer

1. network    the printer answers on its three ports: 8883 (status/commands), 990 (file upload),
              6000 (camera)
2. status     log in over MQTT and receive the printer's status (state, layer, temperatures)
3. camera     receive one camera frame (saved as camera.jpg)
4. files      log in to the printer's SD card over FTPS and list it (read only)
5. commands   (only with --commands, and only while a print is running) pause, then resume,
              and confirm the printer reports each state. This is the one step that moves the
              printer; it asks before doing it.
Steps 1-4 never change anything on the printer. Results go to results/connection-test/<time>/.
"""
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np

PORTS = {8883: "status and commands (MQTT)", 990: "file upload (FTPS)", 6000: "camera"}
HINTS = {
    "network": "Is the laptop on the same network as the printer, with the printer's IP from its "
               "screen (Settings > WLAN)? Campus Wi-Fi often blocks device-to-device traffic: try a "
               "phone hotspot or your own router.",
    "status": "Check the serial number and the 8-character access code (Settings > WLAN/General), "
              "and that LAN Only Mode is on.",
    "camera": "Close Bambu Studio / Bambu Handy (one camera viewer at a time) and check LAN Only Mode.",
    "files": "Check the access code; an SD card must be in the printer.",
    "commands": "Commands need Developer Mode (Settings > WLAN > LAN Only Mode > Developer Mode) "
                "and a print that is running.",
}


def _port_open(ip, port, timeout=3.0):
    try:
        socket.create_connection((ip, port), timeout=timeout).close()
        return True
    except OSError:
        return False


class RealPrinter:
    """The checklist's view of the printer: thin wrappers over printer/bambu.py."""

    def __init__(self, ip, serial, code):
        self.ip, self.serial, self.code = ip, serial, code
        self.link = self.cam = None

    def ports(self):
        return {p: _port_open(self.ip, p) for p in PORTS}

    def status(self, wait_s=15.0):
        from .printer.bambu import BambuLink
        self.link = BambuLink(self.ip, self.serial, self.code)
        t0 = time.time()
        while time.time() - t0 < wait_s and self.link.get("gcode_state") is None:
            time.sleep(0.5)
        with self.link.lock:
            return dict(self.link.state)

    def frame(self, timeout=20.0):
        from .printer.bambu import BambuCamera
        self.cam = BambuCamera(self.ip, self.code)
        return self.cam.grab(timeout=timeout)

    def files(self):
        from .printer.bambu import ImplicitFTPS, _insecure_ctx
        ftp = ImplicitFTPS(context=_insecure_ctx())
        ftp.connect(self.ip, 990, timeout=15)
        ftp.login("bblp", self.code)
        ftp.prot_p()
        names = ftp.nlst()
        ftp.quit()
        return names

    def command_and_wait(self, cmd, state, timeout=60.0):
        self.link.command(cmd)
        return self.link.wait_state({state}, timeout=timeout)

    def close(self):
        if self.cam:
            self.cam.close()
        if self.link:
            self.link.cli.loop_stop()
            self.link.cli.disconnect()


class MockPrinter:
    """A fake P1S that behaves like a healthy one, to see the checklist before the real run.
    `broken` names steps that should fail, to see the hints."""

    def __init__(self, broken=()):
        self.broken, self.state = set(broken), "RUNNING"

    def ports(self):
        return {p: "network" not in self.broken for p in PORTS}

    def status(self, wait_s=0):
        if "status" in self.broken:
            return {}
        return dict(gcode_state=self.state, layer_num=12, total_layer_num=60, mc_percent=20,
                    nozzle_temper=220.0, bed_temper=55.0, wifi_signal="-48dBm", subtask_name="water-tray")

    def frame(self, timeout=0):
        if "camera" in self.broken:
            raise TimeoutError("no camera frame")
        img = np.full((720, 1280, 3), 40, np.uint8)
        cv2.putText(img, "MOCK P1S CAMERA", (420, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        return img

    def files(self):
        if "files" in self.broken:
            raise OSError("530 login incorrect")
        return ["cache", "timelapse", "water-tray.gcode.3mf"]

    def command_and_wait(self, cmd, state, timeout=0):
        if "commands" in self.broken:
            return False
        self.state = state
        return True

    def close(self):
        pass


def run(printer, out="results/connection-test", commands=False, ask=input, log=print):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir, n = Path(out) / stamp, 2
    while run_dir.exists():                            # two runs in the same second
        run_dir, n = Path(out) / f"{stamp}-{n}", n + 1
    run_dir.mkdir(parents=True)
    results = {}

    def step(name, ok, detail):
        results[name] = dict(ok=ok, detail=detail)
        log(f"  {'PASS' if ok else 'FAIL'}  {name:9s} {detail}")
        if not ok:
            log(f"        hint: {HINTS[name]}")
        return ok

    log("Connection test")
    try:
        ports = printer.ports()
        step("network", all(ports.values()),
             ", ".join(f"{p} {'open' if v else 'CLOSED'} ({PORTS[p]})" for p, v in ports.items()))
        if not any(ports.values()):
            return _finish(results, run_dir, log)
        try:
            st = printer.status()
        except Exception as e:                          # wrong code, TLS, refused...
            st, err = {}, f"{type(e).__name__}: {e}"
        else:
            err = "no status message within 15 s"
        ok = bool(st.get("gcode_state"))
        step("status", ok, (f"state {st.get('gcode_state')}, layer {st.get('layer_num')}/{st.get('total_layer_num')}, "
                            f"nozzle {st.get('nozzle_temper')} C, bed {st.get('bed_temper')} C, "
                            f"Wi-Fi {st.get('wifi_signal')}") if ok else err)
        try:
            img = printer.frame()
            cv2.imwrite(str(run_dir / "camera.jpg"), img)
            step("camera", True, f"{img.shape[1]}x{img.shape[0]} frame saved to {run_dir / 'camera.jpg'}")
        except Exception as e:
            step("camera", False, f"{type(e).__name__}: {e}")
        try:
            names = printer.files()
            step("files", True, f"{len(names)} entries on the SD card")
        except Exception as e:
            step("files", False, f"{type(e).__name__}: {e}")
        if commands:
            if st.get("gcode_state") != "RUNNING":
                step("commands", False, f"skipped: the printer is {st.get('gcode_state')}, start a print first")
            elif ask("  The printer will PAUSE and then RESUME the running print. Type yes to continue: ")\
                    .strip().lower() != "yes":
                log("  skipped commands test")
            else:
                t0 = time.time()
                paused = printer.command_and_wait("pause", "PAUSE")
                t1 = time.time()
                resumed = paused and printer.command_and_wait("resume", "RUNNING")
                step("commands", bool(paused and resumed),
                     f"pause {'confirmed' if paused else 'NOT confirmed'} in {t1 - t0:.0f} s, "
                     f"resume {'confirmed' if resumed else 'NOT confirmed'} in {time.time() - t1:.0f} s")
    finally:
        printer.close()
    return _finish(results, run_dir, log)


def _finish(results, run_dir, log):
    ok = all(r["ok"] for r in results.values())
    (run_dir / "report.json").write_text(json.dumps(dict(ok=ok, steps=results), indent=1))
    log(("All checks passed: the laptop and the printer can talk." if ok else
         "Some checks failed: fix the first FAIL above and run again.") + f"  Report: {run_dir / 'report.json'}")
    return dict(ok=ok, steps=results, dir=str(run_dir))
