"""A simulated P1S on this machine that talks exactly like the real one, for testing the live checker
end to end before (and alongside) a real print.

It serves the same two connections the real printer does, over TLS:
  * status and commands: MQTT on port 8883 (user `bblp`, the access code as password), reports on
    device/<serial>/report with gcode_state / layer_num / total_layer_num; pause, resume and stop work;
  * the chamber camera: port 6000, the P1 camera framing (16-byte header + JPEG), about 1 frame a second.
The frames are rendered from the P1S camera position (recovery/p1s_view.py) as the part prints layer by
layer, with a failure starting at a chosen layer if asked. The camera is slightly off from the calibration
the checker uses, as a real one is.

    python -m recovery.cli sim-printer --part parts/mug.yaml --fail warp --fail-at 60
    python -m recovery.cli watch --part parts/mug.yaml --ip 127.0.0.1 --serial SIMP1S --code 12345678
"""
from .. import p1s_view as V                 # first: selects the EGL backend before MuJoCo loads

import json
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..deform import random_failure
from ..p1s_dent_data import base_camera, sample_look
from ..p1s_layer_data import FAILURES, default_config, printed_stack

STATES = ("IDLE", "RUNNING", "PAUSE", "FINISH", "FAILED")


def self_signed_cert(directory):
    """A throwaway certificate for the TLS servers (the real printer's is self-signed too). Uses the
    `openssl` command, so nothing extra needs installing."""
    d = Path(directory)
    cert, key = d / "cert.pem", d / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                    "-nodes", "-days", "30", "-subj", "/CN=sim-p1s", "-keyout", str(key), "-out", str(cert)],
                   check=True, capture_output=True)
    return str(cert), str(key)


# ---------------------------------------------------------------- MQTT (the small part of 3.1.1 the printer uses)
def _varint(n):
    out = bytearray()
    while True:
        b, n = n % 128, n // 128
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _mqtt_str(s):
    b = s.encode()
    return struct.pack(">H", len(b)) + b


def _read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def _read_packet(sock):
    head = _read_exact(sock, 1)[0]
    mult, length = 1, 0
    while True:
        b = _read_exact(sock, 1)[0]
        length += (b & 0x7F) * mult
        if not b & 0x80:
            break
        mult *= 128
    return head >> 4, head & 0x0F, _read_exact(sock, length) if length else b""


class _Server:
    """A TLS listener that hands each client to `handle(sock)` on its own thread."""
    def __init__(self, host, port, ctx, handle):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.ctx, self.handle, self.closed, self.clients = ctx, handle, False, []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.closed:
            try:
                raw, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(raw,), daemon=True).start()

    def _serve(self, raw):
        try:
            s = self.ctx.wrap_socket(raw, server_side=True)
        except (OSError, ssl.SSLError):
            raw.close()
            return
        self.clients.append(s)
        try:
            self.handle(s)
        except (OSError, ConnectionError, ssl.SSLError, struct.error):
            pass
        finally:
            if s in self.clients:
                self.clients.remove(s)
            try:
                s.close()
            except OSError:
                pass

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass
        for c in list(self.clients):
            try:
                c.shutdown(socket.SHUT_RDWR)
                c.close()
            except OSError:
                pass


# ---------------------------------------------------------------- the printer
class FakeP1S:
    """Prints `part` in simulation and serves it as a P1S. ports: 0 picks free ones (see .ports)."""

    def __init__(self, part, serial="SIMP1S", code="12345678", host="127.0.0.1", mqtt_port=8883, camera_port=6000,
                 layer_s=3.0, fail=None, fail_at=None, severity=None, start_layer=0, camera_error=1.0, seed=0,
                 fps=1.0, calib=None, log=print):
        self.part, self.serial, self.code, self.log = part, serial, code, log
        self.nominal = part.layers()
        self.n = len(self.nominal)
        self.layer_s, self.fps = layer_s, fps
        rng = np.random.default_rng(seed)
        self.failure = None
        if fail:
            if fail not in FAILURES:
                raise ValueError(f"fail must be one of {FAILURES}")
            k0 = int(fail_at) if fail_at is not None else int(rng.integers(max(6, self.n // 5), max(7, int(0.7 * self.n))))
            self.failure = random_failure(self.nominal, rng, kind=fail, k0=min(max(1, k0), self.n - 1),
                                          severity=severity)
        cfg = default_config(calib=calib)
        e = camera_error
        self.cam = base_camera(cfg).jittered(rng, pos_mm=2.0 * e, rot_deg=0.8 * e, f_rel=0.02 * e, k1=0.02 * e,
                                             k2=0.01 * e)
        self.xy, self.yaw = tuple(part.origin), 0.0
        b = np.array([l.region.bounds for l in self.nominal if not l.region.is_empty])
        radius = 0.5 * float(np.hypot(b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min()))
        self.look, _ = sample_look(rng, self.cam, self.xy, radius)
        self.photo = V.Photometric.sample(rng)
        self.rng = rng
        self.bulge = float(rng.uniform(0.02, 0.08))
        self.state, self.layer_num = "IDLE", 0
        self._next = int(start_layer)
        self._jpeg, self._jpeg_lock = None, threading.Lock()
        self._report_lock = threading.Lock()
        self._mqtt_clients = []
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._tmp = tempfile.TemporaryDirectory(prefix="sim-p1s-")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(*self_signed_cert(self._tmp.name))
        self._mqtt = _Server(host, mqtt_port, ctx, self._mqtt_client)
        self._camera = _Server(host, camera_port, ctx, self._camera_client)
        self.ports = dict(mqtt=self._mqtt.port, camera=self._camera.port)
        self._show(self._next)                                  # what the camera sees before the first layer

    # ---- what the camera sees
    def _show(self, k):
        """Render the print with layers 0..k-1 done."""
        if k <= 0:
            stack = self.nominal[:1]
            parts, top = {}, 0.0
        else:
            stack = printed_stack(self.nominal, self.failure, k)
            parts, top = {"p": V.stack_mesh(stack, bulge_mm=self.bulge)}, stack[-1].z
        with V.P1SScene(self.cam, self.look, parts, self.xy, self.yaw, top) as s:
            rgb, _ = s.render(list(parts))
        frame = self.photo.perturbed(self.rng).apply(rgb, self.rng)
        ok, jpg = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        with self._jpeg_lock:
            self._jpeg = jpg.tobytes()

    def frame(self):
        """The current camera frame (BGR), as the camera would send it."""
        with self._jpeg_lock:
            return cv2.imdecode(np.frombuffer(self._jpeg, np.uint8), cv2.IMREAD_COLOR)

    def _camera_client(self, s):
        hello = _read_exact(s, 80)
        if hello[48:80].rstrip(b"\0").decode(errors="replace") != self.code:
            return                                              # wrong access code: hang up, as the printer does
        while not self._stop.is_set():
            with self._jpeg_lock:
                jpg = self._jpeg
            s.sendall(struct.pack("<IIII", len(jpg), 0, 1, 0) + jpg)
            time.sleep(1.0 / self.fps)

    # ---- status and commands
    def report(self):
        return {"print": {"command": "push_status", "gcode_state": self.state, "layer_num": self.layer_num,
                          "total_layer_num": self.n, "mc_percent": int(100 * max(0, self.layer_num - 1) / self.n),
                          "mc_remaining_time": int(max(0, self.n - self.layer_num) * self.layer_s / 60),
                          "subtask_name": self.part.name, "gcode_file": f"{self.part.name}.gcode",
                          "nozzle_temper": 220.0, "bed_temper": 55.0, "print_error": 0}}

    def _publish(self, s=None):
        body = json.dumps(self.report()).encode()
        pkt = _mqtt_str(f"device/{self.serial}/report") + body
        pkt = bytes([0x30]) + _varint(len(pkt)) + pkt
        with self._report_lock:
            for c in ([s] if s is not None else list(self._mqtt_clients)):
                try:
                    c.sendall(pkt)
                except OSError:
                    pass

    def _mqtt_client(self, s):
        kind, _, body = _read_packet(s)
        if kind != 1:
            return
        # CONNECT: protocol name, level, flags, keepalive, client id, [will], username, password
        i = 2 + struct.unpack(">H", body[:2])[0] + 4
        flags = body[i - 3]
        fields = []
        while i < len(body):
            n = struct.unpack(">H", body[i:i + 2])[0]
            fields.append(body[i + 2:i + 2 + n])
            i += 2 + n
        password = fields[-1].decode(errors="replace") if flags & 0x40 else None
        if password != self.code:
            s.sendall(b"\x20\x02\x00\x05")                       # not authorised
            return
        s.sendall(b"\x20\x02\x00\x00")
        while not self._stop.is_set():
            kind, flags, body = _read_packet(s)
            if kind == 8:                                       # SUBSCRIBE
                pid = body[:2]
                n_topics, j = 0, 2
                while j < len(body):
                    j += 2 + struct.unpack(">H", body[j:j + 2])[0] + 1
                    n_topics += 1
                s.sendall(bytes([0x90]) + _varint(2 + n_topics) + pid + b"\x00" * n_topics)
                with self._report_lock:
                    self._mqtt_clients.append(s)
                self._publish(s)
            elif kind == 3:                                     # PUBLISH (a command)
                n = struct.unpack(">H", body[:2])[0]
                j = 2 + n
                if (flags >> 1) & 3:
                    s.sendall(b"\x40\x02" + body[j:j + 2])       # PUBACK
                    j += 2
                self._command(body[j:])
            elif kind == 12:                                    # PINGREQ
                s.sendall(b"\xd0\x00")
            elif kind == 14:                                    # DISCONNECT
                break
        with self._report_lock:
            if s in self._mqtt_clients:
                self._mqtt_clients.remove(s)

    def _command(self, payload):
        try:
            msg = json.loads(payload)
        except ValueError:
            return
        if "pushing" in msg:
            self._publish()
            return
        cmd = msg.get("print", {}).get("command")
        if cmd == "pause" and self.state == "RUNNING":
            self.state = "PAUSE"
            self._resume.clear()
            self.log(f"[sim printer] paused at layer {self.layer_num}")
        elif cmd == "resume" and self.state == "PAUSE":
            self.state = "RUNNING"
            self._resume.set()
            self.log(f"[sim printer] resumed at layer {self.layer_num}")
        elif cmd == "stop" and self.state in ("RUNNING", "PAUSE"):
            self.state = "IDLE"
            self._stop_print = True
            self._resume.set()
            self.log(f"[sim printer] stopped at layer {self.layer_num}")
        else:
            return
        self._publish()

    # ---- printing
    def run(self):
        """Print every layer (blocks). layer_num is the layer being printed, 1-based, as on the printer."""
        self._stop_print = False
        self.state = "RUNNING"
        self.log(f"[sim printer] printing {self.part.name}: {self.n} layers, {self.layer_s:.1f} s each"
                 + (f"; {self.failure.kind} starts at layer {self.failure.k0}" if self.failure else "; no failure"))
        for j in range(self._next, self.n):
            self.layer_num = j + 1
            self._publish()
            t_end = time.time() + self.layer_s
            while time.time() < t_end or self.state == "PAUSE":
                if self._stop.is_set() or self._stop_print:
                    return self.state
                if self.state == "PAUSE":
                    self._resume.wait(0.5)
                    t_end = time.time() + 0.5
                else:
                    time.sleep(0.05)
            t0 = time.time()
            self._show(j + 1)
            self.render_s = time.time() - t0
        if not self._stop_print:
            self.state, self.layer_num = "FINISH", self.n
            self._publish()
            self.log(f"[sim printer] finished {self.n} layers")
        return self.state

    def wait_for_watcher(self, timeout=None):
        """Block until something is reading both the status and the camera (watch or agent connected)."""
        t0 = time.time()
        while not (self._mqtt_clients and self._camera.clients):
            if self._stop.is_set() or (timeout is not None and time.time() - t0 > timeout):
                return False
            time.sleep(0.2)
        return True

    def start(self, wait_for_watcher=False):
        """run() on a background thread (after a watcher connects, if asked)."""
        def go():
            if wait_for_watcher:
                self.wait_for_watcher()
            try:
                self.run()
            except Exception as e:              # report it like the printer would, so a watcher does not wait forever
                self.state = "FAILED"
                self._publish()
                self.log(f"[sim printer] failed at layer {self.layer_num}: {type(e).__name__}: {e}")
                raise
        self._thread = threading.Thread(target=go, daemon=True)
        self._thread.start()
        return self

    def heartbeat(self):
        """Periodic status, as the printer sends about once a second."""
        def beat():
            while not self._stop.is_set():
                self._publish()
                time.sleep(1.0)
        threading.Thread(target=beat, daemon=True).start()
        return self

    def close(self):
        self._stop.set()
        self._resume.set()
        self._mqtt.close()
        self._camera.close()
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def main_sim_printer(a, log=print):
    """`recovery.cli sim-printer`: serve a simulated printer until the print ends (Ctrl-C stops it)."""
    from ..live import _load_watch_part
    part = _load_watch_part(a.part, a.gcode, a.plate)
    p = FakeP1S(part, a.serial, a.code, a.host, a.mqtt_port, a.camera_port, a.layer_s, a.fail, a.fail_at,
                a.severity, a.start_layer, a.camera_error, a.seed, calib=a.calib, log=log)
    log(f"simulated P1S on {a.host}: status port {p.ports['mqtt']}, camera port {p.ports['camera']}, "
        f"serial {a.serial}, access code {a.code}")
    src = f"--part {a.part}" if a.part else f"--gcode {a.gcode}"
    log(f"in another terminal:  python -m recovery.cli watch {src} --ip {a.host} --serial {a.serial} --code {a.code}"
        + ("" if (p.ports["mqtt"], p.ports["camera"]) == (8883, 6000)
           else f" --ports {p.ports['mqtt']},{p.ports['camera']}"))
    if a.wait_s:
        log(f"the print starts when watch or agent connects (or after {a.wait_s:.0f} s)")
        p.wait_for_watcher(a.wait_s)
        time.sleep(2.0)                                         # let it read the first status
    p.heartbeat()
    try:
        state = p.run()
        if a.linger_s:
            time.sleep(a.linger_s)                              # let the watcher see the finish
        return state
    except KeyboardInterrupt:
        return "interrupted"
    finally:
        p.close()
