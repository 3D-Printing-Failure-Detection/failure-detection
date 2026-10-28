# Functional recovery from 3D-print failures

A feedback loop that images every layer, catches a deviation, measures the as-printed shape,
checks it against the part's function spec, and then **continues, adapts the remaining
layers, or scraps**. Implements the Project 2 plan (6 stages + data + evaluation).

## Software

| Need | Software |
|---|---|
| Language | Python 3.12 (a venv is at `.venv/`) |
| Geometry | `shapely` 2.x (all contour/polygon math), `numpy`, `scipy` |
| Images | `opencv-python-headless` (render, threshold, contours, camera homography) |
| Trained models | `PyTorch`, `torchvision` (ResNet-18 detector), `segmentation-models-pytorch` (U-Net segmenter), `timm` |
| Slicer | **Bambu Studio** (GUI to slice; its CLI can slice headless) → `.gcode.3mf`, which the code parses as the nominal print. Built-in 2.5D slicer + `trimesh` only for simulation/STL experiments |
| Printer | **Bambu Lab P1S** in LAN mode: MQTT (`paho-mqtt`) for status/pause/resume/stop, FTPS (port 990) to upload, built-in chamber camera (port 6000 JPEG stream); a printed checkerboard to calibrate the camera |
| Interface | `gradio` |
| LLM planner (optional) | `anthropic` SDK (`ANTHROPIC_API_KEY`) |
| Eval / plots | `matplotlib`, `pytest` |

Install: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## How the plan maps to code

| Plan stage | Code |
|---|---|
| Slicer → expected layers | `gcode.py` (writer, parser, splicer), `parts.py`, `slicing.py` |
| Synthetic failures + renders | `deform.py` (collapse, warp, shift, under-extrusion, detach), `camera.py` |
| 1. Detect | `models/classical.py` `ClassicalDetector`; trained: `models/nets.py` `TorchDetector` |
| 2. Measure | `models/classical.py` `ClassicalSegmenter`; trained: `TorchSegmenter` (U-Net, image + expected render) |
| 3. Assess | `geometry/spec.py` (closed walls, flat base, capacity, min wall, volume, through-holes, connected/present features, excess overhang); specs live in each part's YAML |
| 4. Decide | `loop/decide.py` (rules); `loop/llm.py` (LLM may only choose among spec-passing options) |
| 5. Adapt | `geometry/adapt.py`: fit a displacement field (rigid shift + local residual) between the planned and measured contour, warp the remaining layers by it, fading over N layers; optional hole-relief |
| 6. Interface | `app.py` (Gradio), `printer/bambu.py` (Bambu P1S), `sim.py` (simulated printer, same interface) |
| Evaluation | `eval/run_eval.py`: ignore / scrap-restart / pause-and-alert / ours, on injected failures |

Decision logic: build candidate plans (continue; adapt with several fade lengths, with/without
hole-relief), predict the finished part for each, and take the first that passes the spec
(closest to the design first). If none passes → scrap.

## Adding your own part (nothing in the pipeline is part-specific)

A part is one YAML file: geometry + function spec. Copy `parts/_template.yaml.txt` to
`parts/mypart.yaml` (or keep it anywhere and pass the path).

- **Geometry:** a stack of `add` / `cut` extruded shapes (circle, rect, polygon; tapers via
  `r: [a, b]`; optional `z: [from, to]`), **or** `stl: path/to/part.stl` for a real mesh.
- **Spec** (use only what the part needs): `container` (holds liquid), `min_wall_mm`,
  `material_volume_mm3`, `holes` (through-holes), `connected` (must be one piece in a z-range,
  e.g. a handle), `present` (material must exist in a z-range).

The mug, cup, spacer and bracket in `parts/` are just examples written this way. Parts must fit
the camera's field of view (`config.CameraCfg`, 128 mm by default).

```bash
.venv/bin/python -m recovery.cli demo --part parts/mypart.yaml --kind shift
.venv/bin/python -m recovery.cli eval --parts parts/mypart.yaml mug --n 8
```

Limits of "generic": the checks and the adaptation work on layer cross-sections (2.5D parts).
Overhangs, strength and load are not modelled.

## Run

```bash
export PYTHONPATH=.
.venv/bin/python -m recovery.cli demo --part mug --kind collapse --seed 3
.venv/bin/python -m recovery.cli eval --n 8              # table + results/<out>/*.csv
.venv/bin/python -m recovery.cli app                      # Gradio UI
.venv/bin/python -m recovery.cli gen-data --n 4000 --out data/synth.npz
.venv/bin/python -m recovery.cli train --data data/synth.npz --epochs 15
.venv/bin/python -m recovery.cli demo --models torch      # uses runs/{det,seg}.pt
.venv/bin/python -m pytest tests
```

## Bambu Studio + P1S setup (week 1, untested on hardware)

**Slice in Bambu Studio** (P1S profile). Process settings that keep the parsed layers clean:
skirt loops 0, brim off, no supports, "Arc fitting" off, timelapse **traditional** (the head
parks at each layer change, giving the camera a clear view). Right-click the plate →
*Export plate sliced file* → `part.gcode.3mf`. Headless alternative: `slicing.bambu_studio_slice`.
Point the part YAML at it:

```yaml
name: mypart
gcode: mypart.gcode.3mf        # nominal layers = exactly what the printer will print
min_wall_mm: 1.6               # ...plus the function spec (see "Adding your own part")
```

**Printer:** enable LAN mode (and Developer mode if your firmware offers it; recent firmware
restricts third-party control without it). Note the IP, serial number and 8-digit access code.
Ports needed: 8883 (MQTT), 990 (FTPS), 6000 (camera).

**Camera:** the P1S chamber camera looks at the bed at an angle, not top-down. Print a
checkerboard at 100% scale, then run `calibrate` (12 shots at different tilts + one flat on the
bed at a known position). The pose-based `BedCamera.top_down` re-projects each frame onto the
plane at the current layer height, so it needs no overhead camera. For best accuracy fix a
dedicated overhead webcam later; the models only see the re-projected view either way.

```bash
.venv/bin/python -m recovery.cli calibrate --ip 192.168.1.50 --code 12345678 --out calib.npz
# start the print from Bambu Studio / the printer screen, then:
.venv/bin/python -m recovery.cli live --part parts/mypart.yaml --ip 192.168.1.50 \
    --serial 01P00A000000000 --code 12345678 --calib calib.npz
```

**The risky part — changing the remaining layers.** A running Bambu job cannot be edited. The
adapter (`BambuPrinter.replace_remaining`) pauses, stops, uploads a self-contained resume G-code
(heat, no homing, adapted layers, `M73 L<n>` so `layer_num` keeps counting) over FTPS and starts it
with `gcode_file`. That works only if the P1S still knows its position after STOP; if it does not,
pass `--rehome-gcode` (re-home XY and re-reference Z at a bed corner away from the part; never
`G28` over the part). `--mode stream` (send adapted layers via `gcode_line` while paused) is
experimental. **Test both on a scrap print before anything else**: this is the hinge of the
project (the plan's week-1 "pause, edit, resume" test).

## Status: what has and hasn't been verified

Written **and** exercised: the 4 parts pass their own specs; G-code write→parse round-trips
(IoU > 0.99); the classical segmenter/detector show 0 false alarms on clean layers in a spot
check and classify shift/warp/detach well, but collapse vs under-extrusion labels are often
confused; single scenarios (mug collapse, mug shift, cup collapse) are recovered in place where
the alert-only baseline scrapped; the training script ran 1 epoch on 120 samples.

**Not verified:** the full evaluation (an early 40-scenario smoke run showed *ours* and
*ignore* both ending functional in ~48% of cases and ours restarting less than scrap/alert, but
that was before later fixes and is far too small to quote); `pytest tests` (written, never run);
the Gradio app, `slicing.py`, `llm.py`; the whole Bambu path (MQTT/FTPS/camera/Bambu Studio CLI were written from community protocol docs, never run against a printer); real images; trained-model
accuracy (only a smoke run).

## Modelling assumptions to state in the report

- The simulator is a surrogate. Failures grow progressively (collapse over ~3 mm, shift over a
  few layers); once the controller adapts, the cause is assumed transient and later layers
  deposit as commanded (± 0.06 mm noise). Ignoring a failure lets it persist.
- "Functional" is the spec check in `geometry/spec.py` (including an excess-overhang rule), not a
  physical test. Validate with real prints (e.g. fill the mug with water).
- Abrupt shifts and gaps through already-printed walls are unrecoverable by design → scrap.
- Detection threshold is 1.0 mm; smaller deviations are treated as benign.
- A scrapped print counts as restarted successfully; methods differ in filament wasted.
