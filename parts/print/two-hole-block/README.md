# Test part 1: two-hole block

A 50 x 10 x 20 mm block with two vertical **M5 clearance holes (5.5 mm) 30 mm apart**.
**Function:** two M5 bolts on a 30 mm pattern pass straight through. The side walls next to each
hole are only 2.25 mm, so the **same dent** is:
- **cosmetic** in the middle of the block (still works), and
- **functional damage** next to a hole (the hole narrows and the bolt no longer fits).

That gives one shape where "form changed but function kept" and "function lost" are both easy to
produce on purpose and easy to check with a bolt.

Files: `two-hole-block.stl` (print this), `two-hole-block.yaml` (geometry and function spec for
our pipeline), regenerate the STL with `python tools/make_two_hole_block.py`.

## Print it (Bambu P1S, Bambu Studio)

1. Import `two-hole-block.stl`. Keep it flat as imported (holes vertical, 50 x 10 face on the bed).
2. PLA, 0.20 mm layer profile, 2 walls, 15% sparse infill, **no supports, no brim**,
   **Arc fitting off**, Timelapse = **Traditional** (head parks each layer so the camera sees the part).
3. Slice, then *File → Export → Export plate sliced file* (`two-hole-block.gcode.3mf`). Send that
   file back too: it is what `check-gcode` and `watch` read.

In the STL, the left end is at x = 0: **holes at x = 10 and x = 40 mm**, middle of the block at
x = 25 mm. Layer 50 is at z = 10 mm (half height).

## Test runs

| Run | What to do at layer 50 (z = 10 mm) | Expected |
|---|---|---|
| 1 | nothing (clean control) | functional |
| 2 | dent the long front face **in the middle** (x = 25), about 1-2 mm deep | functional (cosmetic dent) |
| 3 | dent the long front face **right next to the right hole** (x = 40), about 1 mm deep | **not** functional (hole too narrow) |

How to dent: add a pause at layer 50 in Bambu Studio's preview (right-click the "+" on the layer
slider → *Add Pause*), or pause from the printer screen when it reaches 10 mm. Wait for the head to
park, press a flat tool (spatula, screwdriver shaft) into the **top edge** of the front face, then
resume. **Safety:** the nozzle is about 220 °C and the bed about 60 °C: use a tool, never fingers,
and keep clear of the nozzle.

During runs 2 and 3, run `watch` from the repo (`main-with-stresstest`) so we get frames before and
after the dent:

```bash
python -m recovery.cli check-gcode two-hole-block.gcode.3mf
python -m recovery.cli watch --gcode two-hole-block.gcode.3mf --ip IP --serial SERIAL --code CODE --calib calib.npz
```

## Check each finished part

- Does an M5 bolt pass fully through **each** hole by hand? (the function test)
- Calipers: each hole diameter, and centre-to-centre spacing (nominal 30.0 mm).
- Note the dent: layer, position, rough depth; take a photo.

Send back: the `.gcode.3mf`, the `results/watch/<time>/` folders, the measurements and photos.
