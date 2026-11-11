# Test part 3: knob (4 lobes)

A 31 mm, 4-lobe thumb knob, 8 mm tall (about 2.4 g of PLA), with a **hex hole straight through
for an M5 nut** (8.2 mm across flats; the nut is 8.0). **Function:** press an M5 nut into the hex,
thread an M5 bolt through it, and turn the bolt by hand with the knob.
- dent a **lobe tip**: the knob looks different but still grips and turns (cosmetic);
- dent the **bottom of a valley** between lobes: only 2.1 mm of wall separates it from the hex,
  so the hex narrows and the nut no longer fits.

Same idea as part 1 (where the dent lands decides the outcome), on a curved shape.

Files: `star-knob.stl` (print this), `diagram.png` (dimensions, where to dent), `star-knob.yaml`
(function spec for our pipeline: the existing `holes` rule measures the hex across its flats).
Regenerate the STL with `python tools/make_star_knob.py`.

## Print it (Bambu P1S, Bambu Studio)

1. Import `star-knob.stl`, flat on the bed as imported (hex vertical).
2. PLA, 0.20 mm layer profile, 3 wall loops, 15% sparse infill, no supports, no brim,
   **Arc fitting off**, Timelapse = **Traditional**.
3. Slice, then *File → Export → Export plate sliced file* (`star-knob.gcode.3mf`). Send that back too.

Layer 20 is at z = 4 mm (half height). See `diagram.png`.

## Test runs

| Run | What to do at layer 20 (z = 4 mm) | Expected |
|---|---|---|
| 1 | nothing (clean control) | nut fits (if a clean print is too tight or too loose, tell us) |
| 2 | dent the **tip of one lobe** about 1-2 mm | still works (cosmetic) |
| 3 | push a small flat screwdriver into the **bottom of one valley**, toward the centre, about 1 mm | nut does not fit |

The valleys are about 3 mm wide, so use a small tool. Pause at layer 20 (Bambu Studio preview:
right-click the "+" on the layer slider → *Add Pause*, or pause from the printer screen at 4 mm),
wait for the head to park, dent, resume. **Safety:** nozzle about 220 °C: use a tool, never fingers.

Run `watch` during runs 2 and 3 (what it does and how: see `../two-hole-block/README.md`):

```bash
python -m recovery.cli check-gcode star-knob.gcode.3mf
python -m recovery.cli watch --gcode star-knob.gcode.3mf --ip IP --serial SERIAL --code CODE
```

## Check each finished part

- **Function test:** does an M5 nut press into the hex (thumb or a vice, not a hammer)? Then
  thread an M5 bolt through and turn it by the knob: does the nut stay put?
- Calipers: the hex across its flats, all three pairs (nominal 8.2 mm), especially the pair
  facing the dented valley.
- Photo of the dent.

Send back: the `.gcode.3mf`, the `results/watch/<time>/` folders, the results and photos.
