# Test part 2: water tray

A 30 x 20 x 12 mm open tray with 1.2 mm walls and floor (about 2.4 g of PLA, holds about 5 ml).
**Function:** holds water without leaking. Here the dent tests **how hard**, not where:
- a **gentle push** only bends the wall: the form changes but it still holds water;
- **poking through** opens the wall: it leaks.

Files: `water-tray.stl` (print this), `diagram.png` (dimensions, where to dent), `water-tray.yaml`
(geometry and function spec for our pipeline, using the existing `container` rule). Regenerate
the STL with `python tools/make_water_tray.py`.

## Print it (Bambu P1S, Bambu Studio)

1. Import `water-tray.stl`, open side up (as imported).
2. PLA, 0.20 mm layer profile, **wall loops 3**, **bottom shell layers 6** (the whole 1.2 mm floor
   solid; the default 3 leaves infill in the floor, which can leak), no supports, no brim,
   **Arc fitting off**, Timelapse = **Traditional**. Check the preview: walls and floor solid, no gaps.
3. Slice, then *File → Export → Export plate sliced file* (`water-tray.gcode.3mf`). Send that back too.

Layer 30 is at z = 6 mm (half height). See `diagram.png`.

## Test runs

| Run | What to do at layer 30 (z = 6 mm) | Expected |
|---|---|---|
| 1 | nothing (clean control) | holds water (if it already leaks, tell us before runs 2-3) |
| 2 | press the long front wall **gently with a flat tool**, about 1-2 mm, without breaking it | holds water (cosmetic dent) |
| 3 | **poke through** the long front wall with a pointed tool (awl, small screwdriver tip) | leaks |

Pause at layer 30 (Bambu Studio preview: right-click the "+" on the layer slider → *Add Pause*,
or pause from the printer screen at 6 mm), wait for the head to park, dent, resume.
**Safety:** nozzle about 220 °C, bed about 60 °C: use a tool, never fingers.

Run `watch` during runs 2 and 3 (what it does and how: see `../two-hole-block/README.md`):

```bash
python -m recovery.cli check-gcode water-tray.gcode.3mf
python -m recovery.cli watch --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
```

## Check each finished part

- **Water test:** fill to the brim, stand it on a dry paper towel for 10 minutes, look for wet
  spots or a falling level. Note: leaks / no leaks, and where.
- Calipers: wall thickness at the dent and away from it; depth of the dent.
- Photo of the dent from the side and from above.

Send back: the `.gcode.3mf`, the `results/watch/<time>/` folders, the results and photos.
