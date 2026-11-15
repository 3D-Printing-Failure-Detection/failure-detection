# Spotting print failures with the printer's own camera

The P1S camera sits low at the front-left of the chamber and looks across the plate through a wide lens.
There is no camera looking straight down, so everything that stands in for the printer (the real printer
code, `watch`, the agent, and the simulator the agent rehearses on) works only on this view.

## How it works
After every layer the agent takes the camera frame and compares it with:
- the frame after the previous layer, and
- an outline of what the print should look like so far, drawn from the sliced file and the camera
  calibration.

A trained network (`recovery/models/p1s_layer.py`) marks where the print looks different from the plan and
what it looks like: a sagging wall (collapse), a wall curling out (warp), layers shifted sideways (shift),
a thin or open wall (under-extrusion), or a part that came off the bed (detach).

The agent then:
- uses the first 3 layers it checks to learn what the camera always flags on this print (calibration
  error, lighting) and ignores that much from then on;
- raises an alarm only after **3 layers in a row** look wrong, because a real failure stays and a fluke
  does not;
- **stops** for a detach or a shift; for anything else, pauses and recommends resuming (the camera cannot
  see the new shape well enough to say the part is ruined), then stops if the problem keeps growing;
- leaves every action to a person to approve;
- logs a layer as unchecked and carries on if the camera can't check it (no frame, or the part isn't
  where the plan says).

## Results (simulation only)
Each test print has its own part, colour, plate, lighting, placement and camera error. No real print has
been scored yet.

**Whole prints**, the way the agent works (`recovery/eval/p1s_print_eval.py`, 80 good and 80 failing
prints, alarm rule as above):

| | usual camera error | camera error ÷ 4 (a good calibration) |
|---|---|---|
| good prints that got a false alarm | **5 of 80 (6 %)** | 7 of 80 (9 %) |
| failing prints caught | **58 of 80 (73 %)** | 62 of 80 (78 %) |
| detach / shift caught | 13 of 13 / 14 of 14 | 13 of 13 / 14 of 14 |
| warp / collapse / under-extrusion caught | 13/18, 8/14, 10/21 | 13/18, 10/14, 12/21 |
| layers from the failure starting to the alarm | median 4, 90 % within 8 | median 4, 90 % within 9 |

**Single layers** (held-out sets the model never trained on):

| test set | failing layers caught | good layers flagged |
|---|---|---|
| new prints of the training parts (1200) | 69 % | 2.5 % |
| part shapes never seen in training (600, `tests/fixtures/unseen_parts/`) | 78 % | 7.3 % |
| twice the usual camera error (595) | 66 % | 3.1 % |

## What it is not good at yet
- **Round parts give most of the false alarms.** The mug, spacer and cup account for 10 of 12 false
  alarms across both whole-print runs; their curved walls get read as a slight warp or collapse. More
  training on round shapes is the next step.
- **Thin or open walls are often missed** (under-extrusion: about half). From this low angle, a thin
  patch on the far side of a part doesn't change much in the picture.
- **The simulator doesn't draw some real failures**: spaghetti, stringing, blobs, a part knocked over.
  A detach is drawn as the print no longer growing.
- **A better calibration doesn't reduce the false alarms.** They come from the model, not the camera.

## Run it
```bash
python -m recovery.p1s_layer_data --n 6000 --out data/p1s_layer_v1 --workers 12                 # ~30 min
python -m recovery.models.p1s_layer build --data data/p1s_layer_v1 --out data/p1s_layer_v1/crops.npz
python -m recovery.models.p1s_layer train --crops data/p1s_layer_v1/crops.npz --out runs/p1s_layer --pretrained  # ~21 min
python -m recovery.eval.p1s_print_eval --prints 160 --ckpt runs/p1s_layer                         # whole prints
```
On the printer: `watch` (checks and logs only) or `agent` (pauses and acts with your approval). Both
use `runs/p1s_layer` and `calib.npz` if they're there.

One mistake that's easy to make: draw the outline with the **calibration** camera, not the camera that
took the simulated pictures. The first version used the picture camera, so the outline fitted perfectly
in training, and with a real calibration every print raised an alarm.
