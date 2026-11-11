# Calibrating the P1S camera without a checkerboard

`python -m recovery.cli calibrate-bed` replaces the checkerboard with the plate itself: the bed is a
known 256 x 256 mm rectangle and the chamber camera is fixed. Clicking points along the four plate
edges in ONE frame fits focal length, one radial-distortion term and the camera pose
(`recovery/calib_offline.py`). Datasheet values (1280x720, 110 deg diagonal FOV) are only a weak prior.

## Why not datasheet values alone
Bambu publishes no intrinsics, distortion or mount pose, and resellers disagree on the FOV (110 vs 119 deg).
`tools/calibration_sensitivity.py` shows datasheet-level errors give 1.5 to 17 mm mean bed error against a 1 mm alarm threshold.

## Expected accuracy (synthetic test, `tests/test_calib_offline.py`)
8 clicks per edge, 0.7 px click noise, wrong FOV prior, lens distortion k1 from -0.1 to -0.4:
mean bed error about 0.5 mm, worst point 1.5 to 2 mm (at the extremes of the plate). This is a simulation:
real accuracy depends on how well the plate edges can be seen and clicked. Always do the checks in step 5.
The checkerboard method remains the most accurate; use it if the checks fail.

## Steps
1. Printer on, bed EMPTY and clean, chamber lit, door closed, bed at the position used during prints (lowered
   Z does not matter, the fit is for the plate plane at Z=0). Printer in LAN mode.
2. Grab a frame and create the click page:
   ```bash
   python -m recovery.cli calibrate-bed --ip <printer-ip> --code <access-code> --out calib.npz
   ```
   (or `--image frame.png` for a saved frame). This writes `calib_frame.png` and `calib_click.html`.
3. Open `calib_click.html` in a browser. For each edge (front = nearest you / the door, then left, right,
   back as you face the printer) pick it in the dropdown and click **8 points spread from end to end along
   the visible plate edge**, including both corners. Click the true edge of the plate, not the bed
   clip or shadow. Press **Copy JSON** and save it as `points.json`. Edges that are cut off by the frame:
   click what is visible; at least 6 per edge are required.
4. Fit:
   ```bash
   python -m recovery.cli calibrate-bed --ip <printer-ip> --code <access-code> --points points.json --out calib.npz
   ```
   It prints the edge residual (RMS and max, in mm) and refuses to save if RMS > 1 mm or max > 2 mm, or the focal
   length is implausible. More, better-spread clicks fix most failures.
5. Check: open `calib_overlay.png`. The green outline and grid must sit on the real plate edges everywhere,
   especially the corners. Then run `watch` on a clean print: every layer should be `ok`.
6. If the camera or printer is moved later, keep the lens and refit only the pose from fresh clicks
   (the four corners are enough, but all four edges with >= 2 clicks each are used to find them):
   ```bash
   python -m recovery.cli calibrate-bed --ip <ip> --code <code> --points points.json --profile calib.npz --out calib.npz
   ```
   If the lens profile does not fit this unit the command says so (corner reprojection error), then use step 4.
