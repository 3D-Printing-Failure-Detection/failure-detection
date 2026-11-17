# Agent test on the P1S (water tray)

**Goal:** prove the agent pipeline works on the real printer:
- it notices the pause, checks the part, recommends an action and waits for approval;
- then the printer resumes, or takes an uploaded G-code file and continues printing.

**Not needed:**
- **Dent model:** you type what you did to the part (`dent X Y DEPTH`), the default on the real printer.
- **Calibration:** without `calib.npz` the agent uses an approximate camera; this only affects the saved photos.
- **LLM:** decisions use the built-in rules (`--decider rules`, the default).

Do the tests in order. Start a test only once the one before it has passed.

| # | Test | Prints | Proves | Status |
|---|---|---|---|---|
| 0 | setup | none | the sliced file is understood; the agent runs in simulation | passed 11-16 |
| 1 | 1 mm dent → **resume** | 1 tray | the agent notices the pause, checks, recommends, and its resume works | passed 11-17 |
| 2 | re-upload test, no dent → **restart** | 1 tray | stop → upload → start works on the P1S (the step behind "correct") | rerun |
| 3 | 3 mm dent → **correct** | 1 tray | the whole chain: corrected layers uploaded and printed, tray holds water | rerun |

Replace `IP SERIAL CODE` below with the printer's values (the same as in `connection-test`).

## Safety

- Hot nozzle and bed: dent with a tool, never fingers.
- Never resume from the printer screen or the app; the agent does it.
- If the bed homes, rises toward the nozzle over the part, or anything collides: press **Stop** on the printer.

## 0. Setup (no printing, about 10 min)

1. Pull the latest code: `git checkout main && git pull` (`agent-pipeline` is the same).
2. Slice the tray in Bambu Studio:
   - Import `parts/print/water-tray/water-tray.stl`, open side up (as imported).
   - Settings: PLA, 0.20 mm layer profile, **wall loops 3**, **bottom shell layers 6**, no supports,
     no brim, **Arc fitting off**, Timelapse **Traditional**.
   - File → Export → *Export plate sliced file*. Save it as `water-tray.gcode.3mf` in the repo folder.
3. Check that our code understands the file:
   ```bash
   python -m recovery.cli check-gcode water-tray.gcode.3mf
   ```
   - **Pass:** the output ends with `looks usable`.
   - The preview PNG it saves looks like the tray (two outlines, no extra holes).
   - If not: stop and send the output and the PNG.
4. Rehearse in simulation:
   ```bash
   python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
       --pause-at-layer 30 --dent 0 -10 3
   ```
   - **Pass:** it recommends **CORRECT**.
   - Press Enter to accept; the simulated tray finishes and holds water (about 5.2 ml).

## How every printed test goes

1. Put a clean plate in. FAILED or FINISH on the screen from the last print is normal.
2. Start the print and the agent:
   - Print: send `water-tray.gcode.3mf` from Bambu Studio as usual.
   - Agent: the command for that test (below), run in the repo folder.
   - Either order is fine. Starting the agent mid-print is fine too, as long as it's before layer 30.
3. Check the agent is following the print:
   - It first says `Connected.`
   - If it says `no status from the printer`: close Bambu Studio / Bambu Handy and run `connection-test`.
   - Then it prints a line per layer, e.g. `printer: RUNNING, layer 12/60`.
   - **If no lines appear, stop and tell us.**
4. When a line ends with `<- pause from the printer screen now` (layer 30, z = 6 mm):
   - press **Pause on the printer screen**;
   - wait for the head to park.
5. The agent says `Printer paused after layer N of M` (test 2: `RESUME TEST after layer N`).
   - Write down N.
   - Nothing happens in the agent until the printer is paused: the prompts only appear after the pause.
6. Do what the test says to the part, then follow the agent's prompts.
7. Approve: the agent shows what it found and `Recommended: X. Enter = accept, or [1] ... [2] ...`.
   Press Enter to accept, or type a number.
8. After the print, do the **water test**:
   - fill the tray to the brim;
   - stand it on a dry paper towel for 10 min;
   - note any leaks and where they are.

## 1. Dent 1 mm → resume (passed 11-17)

```bash
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --dent-test \
    --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
```

- At the pause, gently press the **middle of the long front wall** inward, about 1 mm, with a flat
  tool. Don't break it.
- Press Enter in the agent, then type: `dent 0 -10 1`
- **Pass:**
  - it recommends **RESUME**;
  - accept it: the printer resumes by itself and finishes;
  - the tray holds water.

## 2. Re-upload test (scrap tray, no dent) → restart

The risky one: watch the bed the whole time it restarts.

**Status (11-17):** the stop and the upload worked, and the uploaded file printed fine when started
by hand. Still untested: the agent starting the file by itself.

**What happens and why:**
- A running Bambu job can't be edited, so "correct" doesn't change the paused job. Instead the system:
  1. stops the paused job;
  2. uploads a new file with only the remaining layers (in this test, the original ones, unchanged);
  3. starts that file as a new job.
- The new file heats up, then continues at the current height (about 6 mm).
- It deliberately doesn't home Z or level the bed: the nozzle would hit the part.
- So it only works if the printer still knows where the head is after the stop. This test checks
  exactly that, before any corrected layers are involved.

**Normal:** the job stops → a new job starts on the screen → the printer heats → the next layer
prints right on top of the old one.

**Not normal (press Stop):**
- it homes or levels the bed;
- the bed moves up toward the nozzle over the part;
- the first new layer prints in the air or digs into the part.

```bash
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --resume-test \
    --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
```

- At the pause, **don't touch the part**.
- The agent shows `RESUME TEST` and recommends **RESTART**. Accept it.
- The printer stops. The agent prints `uploaded adapted_<number>.gcode; starting it`, then `... is printing`.
- If instead it says `The printer did not start ... by itself`:
  - start that exact file from the printer screen (SD card files);
  - the agent waits up to 15 min;
  - note which one happened: it tells us whether the printer accepts the start command from the laptop.
- At the end, the bed should drop and the head should park at the back left, away from the part.
- **Pass:**
  - no Z homing over the part, no collision;
  - it continues at the right height, with no visible seam at 6 mm;
  - the head parks at the end;
  - the tray holds water.
- If it fails: press Stop on the printer, then send everything (see below) and a video if you can.
  Do **not** do test 3 until this passes.

## 3. Dent 3 mm → correct

Same command as test 1 (`--dent-test`).

- At the pause, press the middle of the front wall inward about **3 mm**.
- Press Enter, then type `dent 0 -10 3`.
- It recommends **CORRECT**. **Before accepting**, open the picture whose path it prints
  (`correction_from_layer_N.png`):
  - the new layers should start at the dented wall;
  - and step back to the original wall over about 10 layers.
- Accept. The printer stops, uploads and restarts as in test 2 (start the file from the screen if
  the agent asks).
- **Pass:**
  - the new layers step back to the original wall;
  - the head parks at the end;
  - the tray holds water (about 5 ml).

## Send back after each test

- The full terminal output (copy and paste), including the `printer: ...` lines.
- The run folder `results/agent/<date-time>/`, zipped. It has:
  - the log (`events.jsonl`, `summary.json`);
  - the before/after photos;
  - any G-code the agent made.
- The layer number N from step 5.
- Photos of the finished tray: side and top, close up at 6 mm.
- The water-test result.
- **Photos and water test are needed for every test**: they show whether the part works.
- Anything odd: noises, bed moves, a long wait, an error message.
