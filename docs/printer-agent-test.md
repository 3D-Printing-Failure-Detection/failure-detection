# Agent test on the P1S (water tray)

Goal: prove the agent pipeline works on the real printer. That means it notices the pause, checks
the part, recommends an action and waits for approval. Then the printer must resume, or take an
uploaded G-code file and continue printing.

**No dent model and no calibration are needed.** You type what you did to the part
(`dent X Y DEPTH`), which is the default on the real printer. Without `calib.npz`, the agent prints
a note and uses an approximate camera, which only affects the saved photos. Decisions use the
built-in rules (`--decider rules`, the default). No LLM is needed.

Do the tests in order, and only start a test once the one before it has passed.

| # | Test | Prints | Proves |
|---|---|---|---|
| 0 | setup | none | the sliced file is understood; the agent runs in simulation |
| 1 | 1 mm dent → **resume** | 1 tray | the agent notices the pause, checks, recommends, and its resume works |
| 2 | re-upload test, no dent → **restart** | 1 tray | stop → upload → start works on the P1S (the step behind "correct") |
| 3 | 3 mm dent → **correct** | 1 tray | the whole chain: corrected layers uploaded and printed, and the tray holds water |

Replace `IP SERIAL CODE` below with the printer's values (the same ones as in `connection-test`).

## Safety (every printed test)

- The nozzle is about 220 °C and the bed about 60 °C. Dent with a tool, **never fingers**.
- **Never resume from the printer screen or the app.** The agent sends resume itself.
- Keep a hand near the printer screen. If the bed homes, rises toward the nozzle over the part,
  or anything collides, press **Stop** on the printer right away.

## 0. Setup (no printing, about 10 min)

1. Pull the latest code: `git checkout agent-pipeline && git pull`.
2. Slice the tray in Bambu Studio:
   - Import `parts/print/water-tray/water-tray.stl`, open side up (as imported).
   - Settings: PLA, 0.20 mm layer profile, **wall loops 3**, **bottom shell layers 6**, no supports,
     no brim, **Arc fitting off**, Timelapse **Traditional**.
   - File → Export → *Export plate sliced file*. Save it as `water-tray.gcode.3mf` in the repo folder.
3. Check that our code understands the file:
   ```bash
   python -m recovery.cli check-gcode water-tray.gcode.3mf
   ```
   **Pass:** the output ends with `looks usable`, and the preview PNG it saves looks like the tray
   (two outlines, no extra holes). If not, stop and send the output and the PNG.
4. Rehearse in simulation:
   ```bash
   python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
       --pause-at-layer 30 --dent 0 -10 3
   ```
   **Pass:** it recommends **CORRECT**. Press Enter to accept, and the simulated tray finishes and
   holds water (about 5.2 ml).

## How every printed test goes

1. Put a clean plate in. (If the screen shows FAILED from the last print, that's normal.)
2. **Start the print first** (send `water-tray.gcode.3mf` from Bambu Studio as usual).
3. Then, in the repo folder, start the agent with the command for that test (below). It prints
   `agent watching ... layers`.
4. At about **z = 6 mm (layer 30, half height)**, press **Pause on the printer screen**. Wait for
   the head to park.
5. The agent says `Printer paused after layer N of M`. Check that N matches the printer screen
   (±1). Write down both.
6. Do what the test says to the part, then follow the agent's prompts.
7. When the agent asks for approval, it shows what it found and its recommendation:
   `Recommended: X. Enter = accept, or [1] ... [2] ...`. Press Enter to accept, or type a number.
8. After the print, do the **water test**. Fill the tray to the brim and stand it on a dry paper
   towel for 10 min. Note any leaks and where they are.

## 1. Dent 1 mm → resume

```bash
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --dent-test \
    --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
```

- At the pause, gently press the **middle of the long front wall** inward, about 1 mm, with a flat
  tool. Don't break it.
- Press Enter in the agent, then type: `dent 0 -10 1`
- **Pass:** it recommends **RESUME**. Accept it, the printer resumes by itself and finishes, and
  the tray holds water.

## 2. Re-upload test (scrap tray, no dent) → restart

This is the risky one. It has never been tried on the P1S. Watch the bed the whole time it restarts.

**What happens and why.** A running Bambu job can't be edited, so "correct" doesn't change the
paused job. Instead the system:
1. stops the paused job;
2. uploads a new file with only the remaining layers (in this test, the original ones, unchanged);
3. starts that file as a new job.

The new file heats up and then continues at the current height (about 6 mm). It deliberately
doesn't home Z or level the bed, because the nozzle would hit the part. So it only works if the
printer still knows where the head is after the stop. This test checks exactly that, before any
corrected layers are involved.

Normal: the job stops, a new job starts on the screen, the printer heats, and the next layer
prints right on top of the old one. **Not normal (press Stop):** it homes, levels the bed, or
moves the bed up toward the nozzle over the part, or the first new layer prints in the air or
digs into the part.

```bash
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --resume-test \
    --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
```

- At the pause, **don't touch the part**.
- The agent shows `RESUME TEST` and recommends **RESTART**. Accept it. The printer stops, gets the
  original remaining layers as a new file and starts it.
- **Pass:** no Z homing over the part, no collision, it continues at the right height, there's no
  visible seam at 6 mm, and the tray holds water.
- If it fails, press Stop on the printer, then send everything (see below) and a video if you can.
  Do **not** do test 3 until this passes.

## 3. Dent 3 mm → correct

Same command as test 1 (`--dent-test`).

- At the pause, press the middle of the front wall inward about **3 mm**. Press Enter, then type
  `dent 0 -10 3`.
- It recommends **CORRECT**. **Before accepting**, open the picture whose path it prints
  (`correction_from_layer_N.png`). The new layers should start at the dented wall and step back
  to the original wall over about 10 layers.
- Accept. **Pass:** the printer stops, uploads and restarts as in test 2, the new layers step back
  to the original wall, and the tray holds water (about 5 ml).

## Send back after each test

- The full terminal output (copy and paste).
- The run folder `results/agent/<date-time>/`. Zip it. It has the log (`events.jsonl`,
  `summary.json`), the before/after photos and any G-code the agent made.
- The layer numbers from step 5 (agent vs. printer screen).
- Photos of the finished tray (side and top), plus the water-test result.
- Anything odd: noises, bed moves, a long wait, an error message.
