# Agent test on the P1S (water tray)

**Goal:** prove the agent pipeline works on the real printer:
- it notices the pause, checks the part, recommends an action and waits for approval;
- then the printer resumes, or takes an uploaded G-code file and continues printing.

**Not needed:**
- **Dent model:** you type what you did to the part (`dent X Y DEPTH`), the default on the real printer.
- **Calibration:** without `calib.npz` the agent uses an approximate camera; this only affects the saved photos.
- **LLM for tests 0-3:** they use the built-in rules (`--decider rules`, the default). Test 4 uses an
  OpenAI model; its setup is in test 4.
- **The printer starting the new file by itself:** the P1S ignores the laptop's start command
  (confirmed 11-17). When the agent asks, start the file from the printer screen. That is expected.

Do the tests in order. Start a test only once the one before it has passed.

| # | Test | Prints | Proves | Status |
|---|---|---|---|---|
| 0 | setup | none | the sliced file is understood; the agent runs in simulation | passed 11-16 |
| 1 | 1 mm dent → **resume** | 1 tray | the agent notices the pause, checks, recommends, and its resume works | passed 11-17 |
| 2 | re-upload test, no dent → **restart** | 1 tray | stop → upload → start works on the P1S (the step behind "correct") | passed 11-17 (file started from the screen) |
| 3 | 3 mm dent → **correct** | 1 tray | the whole chain: corrected layers uploaded and printed, tray holds water | rerun 11-17: results pending |
| 4 | 3 mm dent, **LLM** decides and plans the fix → **correct** | 1 tray | the LLM chooses the action and designs the correction; our checks verify it | new |
| 5 | 3 mm dent, the **LLM also writes the G-code** → **correct** | 1 tray | the corrected layers' G-code is written by the LLM, checked, and prints | new |

**Short on time?** Tests 3, 4 and 5 can be done back to back in one session (3 trays). Test 5 alone
covers everything 3 and 4 do, plus the LLM writing the G-code.

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

**Status (11-17):** passed. The stop and the upload worked, and the uploaded file printed fine when
started from the printer screen. The P1S does not start it from the laptop, so starting it from the
screen is the normal step now.

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
- The printer stops. The agent prints `uploaded adapted_<number>.gcode; starting it`.
- After about 2 min it says `The printer did not start ... by itself` (expected):
  - start that exact file from the printer screen (SD card files);
  - the agent continues as soon as the print starts (15 min is only the longest it waits);
  - then it prints `... is printing`.
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
- Accept. The printer stops and the file is uploaded as in test 2. Start the file from the printer
  screen when the agent asks.
- **Pass:**
  - the new layers step back to the original wall;
  - the head parks at the end;
  - the tray holds water (about 5 ml).

## 4. Dent 3 mm, the LLM decides and plans the fix → correct

Same as test 3, but an OpenAI model makes the two decisions the rules made before:
- **plans the fix:** how many layers to step back to the original wall over (or keep the dent).
  Our code builds that plan and Sri's checks test it; a failing plan goes back to the model with the
  reasons (up to 3 tries). The model never writes G-code.
- **decides:** resume, correct or stop, with a reason.

Setup (once):
- Create a file named `.env` in the repo folder with one line: `OPENAI_API_KEY=...`
  (get the key from Sohum privately; never commit or share it).
- Check it: `python -m recovery.cli llm-check --decider openai`. **Pass:** `2/2 right`.

```bash
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --dent-test \
    --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE --decider openai
```

- At the pause, press the middle of the front wall inward about **3 mm**, press Enter, type `dent 0 -10 3`.
- Check the summary before accepting:
  - `Planned by openai:...` with the model's reason and how many proposals were checked;
  - `Agent (openai:...): CORRECT because ...`;
  - the picture `correction_from_layer_N.png`: the new layers start at the dent and step back to the
    wall over the number of layers the model chose.
- Accept, then start the file from the printer screen when the agent asks.
- If it says `openai...->rules` or `->search`, the model could not be used and the rules took over:
  the print is still fine, but send the output (`llm_transcript.json`, `planner_transcript.json`).
- **Pass:** as test 3, and the run folder has `planner_transcript.json`.

## 5. Dent 3 mm, the LLM also writes the G-code → correct

Same as test 4 (same `.env`), with one more option. The LLM plans the fix, then writes the G-code of
every changed layer itself (about 15 layers). Each layer is safety-checked and read back before
anything is sent; the layers above the fix are copied unchanged from your sliced file.

```bash
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --dent-test \
    --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE --decider openai --plan-fix gcode
```

- At the pause: dent the front wall about **3 mm**, press Enter, type `dent 0 -10 3`.
- Writing takes about 30 s (the printer stays paused).
- Check the summary before accepting:
  - `G-code: 15/15 changed layers written by openai:gpt-5.4-mini; the other 15 layers copied from the sliced file`;
  - note it if it says `by our writer` for some layers, or `our writer (no usable sliced file)`;
  - the `LLM: ... calls ... $...` line (the cost so far);
  - the picture `correction_from_layer_N.png`, as in test 4.
- Accept, start the file from the printer screen when asked, and **watch the first new layers**:
  this is the first time G-code written by the LLM runs on the printer. Press Stop if the nozzle
  digs in, prints in the air or leaves the part.
- **Pass:** as test 3, and the run folder has `usage.json` and `planner_transcript.json`.
- Also send `water-tray.gcode.3mf` (your sliced file) so we can check the copying of the top layers offline.

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
