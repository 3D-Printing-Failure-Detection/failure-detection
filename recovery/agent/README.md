# Agentic monitoring (`recovery/agent/`)

Connects the printer to a decision-maker and back, with a person in the loop.

## Container dent test (current focus)

One hand-made change per print, at a pause planned in Bambu Studio. The agent is asked once.

```
P1S pauses before layer N (pause added in Bambu Studio)
  1. frame     agent saves a camera frame ("before")
  2. change    the person dents or pokes the tray, presses Enter
  3. frame     agent saves "after" and a picture of what changed
  4. measure   the person types what they did: 'dent X Y DEPTH' | 'poke X Y' | 'none'
               (part-centred mm; the camera can't measure the side-on P1S view yet)
  5. check     if we resume as planned, does the tray still hold water, and can every new
               layer be printed on the one below it?
  6. options   resume                         (still works)
               correct  new G-code for the rest: start on the dented shape, return to the
                        original wall over ~10 layers (only offered if resuming would fail)
               stop                           (no option works)
  7. decide    rules / local Qwen / Claude recommend one, with a reason
  8. approve   the person presses Enter, or picks another option by number
  9. act + log MQTT resume / stop, or stop -> upload -> start for correct; events.jsonl + frames
```

Rehearsed on the simulated printer (`tests/test_agent.py`), checked against the actual printed
shape:

| Change at layer 30 | Resume as planned | Agent | Finished tray |
|---|---|---|---|
| none, 1 mm dent | works | resume | holds water |
| 3-5 mm dent | the next layer hangs over 11-15 mm² of air | correct (back over 10 layers) | holds water, ~5.2 ml |
| poke-through | leaks | stop | stopped at 50% |

```bash
# rehearse (no printer)
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
    --pause-at-layer 30 --dent 0 -10 3          # or --poke 5 -10, or nothing for a clean run
# real P1S: in Bambu Studio add a pause at layer 31 (the first layer above z = 6 mm), export, print,
# then start the agent; it waits for the pause, then walks you through the steps above
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --dent-test \
    --pause-at-layer 30 --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
```

Tray coordinates: centre (0, 0); front wall middle `0 -10`; right wall middle `15 0`; walls 1.2 mm.

### Before the first real dent test

- **Check the layer number at the pause.** The agent assumes Bambu Studio's pause stops *before*
  the chosen layer starts and that the printer then reports that layer's number. Compare the
  layer the agent prints in its summary with the one set in Bambu Studio.
- **"correct" depends on stop → upload → start working on the P1S**, which has never been tried.
  Test it once on a scrap tray with no dent before relying on it. Until then, choose resume or
  stop at the prompt.
- **The side-on camera.** The P1S chamber camera looks across the bed from the front, almost
  horizontally, and the bed moves down while the nozzle stays at a fixed height. So the top layers
  always sit at the same place in the image, but are seen from the side. The existing camera
  measurement (`--measure camera`, `--monitor on`) re-projects frames to a top-down view and is
  not usable on this view, which is why the dent test asks the person. The before/after frames
  are saved so a side-view measurement can be built from real data. Two notes for that work:
  - `BambuPrinter.camera_image` re-projects onto the plane at the layer's height above the bed.
    On the P1S the bed moves down by the same amount, so the top layer stays on the plane the
    camera was calibrated on (if calibrated with the bed at the first layer). That needs fixing
    in the calibration work, not here.
  - A dent on a wall seen edge-on (the tray's right wall, `15 0`) changes the outline the camera
    sees; a dent on the wall facing the camera mostly changes shading.

## Continuous monitoring (top-down camera)

```
every layer   sense -> measure outline -> check each feature -> alarm? -> pause -> decide -> approve -> act
```

Two alarms: the outline is ~1 mm off plan, or a feature breaks while the outline looks close
(a 1 mm dent next to a hole narrows it below bolt size without tripping the outline alarm). An
accepted deviation does not re-alarm on every later layer. On by default in `--sim`; on the real
P1S only with a top-down view (`--monitor on --calib calib.npz`).

```bash
python -m recovery.cli agent --part parts/print/two-hole-block/two-hole-block.yaml \
    --sim --dent 15 -5 1.0 --at-layer 29
```

Rehearsal of all three test parts (clean, cosmetic dent, function-breaking dent, 3 seeds each):
30/30 correct decisions, no false alarms on clean prints.

## Deciders (the LLM slot)

- `rules` (default): resume if every feature passes; otherwise correct if a correction passes;
  otherwise stop. Stop if the part detached.
- `local`: a local Qwen through Ollama (default `qwen3:1.7b`), or any OpenAI-compatible server
  (`--endpoint`, `--local-model`). Falls back to the rules.
- `claude`: needs `ANTHROPIC_API_KEY` (`--claude-model`). Falls back to the rules.
- `auto`: Claude if a key is set, otherwise (or if it fails) local Qwen, otherwise the rules.
- In code: `LLMDecider(complete, fallback=...)` with any `complete(system, user) -> text`.

The model sees the measurement, both function checks (resume, and the correction if there is
one), the filament cost and the allowed actions, and must answer `{"action": ..., "reason": ...}`
with an allowed action. Anything else falls back to the next decider, and the log's `source` says
so (e.g. `local:qwen3:1.7b->rules`). The model never talks to the printer.

### Local Qwen setup (until Claude credits arrive)

Not yet tested on a real machine: whoever sets it up first, please run `llm-check` and report.

```bash
brew install ollama          # macOS; Linux/Windows: https://ollama.com/download
ollama serve                 # leave running (or: brew services start ollama)
ollama pull qwen3:1.7b       # ~1.4 GB; qwen3:8b (~5 GB) is better if the laptop has 16 GB
python -m recovery.cli llm-check --decider local      # two sample alarms; expect 2/2 right
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
    --pause-at-layer 30 --dent 0 -10 3 --decider local
```
