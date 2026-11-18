# Agentic monitoring (`recovery/agent/`)

Connects the printer to a decision-maker and back, with a person in the loop.

## Container dent test (current focus)

One hand-made change per print. The agent only reads the printer's status until someone pauses
it; then it is asked once. The person pauses, dents, and presses Enter in the agent, but does NOT
resume on the printer: the agent resumes, corrects or stops.

```
person pauses the P1S (screen or app; a pause planned in Bambu Studio also works)
  1. frame     agent notices the pause, saves a camera frame ("before")
  2. change    the person dents or pokes the tray, presses Enter in the agent
  3. frame     agent saves "after" and a picture of what changed
  4. measure   the person types what they did: 'dent X Y DEPTH' | 'poke X Y' | 'none'
               (part-centred mm; the camera can't measure the side-on P1S view yet)
  5. check     if we resume as planned, does the tray still hold water, and can every new
               layer be printed on the one below it?
  6. options   resume                         (still works)
               correct  new G-code for the rest: start on the dented shape, return to the
                        original wall over ~10 layers (only offered if resuming would fail;
                        designed by the search, or proposed by the LLM and checked: --plan-fix)
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
# real P1S: start the print and the agent; around half height (z = 6 mm) pause from the printer
# screen; the agent takes over from there
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --dent-test \
    --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
```

Tray coordinates: centre (0, 0); front wall middle `0 -10`; right wall middle `15 0`; walls 1.2 mm.

### Test order on the printer

Step-by-step guide for the person at the printer: [docs/printer-agent-test.md](../../docs/printer-agent-test.md).

| Phase | Run | Passes if |
|---|---|---|
| 0 · setup | `connection-test` (add `--commands` during a print), `check-gcode`, one `--sim` rehearsal | all connection steps PASS; "looks usable" |
| 1 · decisions | dent tests: clean, 1 mm, 3 mm, poke; accept **resume or stop only** | the agent notices the pause, frames saved, recommendations match the table, the agent's resume/stop commands work, water test agrees |
| 2 · mechanism | `--resume-test` on a scrap tray, no dent | no homing over the part, no collision, no seam, holds water |
| 3 · correction | 3 mm dent, approve **correct** | new layers step back to the original wall, holds water, ~5.2 ml |

Each phase starts only after the one before it passes.

### Before the first real dent test

- **Check the layer number at the pause.** The agent counts the layer the printer reports as
  not yet finished (a pause from the screen usually lands mid-layer). Compare the layer the agent
  prints at the prompt with the printer screen.
- **Don't resume on the printer.** If someone does, the agent notices, sends nothing and offers
  "continue" instead of "resume"; the next layers are then already printing over the dent.
- **"correct" depends on stop → upload → start working on the P1S.** On 2026-11-17 the stop and
  upload worked and the file printed fine when started by hand. The P1S ignores the laptop's start
  command (`gcode_file`), so after 2 min the agent asks the person to start it from the screen; that
  is the accepted way for now. Prove it first with the **resume test**: a scrap tray, no dent; at the pause the agent sends the
  *original* remaining layers through stop → upload → start. Watch the bed: on the P1S the bed is
  Z (it drops for each layer and for travel moves), so if it homes Z or rises toward the nozzle
  over the part, stop the printer. Pass = the tray finishes with no visible seam and holds water.
  Until it passes, choose resume or stop at the prompt.

  ```bash
  python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --resume-test \
      --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE     # then pause on the screen
  ```
- **Inspect a correction before approving it.** The agent saves the exact G-code it would send
  (`correction_from_layer_N.gcode`) and a picture of the dented layer and the next new layers
  (`.png`) in the run folder, and prints both paths at the prompt. The correction G-code fills
  the middle of thin walls, retracts before each travel and drops the bed 0.4 mm while
  travelling; temperatures come from the sliced file (bed: the plate it was sliced for).
- **The camera.** The P1S camera looks across the bed from the front-left, almost horizontally,
  and the bed moves down while the nozzle stays at a fixed height. So the top layers always sit at
  the same place in the image, seen from the side. Everything works on that view as it is; there
  is no top-down view. At a dent test the dent detector compares the frames before and after the
  change (`--measure sideview`, the default). It cannot measure depth: a dent counts as 1.5 mm.

## Watching every layer

```
every layer   camera frame -> compare with the previous frame and the plan -> alarm? -> pause -> decide -> approve -> act
```

The failure detector (`recovery/models/p1s_layer.py`, `--layer-model runs/p1s_layer`) says `ok` or which
failure the print looks like: collapse, warp, shift, under-extrusion, detach. It is on whenever that
model is there, in `--sim` and on the real printer alike.
- Detach and shift: stop (nothing sensible can be printed on top).
- Anything else, the first time: the camera cannot see the new shape well enough to say the part
  fails, so the recommendation is resume; a person approves.
- After that it only alarms again if the problem area grows (more than 1.5x the accepted one + 30 px),
  and then recommends stop.
- If the camera cannot check a layer (no frame, the part is not where the plan says) the layer is
  logged as unchecked and the print goes on.

```bash
python -m recovery.cli agent --part parts/print/two-hole-block/two-hole-block.yaml \
    --sim --dent 15 -5 1.0 --at-layer 29
```


## Deciders (the LLM slot)

- `rules` (default): resume if every feature passes; otherwise correct if a correction passes;
  otherwise stop. Stop if the part detached.
- `openai`: an OpenAI model (`--openai-model`, default `gpt-4.1-mini`; `gpt-5.4-mini` also works).
  Needs `OPENAI_API_KEY`, in the environment or in a `.env` file in the folder you run from
  (`OPENAI_API_KEY=sk-...`; `.env` is git-ignored, never commit it). Falls back to the rules.
- `local`: a local Qwen through Ollama (default `qwen3:1.7b`), or any OpenAI-compatible server
  (`--endpoint`, `--local-model`). Falls back to the rules.
- `claude`: needs `ANTHROPIC_API_KEY` (`--claude-model`). Falls back to the rules.
- `auto`: Claude if its key is set, then OpenAI if its key is set, then local Qwen, then the rules
  (each one stands in when the one before fails).
- In code: `LLMDecider(complete, fallback=...)` with any `complete(system, user) -> text`.

The model sees the measurement, both function checks (resume, and the correction if there is
one), the filament cost and the allowed actions, and must answer `{"action": ..., "reason": ...}`
with an allowed action. Anything else falls back to the next decider, and the log's `source` says
so (e.g. `local:qwen3:1.7b->rules`). The model never talks to the printer.

### The LLM plans the correction (`--plan-fix`)

With a language-model decider (`--plan-fix auto`, the default) the model also designs the correction:
it proposes how many layers the new layers take to return to the original wall (or keeps the dented
shape), with a reason. Our code builds that plan (Ishaan's contour fit) and checks it with the same
function checks (Sri's spec); a failing plan goes back to the model with its failure reasons, up to
3 times. If no proposal passes or the model is unavailable, the automatic search (fade 10, 15, 30,
keep) is used and the log says `->search`. `--plan-fix search` keeps the search with an LLM decider.
Code: `recovery/agent/planner.py`; each run saves `planner_transcript.json`.

In simulation `gpt-4.1-mini` planned a passing fix on its first proposal for 3 mm (10 layers), 4 mm on
the right wall and 5 mm (30 layers). For a 3 mm dent the checks pass any fade of 3 layers or more.

### The LLM writes the G-code (`--plan-fix gcode`)

The model plans the fix (as above, at most 15 changed layers) and then writes the G-code of every
changed layer itself, one layer per call, all layers in parallel (`recovery/agent/gcode_writer.py`):
- **It gets:** the layer height, the layer's toolpaths (the planned wall loops and gap fills in bed
  coordinates), line widths, `e_per_mm` from the extrusion formula, speeds and the recipe for
  retract / lift / travel / lower / prime. **It writes:** every line of G-code, including every E value.
- **Checked per layer:** only G0/G1, Z only at the layer height or lifted 0.4 mm, moves inside the
  part's area, E per mm within 15% of the formula, retracts balanced, total E within 10%. Then the
  layer is read back with our parser and must match the planned shape (95%). A failing layer goes
  back to the model with the problems (3 tries), then our writer writes that layer.
- **Checked as a whole:** the read-back layers must pass the function checks, or our writer's file is used.
- **The file:** resume header + the model's layers + the sliced file's later layers copied unchanged
  (cut before its end G-code; if that is not possible, our writer writes them) + our footer.
- **Models:** `--gcode-model` (default `gpt-5.4-mini`) writes; the decider model (default
  `gpt-4.1-mini`) still plans and decides. gpt-4.1-mini could not write it: every layer had wrong E values.

3 mm tray dent in simulation (2026-11-18): `gpt-5.4-mini` wrote all 15 changed layers, each passing on
the first try; read back they match the plan at 99.6-99.8% and the tray holds water. 17 calls,
25k tokens in / 30k out, about $0.15, 27 s.

### Tokens and cost

Every model call (decide, plan, write G-code) is counted: the approval summary and the end of the
run print e.g. `LLM: 17 calls (plan 1, write G-code 15, decide 1), 25,286 tokens in / 29,690 out, $0.1525`,
and the run folder has `usage.json` (per call and per step). Prices: `recovery/agent/usage.py`.

### OpenAI (tested 2026-11-17)

```bash
python -m recovery.cli llm-check --decider openai          # 2/2 right, about 1-2 s per call
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
    --pause-at-layer 30 --dent 0 -10 3 --decider openai
```

In simulation `gpt-4.1-mini` chose the same as the rules on all four tray dent tests: clean and
1 mm -> resume, 3 mm -> correct, poke-through -> stop. The model only chooses among the offered
actions and gives a reason; it never writes G-code or talks to the printer.

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
