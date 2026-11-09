# Agentic monitoring (`recovery/agent/`)

Connects the printer to a decision-maker and back, with a person in the loop:

```
every layer   P1S (MQTT status + camera) or simulator
  1. sense     wait for the layer to finish, grab a frame            tools.next_layer / observe
  2. measure   frame -> printed outline vs the sliced plan           (existing segmenter/detector)
  3. check     if we continue, does each feature still work?         tools.function_report
               (each hole, the container, support...; existing spec rules, one at a time)
  4. alarm?    outline > ~1 mm off plan, OR a feature breaks even though the outline looks close
  5. pause     automatically (safe on the P1S); --no-auto-pause to ask first
  6. decide    decider recommends resume/continue or stop, with a reason     deciders.py
  7. approve   a person accepts (Enter) or picks another option              gate.py
  8. act + log MQTT command; events.jsonl, frames/, llm_transcript.json      loop.py
```

A deviation that was accepted (resumed) does not re-alarm on every later layer; only a failing
function check or a deviation that grows by more than 0.5 mm does.

## Deciders (the LLM slot)

- `rules` (default): continue if every feature passes, stop otherwise; stop if detached.
- `local`: a local Qwen through Ollama (default `qwen3:1.7b`), or any OpenAI-compatible server
  (`--endpoint`, `--local-model`). Falls back to the rules.
- `claude`: needs `ANTHROPIC_API_KEY` (`--claude-model`). Falls back to the rules.
- `auto`: Claude if a key is set, otherwise (or if it fails) local Qwen, otherwise the rules.
- In code: `LLMDecider(complete, fallback=...)` with any `complete(system, user) -> text`.

Every fallback is visible: the log's `source` says e.g. `local:qwen3:1.7b->rules`.

### Local Qwen setup (until Claude credits arrive)

Not yet tested on a real machine: whoever sets it up first, please run `llm-check` and report.

```bash
brew install ollama          # macOS; Linux/Windows: https://ollama.com/download
ollama serve                 # leave running (or: brew services start ollama)
ollama pull qwen3:1.7b       # ~1.4 GB; qwen3:8b (~5 GB) is better if the laptop has 16 GB
python -m recovery.cli llm-check --decider local      # two sample alarms; expect 2/2 right
python -m recovery.cli agent --part parts/print/two-hole-block/two-hole-block.yaml \
    --sim --dent 15 -5 1.0 --at-layer 29 --decider local
```

`llm-check` prints each answer, who made it (the model, or `->rules` if it fell back) and how
long it took. If the model often falls back or answers wrong, use `qwen3:8b` or `--decider rules`.

The model sees the measurement, the per-feature check, the filament cost and the allowed actions,
and must answer `{"action": ..., "reason": ...}` with an allowed action. Anything else falls back
to the rules and is marked `llm->rules` in the log. The model never talks to the printer.

## Run it

```bash
# rehearse a test in software (no printer): dent next to a hole of the two-hole block
python -m recovery.cli agent --part parts/print/two-hole-block/two-hole-block.yaml \
    --sim --dent 15 -5 1.0 --at-layer 29
# (--poke X Y for a poke-through, --auto-approve to skip the prompt; coordinates are
#  part-centred mm: the block's holes are at x = -15 and +15)

# real P1S (print started from Bambu Studio; LAN-only + Developer Mode; camera calibrated)
python -m recovery.cli agent --part parts/print/two-hole-block/two-hole-block.yaml \
    --gcode two-hole-block.gcode.3mf --ip IP --serial SERIAL --code CODE --calib calib.npz \
    --decider auto
```

## What is verified

Simulated rehearsal of all three test parts (clean, cosmetic dent, function-breaking dent,
3 seeds each): 30/30 correct decisions, alarms at the dent layer, no false alarms on clean
prints, 30-160 ms per layer (`tests/test_agent.py`). The simulated camera is easier than the
real one, so this does not show the real P1S works yet: run `watch` on clean prints first and
check the false-alarm rate before letting `agent` pause a real print.
