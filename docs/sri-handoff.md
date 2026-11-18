# Handoff for Sri: three tasks for the LLM pipeline

For: Srivathsan (and the coding AI he gives this to). From: Sohum, 2026-11-18.
Everything here is on `main`. Each task is self-contained and can be done in its own branch + PR.

## Context (read first)

The water-tray dent test runs on the real P1S. A person pauses the print at layer 30, dents the
front wall, and types `dent X Y DEPTH`. The agent then decides what to do and a person approves:

```
pause -> measure (typed) -> check (your geometry/spec.py checks) -> plan the fix -> write G-code -> decide -> approve -> upload
```

What the LLM does now (all on `main`):
- **Decide** resume / correct / stop: `--decider openai` (default model `gpt-4.1-mini`).
- **Plan the fix** (`recovery/agent/planner.py`): it proposes how many layers the new layers take to
  return to the original wall. Your checks verify it; a failing plan goes back to the model with
  the reasons (3 tries); otherwise the old automatic search runs.
- **Write the G-code** of the changed layers (`--plan-fix gcode`, `recovery/agent/gcode_writer.py`):
  one layer per call (about 15 layers for a 3 mm dent), model `gpt-5.4-mini`. Each layer is
  safety-checked, read back with our parser and compared with the planned shape; then the whole
  fix is function-checked with your checks. A layer that fails 3 times is written by our own writer.
  The layers above the fix are copied unchanged from the sliced Bambu Studio file.
- **Tokens and cost** of every call: `recovery/agent/usage.py`, saved as `usage.json` per run.

Results in simulation (3 mm dent): `gpt-5.4-mini` wrote 15/15 layers correctly on the first try;
read back they match the plan at 99.6-99.8% and the tray holds water; about $0.15 and 27 s per run.
`gpt-4.1-mini` got every E value wrong (all caught by the check).

Try it (needs `OPENAI_API_KEY` in a `.env` file in the repo folder; it is git-ignored, never commit it):

```bash
python -m recovery.cli agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
    --pause-at-layer 30 --dent 0 -10 3 --decider openai --plan-fix gcode
```

Run folder: `results/agent/<time>/` (`correction_from_layer_30.gcode`, `.png`, `events.jsonl`,
`planner_transcript.json`, `usage.json`).

Rules for all three tasks:
- Branch from `main`. Do not change current behaviour for anyone who does not opt in (defaults,
  existing benchmarks and their numbers stay the same).
- The full test suite must pass: `python -m pytest -q tests` (about 3 min, 409 pass today). Add tests
  for what you build.
- Never print or commit the API key.

---

## Task 1 (most useful): a real G-code safety check, `gcode_check`

**Why.** The LLM's G-code goes to a real printer. Today a quick stand-in decides whether a layer is
safe: `check_layer` in `recovery/agent/gcode_writer.py:97`. It works, but it was written fast and
covers only the basics. Your check replaces it.

**Interface** (same as the stand-in, so swapping it in is one line at `gcode_writer.py:218`):

```python
# new file, e.g. recovery/geometry/gcode_check.py
def gcode_check(text: str, layer: Layer, jobs, layer_h: float, origin) -> List[str]:
    """Problems in ONE layer of G-code; empty list = safe to print.
    text    the layer's G-code lines (no layer-change markers; those are added by the writer)
    layer   the planned layer (Layer.z, Layer.region in part-centred mm)
    jobs    the planned toolpaths: list of (feature, polyline Nx2 part-centred mm, line width mm),
            from gcode_writer.layer_jobs()
    origin  (x, y) bed position of the part centre; G-code coordinates = part-centred + origin"""
```

Each problem must be one short sentence naming the line, e.g. `line 12: Z5.800 is neither the layer
height 6.200 nor lifted 6.600`. The problems are sent back to the LLM so it can fix its layer.

**The rules the LLM is told to follow** (`SYSTEM` in `gcode_writer.py:41`). Printer state: absolute
XYZ (G90), relative E (M83), hot nozzle. For each path, in order:

```
G1 E-0.8 F1800            retract
G1 Z<z+0.4> F600          lift
;TYPE:<feature>
G0 X.. Y.. F7200          travel to the path's first point
G1 Z<z> F600              lower
G1 E0.8 F1800             prime
G1 X.. Y.. E<len*e_per_mm> F1800   for each next point
```

with `e_per_mm = width * layer_h / (pi * (1.75/2)^2)` (`recovery/gcode.py`, `_e()`).

**What the stand-in checks today** (keep all of it):
- only G0/G1; only the letters X Y Z E F;
- F between 300 and 12000;
- Z only at the layer height or lifted by 0.4 mm;
- every X/Y inside the part's bounds + 2 mm;
- no extrusion while lifted, no extrusion on G0, no retract while moving;
- E per mm within 15% of the formula (over the range of widths in the layer);
- retracts and primes balance (net within 0.05 mm), each at most 1 mm;
- total E within 10% of the planned paths' total.

**What it misses (please add):**
1. **Every planned path is printed, and nothing else:** each `jobs` polyline is traced (within about
   0.1 mm), in a sensible order, with no extra extrusion off the planned paths. Today only the total
   E and the read-back shape catch this.
2. **E per mm per path:** compare each move with the width of the path it belongs to, not with the
   layer's whole range of widths.
3. **Travel safety:** no travel at layer height across the part (it must be lifted first); a retract
   before every travel longer than about 1 mm; a prime after it.
4. **Continuity:** no jumps inside a path (an extrusion move must start where the last one ended).
5. **Speeds by move type:** extrusion F around 1800, travel around 7200, Z moves around 600 (allow a band).
6. **Whole-file check** (new function, e.g. `gcode_file_check(text, layers, origin)`) for the
   uploaded file (`correction_from_layer_N.gcode`):
   - layer heights strictly increase by one layer height;
   - `M73 L<n>` layer numbers run in order;
   - no G28 / G29 / bed leveling anywhere;
   - temperatures only in the header;
   - the footer parks the head.
   It should pass on the files our own writer makes, so check those too.

**Done when:**
- it accepts the output of our own writer, and the correct reply built in the tests
  (`_protocol_gcode` in `tests/test_agent.py`);
- it rejects each bad case in `test_layer_check_names_each_unsafe_line`, plus a test for each new rule;
- `gcode_writer.py` calls it instead of `check_layer`, and the simulated run above still gets 15/15.

---

## Task 2: a maximum sideways step per layer in the overhang check

**Why.** Your overhang check (`recovery/geometry/features.py:39-59`, used by `geometry/spec.py:53`
and by the agent's support check, `recovery/agent/tools.py:97`) lets a wall step up to about 1.6 mm
sideways per layer before it counts any overhang:
- `_unsupported(tol=0.6)`: the first 0.6 mm past the layer below is free;
- `excess_overhang(r=0.5)`: slivers narrower than 1 mm are ignored;
- then the area must exceed `OVERHANG_AREA = 6 mm²`.

So for a 3 mm dent, a fix that returns to the wall over only 3 layers (1 mm per layer) passes. With
a 1.2 mm wall, most of each new layer would then print over air. Common FDM practice is about
0.2-0.35 mm per 0.2 mm layer (45-60°). Now that the LLM chooses the fix (and writes its G-code),
this limit decides what it is allowed to do.

**What to add:**
- A separate check: the largest distance any part of a layer sticks out past the layer below, only
  where the design itself does not overhang. Compare it with a limit, e.g. a new spec key
  `max_step_mm`. Make it a separate feature in the report, with a clear message such as
  `layer 33 steps 1.0 mm past the layer below (limit 0.35)`.
- **Opt-in:** off unless the spec sets `max_step_mm`, so your benchmarks and their numbers don't
  change. We would set it in `parts/print/water-tray/water-tray.yaml`.
- The value: start with 0.35 mm. Omar's test 3 printed a 3 mm fix over 10 layers (0.3 mm per layer);
  his photos of the dent area will tell us whether that printed cleanly. Adjust after that.

**Done when:**
- with `max_step_mm: 0.35`, the 3 mm dent fix over 3 layers fails and over 10 layers passes;
  `PrinterTools.try_correction(fade)` makes both (see `test_printed_layers_are_not_judged_for_hanging_over_air`
  in `tests/test_agent.py` for how to set it up);
- without the key, every existing test and benchmark gives the same results as before.

---

## Task 3: spec gen with OpenAI, run on the tray

**Why.** `recovery/specgen.py` (your spec author) only supports Claude, and we have no Anthropic key,
so it has never run with an LLM. We have an OpenAI key.

**What to do:**
1. Add an OpenAI path to `author_spec` (`specgen.py:300`). It already accepts a `client` with a
   `messages.create(...)` method (`specgen.py:339-353`). Either write a small adapter around
   `recovery/agent/deciders.openai_complete` (it takes `api_key=` and `usage=`), or add a
   `provider="openai"` option. Read the key with `recovery/agent/cli.load_dotenv()`.
2. Add `--provider openai` and `--model` to the `specgen` command (`recovery/cli.py`, around line 65).
   Keep the existing fallback to the heuristic.
3. Track tokens and cost with `recovery/agent/usage.Usage` and print `usage.line()` at the end.
4. Run it on the tray and compare it with the hand-written spec:
   ```bash
   python -m recovery.cli specgen --part parts/print/water-tray/water-tray.yaml \
       --describe "a small tray that must hold water without leaking" --provider openai --out /tmp/tray-llm.yaml
   ```
   Compare with `parts/print/water-tray/water-tray.yaml`:
   - capacity range (hand-written: 4200-6300 mm³);
   - base height (1.2 mm);
   - any extra rules it adds (wall thickness, loads);
   - whether it verified, and in how many rounds.
5. Write the comparison in a few lines (a table is fine) in `docs/specgen-tray.md`. It goes in the report.

**Done when:**
- the command above produces a verified spec, or a clear reason why not;
- the comparison is written;
- running without a key still falls back to the heuristic, as before.

---

## Where things are

| What | File |
|---|---|
| Agent loop (pause → decide → act) | `recovery/agent/loop.py` |
| Checks per feature, the agent's support check | `recovery/agent/tools.py` (`function_report`, `_support`, `try_correction`) |
| LLM plans the fix | `recovery/agent/planner.py` |
| LLM writes the G-code, stand-in safety check, copying the original layers | `recovery/agent/gcode_writer.py` |
| Model clients, rules decider | `recovery/agent/deciders.py` |
| Tokens and cost | `recovery/agent/usage.py` |
| Your checks | `recovery/geometry/spec.py`, `recovery/geometry/features.py` |
| G-code writer and parser | `recovery/gcode.py` (`write_layers`, `gcode_to_layers`, `_e`) |
| Resume file header and footer, upload | `recovery/printer/bambu.py` (`resume_header`, `FOOTER`) |
| Tests | `tests/test_agent.py` (bottom: planner and G-code writer tests) |
| Printer test guide (Omar) | `docs/printer-agent-test.md` (tests 4 and 5 are the LLM ones) |
