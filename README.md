# Functional recovery from 3D-print failures

A feedback loop that **stress-tests a part before printing, images every layer while printing,
and when something goes wrong measures the as-printed shape and decides to continue, adapt the
remaining layers, or scrap**, so the finished part still does its job even if it looks different.
Target hardware: **Bambu Lab P1S + Bambu Studio**. Based on the [Project 2 plan](#project-plan)
(also in [docs/project-plan.pdf](docs/project-plan.pdf)).

> **Status: research prototype.** The classical loop now runs against three MuJoCo
> printer rigs, with paired failure evaluations and regression tests. See the
> [review and generalization report](docs/mujoco-report.md) for measured results,
> corrected errors, reproduction commands, and simulation limits. Trained-model
> development results are recorded separately below. Real-printer recovery remains
> unvalidated.

The repository-root [project report (PDF)](report.pdf) explains the system,
current implementation, experimental results, limitations, and next steps.
Its source is `docs/report.tex`; rebuild it with `python3 tools/build_report.py`.

## MuJoCo across printer configurations

The optional backend executes extrusion waypoints on Cartesian, ideal CoreXY,
and moving-bed rigs. The same perception and policy consume calibrated MuJoCo RGB
images on all three. These are illustrative configurations, not calibrated models
of commercial machines. Deposition and failure evolution remain explicit geometric
surrogates; this does not simulate molten plastic or adhesion.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-sim.txt
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m recovery.cli mujoco-eval --n 2 --seed 10 --workers 6 \
  --out results/mujoco-persistent
```

This images every monitored layer, includes clean controls and all four policies,
and leaves injected failures active after adaptation. Use a fresh output directory
for each run. The optimistic assumption in the older simulator can be measured
separately with `--clear-failure-on-adapt --methods ours`. Results report observed
in-place success separately from a hypothetical restart's filament cost. Raw runs,
summary tables, environment metadata, and a source digest are saved together.

On this machine, the environment and raw experiment outputs live under
`/mnt/data/ishaan/hri-workspace/failure-detection`; `.venv` is a local symlink.
On another machine, choose an appropriate disk for environments and results.

The [follow-up policy trial](docs/functional-policy-trial.md) tests three cumulative
changes: `measured` retains every measured layer and adds functional alarms;
`predictive` also forecasts persistent deposition error and compensates commands;
`functional` also ranks feasible plans by functional margin under geometric
uncertainty. These are experimental options; `ours` remains the default. Compare
them with `--methods ignore ours measured predictive functional`. The CLI demo
also accepts `--policy functional`, but its older simulator clears failures after
adaptation; use MuJoCo without `--clear-failure-on-adapt` to test persistence.

The [notebook-inspired constrained policy](docs/notebook-recovery-trial.md),
available as `guarded`, separates function, executable recovery and intervention
cost. It retains a passing continuation, checks the remaining replan budget and
printer bounds, and builds corrections using an explicit per-layer assumption.
Compare it with `--methods ignore ours predictive functional guarded`. The
notebook's success probability remains an unknown input; a geometry margin is
not used as a probability. This policy also remains opt-in.

The [contour prediction trial](docs/contour-prediction-trial.md) adds the opt-in
`contour` policy. It matches inner and outer wall boundaries separately when
forecasting deposition and computing compensation, while retaining the guarded
policy's functional checks and action limits. Compare with
`--methods ignore ours guarded contour`; the demo also accepts `--policy contour`.
Use the persistent-failure MuJoCo evaluation for recovery evidence.

The [learned recovery experiment](docs/neural-recovery-plan.md) trains camera and
temporal neural networks and runs a free local Qwen model in the decision loop,
without an API key. Its optional `neural_risk` and `neural_risk_agent` policies can
intervene before a tight fit becomes unusable. The combined model and Qwen
completed 67/100 historical failure cases versus contour's 62/100, preserving its
successes with zero defective finishes instead of one. All 20 clean controls
passed. The complete first frozen comparison across 15 configurations (4,320 runs)
rejects that candidate: 126/240 successes versus contour's 128/240 and analytical
preventive recovery's 133/240, with five defective finishes versus three for the
strongest baseline. The revised fit policy reaches 13/20 validation successes
versus contour's 10/20 and the earlier neural policy's 12/20, preserving both
methods' passing cases with no defective finishes. A broader completed fit study
scores 78/120 for both neural variants versus 76/120 for analytical preventive
recovery, with no defective finishes and all 24 clean controls passed. It still
loses one analytical success, so it fails the preservation requirement. Qwen
makes 158 actual requests but adds no successes over the neural policy alone.
The new opt-in `--neural-hole-envelope` proposal restores that lost bracket and
completes the full fit grid at **79/120** with and without Qwen, versus analytical
preventive recovery's **76/120**. It preserves every analytical and previous-neural
success, with zero defective finishes and all 24 clean controls passed. Qwen makes
161 actual requests and ties neural-only outcomes. Historical and validation fit
checks also preserve the previous best results (33/50 and 13/20).
Camera regressions remain unresolved: a two-image network improves boundary error
but still invents two wall gaps. A matched dense-connectivity trial also retains
both gaps, despite removing false support and base alarms; its checkpoint is
rejected. Rebalancing rare thin-material training tiles fixes one gap but adds
eight false base alarms, so that model is also rejected. Adding the commanded
layer mask removes both false gaps, but leaves two false base alarms and one
false hard-thin layer; this model still fails. Longer training, verified through
higher-precision inference, reintroduces one gap and retains those extra alarms.
It is also rejected. Spatial-gradient training reintroduces two gaps, and an
actual recovery trial using classical bases with learned walls scores 2/4 versus
the previous camera policy's 3/4, so both changes are rejected. The completed
full-grid trial combines the improved learned fit forecast with analytical
containers: **136/240 functional completions**, with and without Qwen, versus
the matched analytical policy's **133/240** using the same current measurement
code. All of that baseline's successes are retained; both have one defective
finish and pass all 48 clean controls. The complete comparison has 5,184 audited
rows across 18 configurations. Qwen makes 499 actual
requests and adds no completions. These are consumed development cases; some
successes from other earlier policies are still lost. The older analytical
version had three defective finishes: the reduction to one also occurs without
learning, so it is not a demonstrated neural-model benefit. A separate 24-run
historical check shows that analytical
containers lose two earlier rescues; using the original learned camera as a
joint-forecast alternative retains both, with no defective finishes and all
clean controls passed. Qwen makes 11 calls and ties neural-only. That variant's
full-grid follow-up is complete: **141/240 successes but five defective finishes**,
versus the prior candidate's 136 successes and one defect. All active recovery
baselines' successes are retained; five ignore successes remain lost. The
5,760-row comparison covers 20 configurations. Qwen makes 527 actual calls and
ties neural-only. The extra five successes restore usable prints rather than
add genuine rescues; those remain 18. Four additional defects follow corrections
supported only by the learned camera despite failed uncertainty checks. An
opt-in guard now rejects such corrections. Its complete 42-row regression passes:
both variants retain all ten required fault successes and six clean controls,
stop the four extra defective prints, and retain the one pre-existing defect.
Qwen makes 60 calls and ties neural-only. All 195 repository tests pass. The
complete 576-row guarded rerun now records **141/240 successes and one defective
finish with Qwen**, retaining all 48 clean controls. Its 6,336-row comparison
preserves all 20 earlier configuration summaries. Qwen makes 523 actual calls.
Neural-only records 140/240 with one defect: its extra loss has an anomalous
camera/measurement trace before the fault and before any Qwen decision, so the
one-case difference is not an established agent benefit. The separate repeatability
diagnostic reproduces identical early frames and both variants succeed on full
replay; the original failure is retained and its cause remains unresolved. The complete
historical regression passes: both variants retain 67/100 successes with zero
defects and all 20 clean controls, including every prior successful case.
The fit-only comparison retains 33/50 successes with zero defects. Five ignore
successes remain lost on the broader development grid, and
untouched-test superiority is unproven.
Future evaluations record compact fingerprints of every camera image through
the first adaptation. Audits compare these prefixes across policies, including
separate result directories; final comparisons can require this evidence from
every configuration. Legacy results remain explicitly unverified for camera
input consistency. All four instrumentation checks pass on the anomalous cup
and a historical recovery: 268 paired frames match, with unchanged outcomes,
action sequences, physics and policy options.
The trend correction also passes its complete 46-row targeted regression:
both variants preserve all ten earlier fault successes, add one usable mug
completion, and pass all six clean controls with only the same existing defect.
Qwen makes 62 calls and ties neural-only. The 864-row full follow-up is complete:
**both variants finish 142/240 fault cases (59.2%), with one defective completion
and all 48 clean controls passing**. This exceeds the previous best count of 141
and retains every active recovery baseline success. Qwen makes 518 actual calls
but adds no completions over neural-only. Four ignore successes remain lost;
genuine rescues remain 18 versus matched analytical's 15. All 22 earlier summaries
are unchanged in the 7,200-row comparison. A reporting-only label repair resolves
an old/new source collision; the original failed report record and all simulation
results are preserved. The new analytical trend control regresses to 127/240
with three defects, so all stronger earlier analytical controls remain included.
Copies of all earlier policy sources now support camera traces for the final
comparison; all 25 configurations preserve their parsed policy options and
model paths. This preparation executes no simulations or fresh test cases.
Ten real checks across all four earlier runtime versions now preserve outcomes,
actions, measurements and inference counts, with 561 matching paired camera
frames. Completed results can also be stored losslessly: packing copied real
evidence preserves every restored byte, audit and comparison result exactly.
The historical follow-up is complete and **rejects the global trend correction**:
both variants fall to 64/100 successes with one defective finish, versus the
preceding guarded policy's 67/100 with zero defects. All 20 clean controls pass;
three earlier mug recoveries are lost. Qwen makes 190 calls and ties neural-only.
Fit-only remains 33/50 with zero defects. The next controller change must retain
the original executable recovery plans instead of delaying their interventions.
The opt-in `--boundary-trend-continuation-fallback` now preserves those original
plans. Only when all original choices fail does it test unchanged continuation
with damped trends, requiring both camera histories to pass the full spec.
All 16 paired runs pass: both variants preserve the three historical mug
recoveries with unchanged actions and add the usable mug completion. The next
42 preservation runs also pass: both variants retain ten required fault successes
and six clean controls, with no additional defects. All 3,345 common camera frames
match across the 58 runs. Fit forecasts and the physical specification remain
unchanged; no broader success-rate improvement is established yet.
The final 7,200-row command matrix is prepared and passes all 300 argument checks,
with identical frozen policy settings and model paths. It uses both existing Qwen
workers and keeps fresh seeds untouched until development and historical gates
pass, including resolution of every lost passing case. Exact export reconstruction
and verified compression keep the planned run within the storage budget.
Fixed averaging and asymmetric camera
training both fail their geometry gates and are rejected. A learned spatial
output layer on the frozen camera now passes the complete known 160-layer cup
diagnostic, removing its false gap/base/thickness alarms. However, its complete
2,040-frame validation misses ten real gaps versus the original camera's five,
despite reducing false gap alarms from 63 to 35. It fails that gate, and its
conditional recovery run is skipped; no new recovery gain is claimed.
A follow-up doubles the training penalty for invented material. Its matched
control reproduces exactly, but the candidate introduces two false wall gaps in
the full cup diagnostic, so it is also rejected before recovery evaluation.
A further 89-parameter nonlinear camera head slightly lowers tile boundary error
while slightly reducing boundary sign accuracy. Its linear control reproduces
exactly and both exports pass. It passes the complete known cup diagnostic,
but the completed 2,040-frame check misses eleven real gaps versus the original
camera's five. It is rejected before recovery evaluation.
This experiment does not change the policy in the completed benchmark.
A separate replay attributes a lost mug success to a false future connection
prediction, while both complete camera histories pass. An opt-in
`--contour-trend-shrinkage` discounts existing measurement uncertainty from
analytical trend magnitudes. Both candidate variants finish the mug without
adaptation, while both matched controls scrap it. Qwen makes three candidate
calls and ties neural-only. Its complete preservation regression passes, followed
by the 142/240 full comparison above. This added completion avoids an unnecessary
scrap; it does not add a genuine rescue over the preceding guarded policy.
New untouched seeds
are reserved for another final test;
these policies remain opt-in.
An optional camera model now accepts the preceding commanded shape alongside
the current command and two observed images. Its matched control reproduces
exactly; the new model lowers tile-validation boundary error from 0.0788 to
0.0758 mm. All four complete-print camera checks fail across 640 actual frames,
so this model is rejected before any recovery trial.
Existing camera checkpoints produce identical outputs on retained images.
The updated source passes 201 tests with three optional modules skipped, plus
20 Torch/export tests. Prediction improvements alone do not establish better
print success.
The experiment records real model calls, failed attempts and neural-only
ablations. It does not count an agent response or lower prediction error as a
successful print.
Screening all 26 distinct retained cameras finds no model that resolves either
known mug error. The nonlinear camera alone passes both local cup checks while
retaining six selected real defect detections. A separate visibility diagnostic
shows that the mug's thin wall loses closure in the rendered scene raster; the
cup retains its raw raster cavity but loses visible closure after projection.
Exact unions of predicted positive pixels resolve no additional model failures.
A matched training experiment gives scarce true thin-material pixels half
the training loss weight, with unchanged inputs and validation selection. Its
control reproduces exactly, but recall of those pixels falls from 35.2% to 29.7%
on validation tiles despite improving on training tiles. All four complete-print
diagnostics fail across 640 frames: both mugs retain false leaks and both cups
retain false hard-thin layers. The source passes 221 tests plus
29 tests covering Torch export, refinement and labels. No recovery gain is established.
A subsequent training augmentation varies fault locations and severities on
retained clean training deposits while preserving the camera and fault model.
Its 1,344 extra tiles contain 30,090 thin-material pixels across 64 groups; the
two largest groups supply 18.3%, versus 93.3% in the original rare-pixel labels.
The original validation set and number of training samples per epoch stay fixed.
Both models finish training and the control reproduces exactly. The candidate's
boundary error and rare-pixel validation recall worsen; all four full-print
diagnostics also fail across 640 frames. The updated source passes 230 repository tests and 38 Torch
and training checks; no recovery improvement is established.
The next paired candidate combines those same augmented examples with the
earlier thin-pixel loss weighting. All added targets and labels reproduce
exactly, and its unweighted control reproduces the coverage checkpoint. Training
finishes with rare-pixel validation recall at 59.1%, versus the original 35.2%,
but more false-positive material and worse average error. The complete Cartesian
mug check now passes; CoreXY closes its false gap but retains one false hard-thin
layer. Both cups retain two false hard-thin layers. All 640 control feature
records reproduce exactly; one of four complete checks passes, so the model
is rejected before recovery evaluation.
The latest experiment freezes this camera and trains small spatial refinement
layers on the same augmented data. Both remove the remaining false hard-thin
alarms on four sampled images, with unchanged thresholds and inputs. Both heads
pass all four complete-print checks across 640 frames. However, the completed
2,040-frame check rejects both: they lose two and three real-defect detections
that the original camera got right, and raise false alarms from 33 to 76 and 85.
The conditional recovery comparison is skipped. A separate replay of one known
defective finish shows that the original neural camera already detects its thin
wall, while the classical camera misses it. Two stronger material-overprediction
penalties also fail a seven-frame screen: both restore false leaks on good mugs
and still miss real defects. The old control reproduces exactly.
An opt-in `--boundary-observed-wall-guard` now retains the existing neural
camera's hard-thin warning on a printed wall, even when no plan passes both views.
Its four paired known-defect runs pass: controls finish defective after one
adaptation; both guarded variants stop at layer 101 without adapting. All 510
common camera frames match. The 50 retained regression rows are running in tmux.
Stopping a defective finish is not a
recovered print, and no new recovery success rate is established. The implementation passes 246
repository tests and 41 Torch/training checks.

## The idea in physical terms

A print is a stack of ~0.2 mm slabs. If slab *k* lands somewhere other than planned (wall sags,
the part slips on the bed, a section is starved of plastic) every slab above it inherits the
error. Scrapping wastes hours; ignoring it can ruin the function (a mug that leaks, a spacer whose
bolt holes moved). The loop looks at the newest slab, measures how far it is from plan, asks
"can this part still do its job?", and if so **re-plans the slabs not yet printed** to start from
what is really there (a circular mug becomes a teardrop mug that still holds water).

## Flowchart

### End to end

```mermaid
flowchart TD
    A[Part: STL, Bambu .gcode.3mf or YAML CSG] --> B[Spec author<br/>LLM + verifier: form and function]
    B --> C[Part spec: geometry rules + loads]
    C --> D{Pre-print FE stress test<br/>leak, safety factor, deflection}
    D -- fails --> X[Fix design or spec]
    D -- passes --> E[Slice in Bambu Studio and start print on P1S]
    E --> F[Every layer: chamber camera frame]
    F --> G[Measure: segmenter extracts current-layer contour]
    G --> H[Detect: deviation over 1 mm and failure class]
    H -- ok --> I{More layers?}
    I -- yes --> F
    I -- no --> Z[Done: part meets spec]
    H -- deviation --> J[Assess + Decide]
    J -- continue --> I
    J -- adapt --> K[Adapt remaining layers<br/>warp, fade, hole relief]
    K --> L[Pause, upload resume G-code, resume]
    L --> I
    J -- scrap --> S[Stop and restart]
```

### Decision on a detected deviation

```mermaid
flowchart TD
    D0[Deviation detected] --> D1{Failure class}
    D1 -- detach or nothing to build on --> SC[Scrap]
    D1 -- other --> D2[Predict finished part if we continue]
    D2 --> D3{Meets spec?}
    D3 -- yes --> CO[Continue]
    D3 -- no --> D4[Build candidate plans:<br/>fade lengths inf..15 layers, with and without hole relief]
    D4 --> D5[Sort by fidelity to the design]
    D5 --> D6{Geometry spec passes?}
    D6 -- no, try next --> D5
    D6 -- yes --> D7{FEA enabled?}
    D7 -- yes --> D8{Stress cases pass?}
    D8 -- no, try next --> D5
    D8 -- yes --> AD[Adapt with this plan]
    D7 -- no --> AD
    D5 -- none left --> SC
```

## Software

| Need | Software |
|---|---|
| Language | Python 3.12 |
| Geometry | `shapely` 2.x, `numpy`, `scipy` |
| Images | `opencv-python-headless` (render, contours, camera homography) |
| Detector | **ConvNeXt-V2** via `timm` (replaces the old TorchVision ResNet) |
| Segmenter | **SegFormer / MiT** via `transformers` (replaces the old U-Net) |
| Stress simulator | built-in voxel FEA (`recovery/fea/`), `numpy` + `scipy` |
| Spec author | `anthropic` SDK (Claude, `claude-sonnet-5-5`), `ANTHROPIC_API_KEY`; heuristic fallback without it |
| Slicer | **Bambu Studio** (`.gcode.3mf`); built-in 2.5D slicer + `trimesh` for STL experiments |
| Printer | **Bambu Lab P1S**, LAN mode: MQTT (`paho-mqtt`), FTPS 990, camera port 6000 |
| Interface / eval | `gradio`, `matplotlib`, `pytest` |

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=.
```

## Generic by construction

Nothing in the pipeline knows about mugs. A part is data: geometry plus a spec.

- **Geometry:** a YAML stack of `add`/`cut` shapes (circle, rect, polygon, tapers), an `stl:`
  mesh, or a Bambu Studio `gcode:` file.
- **Spec rules:** `container`, `min_wall_mm`, `material_volume_mm3`, `holes`, `connected`,
  `present` (all optional) plus `material`, `loads` and `description`.
- `parts/*.yaml` (mug, cup, spacer, bracket) are only examples. Start from
  `parts/_template.yaml.txt`.

### LLM spec author (pins form and function)

```bash
.venv/bin/python -m recovery.cli specgen --part parts/mypart.yaml \
    --describe "bracket holding a 2 kg shelf on two M5 bolts"   # add --no-llm for heuristics only
```

The model sees a geometry summary (layer areas, wall thickness, holes, components per z-range)
and the description, and returns the spec: form rules and the physical load cases. A verifier
then checks the spec against the nominal part (geometry rules and a stress test) and feeds
violations back for up to 3 rounds, so an over-strict or inconsistent spec is repaired before it
is accepted. Output goes to `parts/specs/<name>.yaml`; reference it with `spec_file:`. The LLM
never decides pass/fail at run time; the same deterministic checks do.

## Stress-testing simulator (`recovery/fea/`)

A real (small) finite-element solver for the part about to be printed:

- The sliced layers are voxelised into hex8 bricks with a **transversely isotropic** FDM
  material (PLA, PETG, ABS), weak across layers.
- Loads come from the spec: hydrostatic liquid pressure on walls and floor (with a flood-fill
  leak test through any gap), point/face forces on selected regions, fixed supports.
  Pieces not connected to a support are reported as broken off.
- Matrix-free Jacobi-preconditioned CG solves for displacement. Output: safety factor from the
  99th-percentile element's failure index (von Mises in-plane, interlayer tension and shear),
  max deflection, leak, capacity, with a stress-map figure.
- It is used three ways: **pre-print gate** (`live` refuses a part that fails unless
  `--skip-stress`), **candidate filter** while adapting (`use_fea`), and **what-if** on injected
  failures.

```bash
.venv/bin/python -m recovery.cli stress --part mug --h 1.0
.venv/bin/python -m recovery.cli stress --part bracket --failure shift --severity 3 --seed 1
```

This is linear elasticity at coarse resolution, a screening tool, not certification.

## Perception models (video-style, no TorchVision)

Both nets see a short temporal stack: previous frame, current frame, expected render (7
channels), so they see what changed between layers. Train on synthetic data, fine-tune on real:

```bash
.venv/bin/python -m recovery.cli gen-data --n 4000 --out data/synth.npz
.venv/bin/python -m recovery.cli train --data data/synth.npz --epochs 15 --pretrained
.venv/bin/python -m recovery.cli train --data data/synth.npz --init runs --real data/real.npz --epochs 8 --lr-det 1e-4 --lr-seg 3e-5
```

Classical (threshold) versions are the default and need no training. SAM 2 video tracking is a
possible next step, not wired in.

## How the plan maps to code

| Plan stage | Code |
|---|---|
| Slicer to expected layers | `gcode.py`, `parts.py`, `slicing.py` |
| Synthetic failures and renders | `deform.py`, `camera.py`, `models/synth_data.py` |
| 1. Detect | `models/classical.py`, `models/nets.py` (ConvNeXt-V2) |
| 2. Measure | `models/classical.py`, `models/nets.py` (SegFormer) |
| 3. Assess | `geometry/spec.py`, `fea/` |
| 4. Decide | `loop/decide.py` (rules), `loop/llm.py` (LLM picks among spec-passing options) |
| 5. Adapt | `geometry/adapt.py` |
| 6. Interface | `app.py`, `printer/bambu.py`, `sim.py` |
| Evaluation | `eval/run_eval.py` (ignore, scrap, alert, ours) |

## Run

```bash
.venv/bin/python -m recovery.cli demo --part mug --kind collapse --seed 3
.venv/bin/python -m recovery.cli eval --n 8
.venv/bin/python -m recovery.cli app
.venv/bin/python -m pytest tests
```

## Bambu Studio + P1S setup (week 1, untested on hardware)

**Slice in Bambu Studio** (P1S profile). Process settings that keep the parsed layers clean:
skirt loops 0, brim off, no supports, "Arc fitting" off, timelapse **traditional** (the head
parks at each layer change, giving the camera a clear view). Right-click the plate →
*Export plate sliced file* → `part.gcode.3mf`. Headless alternative: `slicing.bambu_studio_slice`.
Point the part YAML at it:

```yaml
name: mypart
gcode: mypart.gcode.3mf        # nominal layers = exactly what the printer will print
min_wall_mm: 1.6               # ...plus the function spec (see "Adding your own part")
```

**Printer:** enable LAN mode (and Developer mode if your firmware offers it; recent firmware
restricts third-party control without it). Note the IP, serial number and 8-digit access code.
Ports needed: 8883 (MQTT), 990 (FTPS), 6000 (camera).

**Camera:** the P1S chamber camera looks at the bed at an angle, not top-down. Print a
checkerboard at 100% scale, then run `calibrate` (12 shots at different tilts + one flat on the
bed at a known position). The pose-based `BedCamera.top_down` re-projects each frame onto the
plane at the current layer height, so it needs no overhead camera. For best accuracy fix a
dedicated overhead webcam later; the models only see the re-projected view either way.

```bash
.venv/bin/python -m recovery.cli calibrate --ip 192.168.1.50 --code 12345678 --out calib.npz
# start the print from Bambu Studio / the printer screen, then:
.venv/bin/python -m recovery.cli live --part parts/mypart.yaml --ip 192.168.1.50 \
    --serial 01P00A000000000 --code 12345678 --calib calib.npz
```

**The risky part — changing the remaining layers.** A running Bambu job cannot be edited. The
adapter (`BambuPrinter.replace_remaining`) pauses, stops, uploads a self-contained resume G-code
(heat, no homing, adapted layers, `M73 L<n>` so `layer_num` keeps counting) over FTPS and starts it
with `gcode_file`. That works only if the P1S still knows its position after STOP; if it does not,
pass `--rehome-gcode` (re-home XY and re-reference Z at a bed corner away from the part; never
`G28` over the part). Streaming mode is rejected because it has no execution acknowledgements.
Pause, stop, and replacement-start timeouts now abort instead of silently proceeding.
**Validate restart behavior on a scrap print first**: this is the hinge of the
project (the plan's week-1 "pause, edit, resume" test).

## Status: what has and hasn't been verified

The [dated report](docs/mujoco-report.md) and [validation records](docs/validation/)
supersede the original branch's unexecuted-test claims. Validation includes the
classical loop, all four part specifications, G-code parsing, actual MuJoCo camera
and motion tests, an analytical FEA patch test, nominal structural loads, offline
spec generation, STL cross-section smoke testing, and saved SegFormer architecture
reload/inference. Full learned-model training and accuracy, live LLM calls, the
Gradio UI, Bambu Studio CLI, and physical MQTT/FTPS/camera/resume behavior remain
outside those results.

## Modelling assumptions to state in the report

- The simulator is a surrogate. Failures grow progressively (collapse over ~3 mm, shift over a
  few layers); once the controller adapts, the cause is assumed transient and later layers
  deposit as commanded (± 0.06 mm noise). Ignoring a failure lets it persist.
- "Functional" is the spec check in `geometry/spec.py` (including an excess-overhang rule), not a
  physical test. Validate with real prints (e.g. fill the mug with water).
- Abrupt shifts and gaps through already-printed walls are unrecoverable by design → scrap.
- Detection threshold is 1.0 mm; smaller deviations are treated as benign.
- The legacy `eval` command assumes a clean restart succeeds. The new `mujoco-eval`
  reports a scrap as a scrap; a restart is only a separately named cost assumption.

## Project plan

The input document for this work, reproduced in full (PDF: [docs/project-plan.pdf](docs/project-plan.pdf)).


### Problem and goal
When a print fails mid-way, today's options are to scrap it or keep printing a defective part. Both waste hours and filament. Our goal is a feedback loop that catches the failure during the print and adapts the rest of the print so the part still works, even if it looks different.

Example from class: a mug's wall deforms so its circular cross-section becomes a teardrop. Instead of scrapping it, the system continues as a teardrop mug that still holds liquid and keeps its handle. Function is preserved; aesthetics change.

### The feedback loop
Every layer is imaged and checked. On a deviation, the system measures the as-printed shape, checks the part's function spec, and either continues, adapts the remaining layers, or scraps.

### Scope for 4 weeks
The 4-week version targets simple parts where function can be checked from cross-sections, and deformations that leave a usable base to build on.

| | In scope | Out of scope |
|---|---|---|
| Parts | 2.5D parts whose function is set by cross-sections: cups and mugs, containers, brackets, spacers | Complex organic shapes, multi-material, supports-heavy parts |
| Failures | Wall deformation, warping, partial layer shift, under-extrusion of a section | Spaghetti or a detached part (nothing to build on), so these go to scrap |
| Function | A short spec per part, e.g. holds liquid (closed walls, flat base), volume within a tolerance, minimum wall thickness, handle attached | Strength or load testing beyond simple checks |
| Printer | Pause, adapt remaining G-code, resume on one printer we have access to | Fully hands-off physical removal or ejection |

### Components and models
The trained models cover detection and measurement; adaptation is geometry code with an LLM or rules choosing the strategy.

| Stage | What it does | How |
|---|---|---|
| 1. Detect | Flag that a layer deviates and classify the failure type | Fine-tuned image classifier or detector on layer images (trained model) |
| 2. Measure | Extract the as-printed contour of the current layer and compare it to the expected contour from the G-code | Fine-tuned segmentation model on top-down images, registered to the rendered expected layer (trained model) |
| 3. Assess | Check whether the part can still meet its function spec | Rule checks on the measured shape: closed walls, wall thickness, volume estimate |
| 4. Decide | Continue, adapt or scrap | Policy over severity, functional slack, time and filament left; rules first, LLM planner as an option |
| 5. Adapt | Re-plan the remaining layers to start from the deformed contour and still meet the spec (circle becomes teardrop) | Morph remaining cross-sections from the measured contour, re-slice from the current height, resume |
| 6. Interface | Show the deviation, the decision and a preview of the adapted part | Gradio app; printer control through Klipper/Moonraker or OctoPrint |

### Data
Most training data will be synthetic, because printer time is limited; real prints are for validation.

| Source | What it gives | Size and caveats |
|---|---|---|
| Synthetic layers | Expected layer renders from sliced G-code with injected deformations (warp, shift, wall collapse); exact ground-truth contours | Unlimited; PrusaSlicer runs headless at under 0.5 s per slice. Gap to real camera images |
| Slice-100K | 100k models with G-code to slice and deform | Download location to confirm |
| CAXTON (Cambridge, 2022) | 1.2M real print images from 192 parts, labelled with process parameters | Labels are parameters, not geometry; public availability to confirm |
| Public failure images (Roboflow, Hugging Face) | Labelled failure types for the detector | Small (the Hugging Face set has 73 images); mixed licences |
| Our prints | Real layer images with deliberately induced failures, plus end-to-end demos | Roughly 5-10 prints given printer time |

### Evaluation
The main question: when a failure happens, how often does the loop end with a part that still meets its function spec, compared with scrapping or ignoring it?

- Detection and measurement: classification accuracy and false alarms; contour error (IoU, mm deviation) on held-out synthetic and real layers.
- Recovery in simulation: inject failures at random heights in sliced test parts, run the loop, and check the final geometry against the spec. Report the share of functional parts plus filament and time saved.
- Baselines: (a) ignore the failure and keep printing, (b) scrap and restart, (c) pause and alert a human.
- Physical validation: a few real prints with an induced failure, resumed with adapted G-code; check function directly (e.g. fill the mug with water).

### Week-by-week plan
1. **Week 1: problem and pipeline.** Interview a few makerspace or print-farm users about failures. Pick 3-5 test parts and write their function specs. Build the slicer-to-render-to-deform pipeline. Mount the camera and test pause, edit and resume on the printer with one manual example.
2. **Week 2: models.** Generate synthetic data; fine-tune the detector and the contour segmentation model. Collect the first real layer images with induced failures.
3. **Week 3: the loop.** Implement assess, decide and adapt (contour morphing and re-slicing from the current height). Run the loop end to end in simulation; first physical adapted print.
4. **Week 4: evaluate and ship.** Baseline comparisons, real-print validation, Gradio app, report and presentation.

Concept generation and selection for the product-development write-up (scrap-and-requeue agent vs. functional adaptation vs. alert-only) runs alongside week 1.

### Prior work and what's new
Existing closed-loop systems correct process parameters, not geometry. Changing the remaining design mid-print to keep the part functional appears to be open.

| Work | What it corrects | Gap for us |
|---|---|---|
| CAXTON (Cambridge, 2022) | Flow rate, speed, temperatures, Z offset, in real time | Parameters only; no geometry change |
| LLM-3D Print (2024) | GPT-4o agents read layer images and adjust parameters via Klipper | Never alters G-code geometry; layer shifts excluded as unfixable |
| Task-driven printability assistance (2026) | LLM + knowledge graph recommends design and material changes for a stated task | Before printing only; recommends, doesn't modify |
| Manual resume of a failed print | A maker trimmed the G-code and resumed by hand | Shows physical resume is possible but fiddly; layer bonding failed |

Our contribution: detect and measure the deformation, then adapt the remaining geometry to a function spec and resume.

### Risks and open questions

| Risk | Mitigation |
|---|---|
| Resuming on a deformed part may bond poorly or collide with it | Pause, adapt and resume without removing the part; first layers of the new plan start from the measured contour; test early in week 1 |
| Synthetic images differ from real camera views | Use synthetic for pretraining, fine-tune on our real layer images, report real-image results separately |
| Top-down camera may not measure the contour precisely enough | Fixed camera mount and calibration target; compare against a caliper measurement on a few prints |
| Function specs are part-specific | Limit to 3-5 parts with written specs; frame specs as a reusable template |
| Very few real prints | Evaluate mostly in simulation; physical prints as validation and demo |

Open questions:
- Which printer do we have, and can it run Klipper/OctoPrint with a camera?
- Is CAXTON's image dataset publicly downloadable?
- Should the loop also include the scrap-and-requeue agent idea as the "scrap" branch?

### Sources
- Generalisable 3D printing error detection and correction via multi-head neural networks (CAXTON), Nature Communications 2022
- Matta: Detecting and correcting 3D printing errors on the fly
- LLM-3D Print: Large Language Models To Monitor and Control 3D Printing
- Task-Driven 3D Printability Assistance via Geometry- and Knowledge-Grounded LLM Reasoning
- Fabbaloo: Maker experiments with manually resuming a failed 3D print
- Slice-100K: A Multimodal Dataset for Extrusion-based 3D Printing
- Roboflow: 3D print fail detection dataset
- Hugging Face: failures-3D-print dataset
