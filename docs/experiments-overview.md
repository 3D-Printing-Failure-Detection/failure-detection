# Experiments overview (moved out of the README)

Simulation and trained-model experiment notes. The top-level [README](../README.md) is the user guide.

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
common camera frames match. All 50 retained regression rows also pass. Each
variant preserves 14/19 fault successes and all six clean controls, with unchanged
successful action sequences and defective finishes reduced from one to zero.
The complete comparison checks 8,217 common camera frames; Qwen makes 88 actual
preservation requests and matches neural-only. Stopping a defective finish is
not an additional recovered print, and this targeted check is not a fresh benchmark.
Two further spatial heads receive the same observed images and commanded mask
directly. Both improve average boundary error but fail the seven-frame functional
screen, so neither enters recovery. Two missed defects have closed but excessively
thin walls: the existing extra training weights target roughly 0.3 mm material,
while the hard wall limit is 0.96 mm for mugs and 0.84 mm for cups. A training-only
supervision audit targets both sides of that physical boundary. Two audit failures
exposed original-target rounding and invalid derived geometry; both are retained.
The corrected audit reproduces all 7,104 targets exactly and preserves all old
critical labels. No gain is established by creating labels.
The implementation passes 249 repository tests and 54 focused Torch/training checks.
