# Learned recovery and local supervision

The objective is more functional completed prints and genuine recovery from
failures, compared on identical cases with every earlier policy. A trained model
or a successful language-model request alone does not satisfy that objective.

Current verdict: higher completion counts, but no accepted overall improvement.
The complete joint-camera policy finishes **141/240 fault cases with five
defective finishes**, versus the preceding candidate's 136 successes and one
defect. All active recovery baselines' successes are retained, but five ignore
successes remain lost. Qwen makes 527 calls and ties neural-only. The complete
matched comparison covers 5,760 rows across 20 configurations. A new guard on
uncertain, unsupported camera corrections passes its complete 42-row regression:
all ten required fault successes and six clean controls are retained, four extra
defective finishes become scraps, and the prior defective finish remains. Qwen
makes 60 actual calls. The guarded full-grid comparison is now running; its
overall result is not yet established.

The complete analytical-container/learned-fit policy finishes **136/240
fault cases**, versus the matched current analytical policy's **133/240**,
retaining all of that baseline's successes. Both have one defective finish and
pass all 48 clean controls. Qwen makes 499 actual calls and ties neural-only.
The older analytical version had three defective finishes; that reduction also
occurs without learning. The matched comparison attributes three additional
completions to the learned policy and associated proposals, not to accompanying
measurement fixes. Some other baseline and historical successes remain lost;
fresh-test superiority is unproven.

The latest shared-hole proposal completes **79/120 fit failures**, versus analytical
preventive recovery's **76/120** and the preceding neural policy's **78/120**.
Both neural-only and Qwen variants preserve every success from those methods,
finish zero defective prints, and pass all 24 clean controls. Qwen makes 161
actual requests and adds no completions over neural-only. Historical and
validation fit checks preserve the previous best at 33/50 and 13/20, with no
defective finishes and all ten/four clean controls passed.

These are consumed development cases, covering brackets and spacers. Container
perception remains unresolved. The original combined candidate reached 67/100
historical successes versus contour's 62/100 but failed its complete first fresh
test: 126/240 versus contour's 128/240 and analytical preventive recovery's
133/240, with five defective finishes versus three. That result is retained as a
rejection. The latest two-image camera also fails its geometric diagnostic;
dense connectivity training also retains both false gaps and is rejected. Balanced
sampling removes one gap but introduces eight false base alarms, so it is rejected
as well. Commanded-mask conditioning removes both gaps but still has false base
and thinning alarms. Longer training, verified with higher-precision inference,
reintroduces one gap and is also rejected. Fresh seeds 2001–2004 are untouched.
Spatial-gradient training and the subsequent base/wall camera split also fail.
The completed full-grid trial retains analytical containers and the improved
learned fit forecast; its result is the 136/240 development count above.
An actual historical check confirms that this simpler variant loses two earlier
container rescues. Keeping the earlier learned camera as a joint-forecast
alternative restores both; its complete full-grid follow-up produces the
141-success/five-defect trade-off above.

## Current camera cross-check and context experiments

The exact-target sampling audit replays all 5,760 dense tiles bitwise. Only 39 of
3,720 training tiles contain the selected thin-material pixels (1,648 positives);
26 validation tiles contain 1,147 such positives. The diagnostic uses a 0.15 mm
discrete opening of true material on the existing native grid, excludes nominal
thin pixels, and includes a halo to avoid crop-edge artifacts. This identifies
training exposure, not a new functional criterion or a runtime repair.

`balanced-camera-v14` assigns half the training sampling mass to those 39 tiles,
with replacement and the same sample count per epoch. The other half goes to the
remaining training tiles. Validation labels never determine training weights.
The unchanged control exactly reproduces the dense distance checkpoint, and both
exports pass. Thin-material recall rises from 21.54% to 56.98% in training and
6.02% to 21.62% in validation. The complete cup diagnostic fixes the layer-86 gap,
retains the layer-85 gap, and introduces eight false base alarms and one false
hard-thin layer. The model fails the predeclared gate; recovery and wider geometry
follow-ups are skipped. Better local recall is not a successful recovery result.

`command-camera-v15` tests a known input already supplied by the controller: the
binary mask of the commanded layer. It appends this mask to the current and prior
observed image tiles. Actual deposited geometry supplies labels and verifies
sampling only. The builder replays every original target and retains image
channels, targets, groups and splits bitwise; masks come from captured commands.
The image-only balanced control must exactly reproduce v14. Both variants retain
the same sampling, loss and 40-epoch schedule. The extra input changes the
parameter dimensions, so identical initialization is not claimed. The full cup
diagnostic gates actual recovery and 2,040-frame geometry validation, including
true wall gaps to check that the model does not merely copy the commanded shape.
Both models finish training with verified exports. The control exactly reproduces
v14, and command conditioning lowers boundary MAE from 0.08212 to 0.07876 mm.
The full 160-layer cup diagnostic removes both false gaps and has zero false
overhang, but retains two false base alarms and one false hard-thin layer. It
fails the gate; recovery and wider geometry validation are skipped. This is a
camera diagnostic improvement, not a new successful recovery or accepted model.

The best validation loss occurs at the final epoch, 40. The predeclared
`command-long-camera-v16` therefore extends training to 160 epochs on exactly the
same data, sampling and objective. It requires the first 40 epoch records and
initialization to match the prior run exactly, and its full-stack control must
reproduce the prior diagnostic. Selection still uses full validation loss; the
cup outcome does not choose an epoch. The same gate controls follow-ups. The
longer run completes all 160 epochs, selecting epoch 155, but fails its numerical
export gate: maximum float32 Torch/NumPy disagreement is 0.000010848 mm versus the
0.000010 mm tolerance. Exactly one of 8,355,840 validation output signs differs.
A float64 reference attributes the difference to accumulation precision; the
first 40 training records exactly match v15. The failed job and checkpoint are
retained, and no print diagnostic or recovery result is credited to this run.
The first diagnostic used the wrong edge weighting in its separate loss field;
the retained correction uses the exact trainer weighting. Numerical differences,
sign counts and the original failure remain unchanged.

`command-precise-camera-v17` repeats the same 160-epoch training with an explicit
float64 inference marker. Training and stored weight arrays remain float32;
NumPy and OpenCV accumulate in float64. It requires all 160 training records and
every learned weight array to equal the failed export exactly. The strict export
tolerance, functional checks and follow-up gate remain unchanged. Legacy models
retain float32 inference. All 160 training records and every stored weight array
match the failed run exactly. The export error is now 0.000006810 mm, below the
unchanged tolerance. Boundary MAE improves from 0.07876 to 0.06862 mm and
near-boundary sign accuracy from 78.11% to 81.07%. However, the complete cup
diagnostic reintroduces one false wall gap and retains two false base alarms and
one false hard-thin layer. Its reused 40-epoch control reproduces the earlier
diagnostic exactly. The longer model fails the gate; follow-ups are skipped.

A separate retrospective audit covers all 2,040 validation tiles from 36 groups.
It scores actual missing material inside the captured command, using saved signed
distances below -0.05 mm. Missing-material point recall is 91.70% for 40 epochs and
92.18% for 160 epochs across 156,943 points; actual-material recall is 98.29% and
98.27%. Thus the extra command input still permits detection of deviations, but
these correlated tile scores neither establish complete-gap recall nor replace
the failed whole-print check. No candidate from these camera trials is promoted.

`gradient-camera-v18` tests a spatial distance-gradient loss on the same commanded
tiles, with weights 0.1 and 0.3 alongside a zero-weight control. The loss matches
adjacent signed-distance differences to their training labels; it does not simply
smooth the output or repair geometry at runtime. Both candidates retain the same
initialization, sampling, split and 40-epoch schedule. The control reproduces v15
arrays and its checkpoint hash exactly. Every export passes. Boundary errors
are 0.07876 mm (control), 0.08678 mm (0.1) and 0.08631 mm (0.3). Both candidates
reintroduce two false wall gaps. The 0.3 model removes false base and hard-thin
alarms, but still fails the full-print gate. Both are rejected and their
conditional recovery and wider geometry trials are skipped.

The retrospective `distance-offset-audit-v19` checks whether a simple signed
distance offset is supported by training residuals. For points within 0.15 mm of
the true boundary, mean target minus prediction is -0.01629 mm in training and
-0.01377 mm in validation. The corresponding wall-only means are -0.00715 and
-0.00299 mm. A blanket positive offset to thicken the estimate therefore opposes
the average residual; none is applied. These averages do not establish accuracy
on rare thin walls.

`command-base-topology-v20` inspects the commanded model's raw output on the first
five cup frames. The false holes at layers 0 and 4 are disconnected from exterior
background under both four- and eight-neighbor connectivity, before contour
conversion. Each converted hole has area 0.000335 mm², versus 0.003906 mm² per
native output pixel. The layer-0 hole lies inside actual material; the two layer-4
holes are background pockets enclosed by an erroneous predicted bulge. This is
distinct from the earlier diagonal-connectivity conversion issue. No hole filling,
measurement offset or change to the functional specification is applied.

`base-wall-recovery-v21` evaluates a separate hybrid policy in the actual recovery
loop. The optional `--boundary-classical-base` uses classical measurements for
the solid-base indices declared by the original part specification and learned
measurements for walls in the alternative history. Both models still observe
every image, including after a changed command; earlier measurements are never
rewritten. Primary history stays classical. The split requires
`--boundary-fallback`; all existing proposal, execution and functional checks
remain in force. A matched ablation uses the identical checkpoint without the
base split. The predeclared comparison retains all six previous methods on four
failure cases and two clean controls: 18 new rows plus 36 prior rows, with source,
model and supervisor provenance pinned separately. The reserved fresh seeds are
not used. All 54 rows are complete and audited. Both new variants score 2/4
fault cases, versus the previous joint-camera policy's 3/4 and ignore's 4/4,
with zero defective finishes and both clean controls passed. Qwen makes 12
actual requests and ties neural-only. Removing the base split also scores 2/4,
so the split adds no completion. The thinning case still scraps at layer 86
after the learned alternative permitted continuation at layer 85. The shifted
cup also scraps after one adaptation, losing a previous joint-camera success.
The hybrid is rejected and its conditional broader container run is skipped.

## Full comparison of learned fit recovery and analytical containers

`fit-analytical-containers-v22` returns to the direct overall comparison. One
frozen policy uses classical camera measurements for every part, the trained
equivariant forecast and shared-hole proposal for specification-defined fit
parts, and analytical forecasts for containers. It runs with and without actual
local Qwen. The earlier 79/120 fit result motivates this test; it is not added
arithmetically to old container results or credited to this new run.

The protocol fixes all three printer profiles, all four parts, all five failure
kinds plus clean, and consumed seeds 1001–1004: 576 new rows. The comparison
retains the original 4,320 raw rows across all 15 full-grid configurations, with
their original sources and model provenance. Both lost baseline successes and
defective finishes must remain visible. The old agent shards have an asset-only
attestation and a richer server attestation for the same Qwen/runtime assets;
their original records must be preserved when verifying cross-version results.
Evaluation and audit run in tmux window `neural-recovery:hybrid-all-v22`, with
logs and exit status on the data volume. Evaluation, audit and the complete
4,896-row comparison pass. Both neural variants complete 136/240 fault cases,
versus analytical preventive recovery's 133/240. They retain all of that baseline's
successes and gain spacer shift at seed 1002 on all three profiles. Defective
finishes fall from three to one; all 48 clean controls pass. Qwen makes 499
actual requests and ties neural-only. Other baseline successes remain lost,
including ten cases that ignore finishes successfully. Historical regression
and a new untouched test remain required. No result from this consumed grid
can establish fresh-test superiority.

The comparison verifier now preserves the exact manifest hash and original
supervisor attestation for each old worker. A pinned runtime check must show the
same model assets and inference command settings. Only host/port values and JSON
whitespace are normalized; changed weights, inference settings, records and
unused shard pins are rejected. All 23 focused provenance tests pass. Replaying
the complete original comparison reproduces all 15 policies' outcome/call counts
and every paired gain, loss and defect difference across 4,320 unchanged rows.

A separate retrospective attribution distinguishes completed prints from actual
rescues. The analytical preventive policy's 133 completions include 15 cases
where ignore fails, while losing ten cases that ignore passes. The old combined
neural policy has 15 rescues but loses 14 ignore successes; its Qwen variant has
12 rescues and loses 14. Every completed print with zero adaptations matches its
ignore counterfactual. The new full run has 18 genuine rescues, versus the earlier
analytical policy's 15, while still losing ten ignore successes. Its 136 completed
prints include 109 without adaptation and 27 after adaptation. Ordinary
successful continuations are not credited as recoveries.

`matched-analytical-v26` reruns analytical preventive recovery using the exact
current source and classical camera used by v22. This separates accompanying
measurement fixes from the learned policy and its associated recovery proposals.
The complete 288-row grid uses no neural or agent calls; it joins all 17
existing configurations under a separate label, preserving their raw inputs.
The run completes all 288 cases. The combined comparison now contains 5,184
audited rows across 18 configurations: current analytical recovery scores 133/240,
versus 136/240 for both learned variants. Every analytical success is retained;
the three gains remain spacer shift at seed 1002 on all three profiles. Both
policies finish one defective print and pass all 48 clean controls. Genuine
rescues are 15 versus 18; both lose ten ignore successes. Thus the three-print
gain survives the same-code control, while the old three-to-one defect reduction
also occurs without learning. This compares the learned policy and associated
proposals, not the forecast in isolation.

The first audit stopped because it required a trained checkpoint even for this
explicitly model-free control. No simulation is rerun. A new `--analytical-only`
audit requires known analytical methods, no loaded model assets, zero neural/
camera/agent calls and analytical measurement/forecast records. The default
audit still requires trained-model provenance. All 38 audit/provenance checks
pass. Re-auditing the 576 neural/Qwen rows preserves every prior audit field
except the auditor digest, including all 499 actual calls. The original failed
audit, successful audit-only resume and unchanged raw-result hashes are retained.

Two further camera diagnostics are rejected. `camera-average-diagnostic-v24`
averages the short- and long-trained models' signed distances with fixed equal
weights, before contour conversion. Both member controls reproduce exactly, but
the mean retains one false wall gap, two false base layers and one false hard-thin
layer. Classical bases remove only the base alarms. `asymmetric-camera-v25`
triples the training cost of underpredicted signed distance, retaining the same
data, initialization, sampling and 40 epochs. Its unit-weight control reproduces
the original checkpoint exactly. The asymmetric model worsens validation boundary
error from 0.07876 to 0.09269 mm and sign accuracy from 78.11% to 74.20%; its full
cup diagnostic has one false gap, two false base layers and two false hard-thin
layers. Both exports pass. These are not calibrated confidence bounds. Neither
trial receives recovery credit; broader follow-ups are skipped. Eight focused
Torch tests pass, including the unchanged default loss and asymmetric gradients.

## Explaining false thickness alarms and testing spatial refinement

`thin-attribution-v27` decomposes saved measured geometry on two known cup frames
using the exact functional thickness calculation. The check measures material
removed by a morphological opening, relative to the nominal layer. On layer 86,
the short-trained model's thin region contains 6.1633 mm² inside genuinely thin
material, 0.2431 mm² inside genuinely thick material, and 2.0401 mm² outside the
actual print. These partitions reproduce the original metric exactly after
accounting for the opening's small exterior area. The predicted excess is
0.030745, above the unchanged 0.03 threshold; actual excess is 0.024405. Removing
contour simplification worsens the prediction to 0.037241. Simply predicting more
material or removing simplification therefore does not address this error.
Truth is used only for this retrospective explanation, never as a runtime input.

`native-refinement-v28` freezes the entire commanded temporal camera and learns
a 3×3 convolution plus bias on its native distance logits before tanh: ten new
parameters, initialized as identity. The original distance loss, 40 epochs and
training-only balanced sampling are retained. Valid convolutions score the common
62×62 tile interior, excluding padding at crop edges. Both base and refinement
use the same dataset and disjoint 66 training/36 validation trajectory groups;
the CLI also verifies inherited training provenance before a future run.

Training completes in 21.2 seconds. On those common interiors, validation
boundary error falls from 0.078446 to 0.072209 mm and near-boundary sign accuracy
rises from 78.16% to 80.50%. The frozen base arrays remain identical. Export
disagreement is 0.000006527 mm, below the unchanged 0.000010 tolerance. Full-image
and training inference agree across native block boundaries. The identity control
preserves the old camera field exactly. All 24 focused camera/training checks
pass; the repository suite passes 147 tests with three optional modules skipped.

The complete 160-frame cup diagnostic passes for both the raw refined camera and
its classical-base/learned-wall alternative: zero false wall gaps, base failures,
hard/soft thickness failures and overhang failures. The original and identity
controls reproduce the earlier two false base layers and one false hard-thin
layer exactly. All models observe all 160 frames. The broader 2,040-frame
validation is complete and rejects the model: it misses ten real gaps versus
the original camera's five. Its false gap alarms fall from 63 to 35, and severe
thinning misses from 40 to nine, but that does not waive the missed-gap gate.
The weights are selected by tile validation loss, never by the cup outcome. This is
an experimental learned output layer, not calibrated confidence or a fixed mask
repair. No recovery gain or fresh-test superiority is established by training.

| Camera | False gaps | Missed gaps | False severe thinning | Missed severe thinning | False base alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 63 | 5 | 0 | 40 | 0 |
| Commanded control | 34 | 13 | 53 | 9 | 452 |
| Learned refinement | 35 | 10 | 16 | 9 | 5 |

All methods see the same frames. There are 57 real open-wall frames and 48 real
severe-thinning frames. The 864 base frames contain no actual base failures, so
base-fault recall is not established. The refinement improves the commanded
control substantially but still fails comparison with the original gap detector.

`refined-recovery-gate-v29` was queued behind completed real-gap validation and
the matched analytical-control run. It runs only if the existing geometry gate
passes, after checking available storage. Its 30 planned rows test the refined
policy with and without actual Qwen, plus a neural-only ablation without the
classical-base split. Two comparisons remain separate: the four-fault/two-clean
camera gate retains all nine earlier configurations (72 total rows), while the
two-fault/two-clean historical gate retains all six earlier configurations (36
total rows). The same frozen policy must complete all six faults and all four
clean controls with no defective finishes and actual model calls before broader
regression. This preserves the earlier shifted-cup and historical rescue cases;
it does not treat the passed nominal-camera diagnostic as a recovery outcome.
The initial waiting runner stopped on the analytical audit failure before running
any recovery rows. Its resumed runner pins the successful audit-only completion
and retains the same model, cases and geometry gate, in tmux window
`neural-recovery:refined-recovery-v29-resume`. Both original failed attempts remain
available; only the waiting orchestration resumes.
The completed geometry gate rejects v28, so the resumed runner terminates as
`skipped`: zero recovery rows run and zero recovery outcomes are credited. The
planned comparisons and original failures remain retained.

`gap-sensitive-refinement-v30` tests whether penalizing invented material can
address the missed gaps. It retains the frozen camera, data, groups, sampling,
identity initialization and 40 epochs; only the SmoothL1 cost for predicted
signed distances above their targets doubles. The unit-weight control reproduces
v28's checkpoint and full cup diagnostic exactly. Both exports pass the original
numerical tolerance. The candidate's boundary MAE is 0.074073 mm versus the
control's 0.072209, and near-boundary sign accuracy is 79.98% versus 80.50%.
The complete 160-frame cup diagnostic finds two false wall gaps for both the raw
candidate and its classical-base hybrid. It is rejected; broader geometry
validation and recovery are skipped, with zero recovery outcomes credited.
The loss is not a calibrated confidence bound. All 25 focused camera/training
tests pass, including the unchanged unit-weight objective and asymmetric gradient.
The retained [condition](validation/neural/gap-sensitive-refinement-v30-condition.json)
and [comparison](validation/neural/gap-sensitive-refinement-v30-comparison.json)
record this negative result. The completed recovery comparison below redirects
work toward unsupported corrections; seeds 2001–2004 remain untouched.

`nonlinear-refinement-v35` tests whether a local nonlinear correction can avoid
the linear filter's missed real gaps. It freezes the same commanded temporal
camera and adds a 3×3 convolution with eight hidden channels, ReLU, a 1×1 output
and a residual connection before tanh. Only 89 parameters are trained. Zero
output weights and bias initially preserve the original logits exactly. The
linear control, dataset, 66/36 trajectory groups, sampling, symmetric loss and
40-epoch schedule remain matched. Neither diagnostic outcomes nor gap counts
select the checkpoint.

The retrained linear control reproduces v28's checkpoint hash exactly. Both
heads begin with identical validation predictions and preserve every backbone
weight. The nonlinear head's boundary MAE is 0.071095 mm versus 0.072209 for the
linear head; near-boundary sign accuracy is 80.39% versus 80.50%. Export error
is 0.000008285 mm, within the unchanged 0.000010 mm tolerance. These tile scores
do not establish fewer missed gaps or better recovery.

The full 160-frame cup diagnostic passes for both raw learned geometry and the
classical-base hybrid: no false gaps, base failures, thinning or overhang. The
linear control and actual truth reproduce exactly. This releases the same
2,040-frame, 36-group geometry comparison, which is running. The original
camera and linear control must reproduce all prior frame measurements. Gap
false positives, gap misses and severe-thinning misses may not exceed the
original camera's counts. Failed gates skip recovery; fresh seeds stay untouched.
The experiment runs in `neural-recovery:nonlinear-v35`, using two CPU threads and
an 8 MB artifact allowance that reserves space for the live full policy test.
All 159 repository tests pass with three optional modules skipped; all 22
focused camera/export tests pass in the existing Torch environment. They cover
full images, native block seams, float32/float64 export, exact identity behavior,
invalid weights and conflicting heads. The running guarded policy keeps its
separate frozen source and original camera checkpoint.

## Diagnosing a false future connection failure

`forecast-attribution-v36` replays the consumed bedslinger mug-collapse case at
seed 1001, with no changed commands or recovery. Both camera models see every
one of the 160 actual frames. At layer 137, both current contour forecasts
reproduce the recorded false connection failure near z=35.2 mm. Substituting
actual prefix geometry leaves that forecast failure intact, while combining
the measured prefix with the actual future passes. The actual complete print
and both full measured histories pass. Holding the latest measured deformation
without extrapolating its recent trend passes from either camera history.
This attributes the selected false alarm to the future forecast; it does not
establish a successful recovery or a generally better predictor.

The optional `--contour-trend-shrinkage` subtracts the existing observation
uncertainty from the magnitude of each clipped median increment, floored at
zero. It retains the increment's direction, the latest measured deformation,
all prior measurements and the existing three-layer extrapolation limit.
The trained fit forecasts are unchanged. No fault identity, outcome or new
fitted threshold enters the policy; the uncertainty remains an engineering
estimate rather than a calibrated probability.

`trend-shrinkage-v37` first runs a same-source default mug control and candidate,
each with and without actual Qwen. The control must reproduce the earlier
scrap and action sequence. Both candidates must finish the print successfully,
with no defective finish and recorded shrinkage calls, before all 42 prior
targeted rows run. The complete gate then requires all ten earlier fault
successes and six clean controls, the new mug success, at most the same single
existing defective case, and actual neural/camera/agent inference. Fit rows
must record zero shrinkage calls. There are 46 new rows if the first phase
passes: two controls and 44 candidate rows. Broader comparisons remain required.

All 164 repository tests pass with three optional modules skipped; 21 focused
forecast/fallback tests pass. They check retained current deformation, suppression
of increments below existing uncertainty, preserved larger trends and horizon,
unchanged observation histories, propagation to both camera forecasts and
unchanged learned fit states. The experiment runs in
`neural-recovery:trend-shrinkage-v37`, with at most two simulation workers and
an 8 MB artifact allowance. Older full and camera jobs retain their frozen sources.

## Preserving historical container rescues

`joint-historical-gate-v23` compares the complete improved fit policy with
analytical containers against the same policy using the earlier trained camera
as an independent alternative (`--boundary-fallback --boundary-joint-forecast`).
It uses the earlier camera checkpoint, not a rejected replacement. Both neural-only
and actual Qwen variants run on bedslinger cup under-extrusion and mug shift at
seed 10, plus both clean controls. Contour and ignore are rerun as paired controls
under the same frozen source. These are selected historical cases, not fresh data.

All 24 rows are complete and audited. Both analytical-container variants and
contour complete **0/2** faults; ignore finishes both defective. Both joint-camera
variants complete **2/2**, with zero defective finishes and both clean controls
passing. The cup uses one adaptation and the mug three. Qwen makes 11 actual
requests and matches neural-only. This preserves two earlier rescues that the
simpler variant loses; it does not establish a new overall best result.

The predeclared gate passes and its full 576-row follow-up completes. Both
variants finish 141/240 faults, five defective prints and all 48 clean controls.
Qwen makes 527 calls and adds no completions. Every success from the 136-success
candidate and the other active recovery baselines is retained; five ignore
successes remain lost. `joint-matched-comparison-v31` adds the completed same-code
analytical control without rerunning any simulation. All previous 19 summaries
remain identical in the resulting 5,760-row, 20-configuration comparison. The
analytical control remains 133/240 with one defect. Source/model pins, all raw
outcomes, gate failures and audits remain retained.

The five gains over v22 are cartesian cup collapse/1003 and shift/1002, corexy
mug collapse/1001 and warp/1001, and bedslinger mug warp/1001. Four finish without
adaptation; one follows adaptation. Genuine rescues remain 18, while lost ignore
successes fall from ten to five. All four extra defects follow a learned-camera
correction when no classical plan passes and every alternative has robust margin
-1. They occur on mug under-extrusion/1002 on all three profiles and
corexy mug under-extrusion/1004. The prior bedslinger mug under-extrusion/1004
defect remains. These are recorded model assessments, not causal proof of why
the physical simulation failed. The count gain does not waive the defect regression.

## Guarding unsupported camera corrections

`--boundary-robust-fallback` is opt-in. When no classical-camera plan passes, a
learned-camera correction must also pass its existing uncertainty envelope:
finite `robust_margin >= 0`. Failed alternatives remain visible in the decision
record and cannot be offered to Qwen as eligible. A passing continuation and a
jointly checked correction retain their existing behavior. This uses the
unchanged functional check; it is not a calibrated probability or a new geometry
tolerance. Both historical rescues and successful container corrections in the
completed full grid used jointly checked plans, motivating this distinction.

`robust-fallback-v32` freezes the guard with the same trained forecast, original
camera and local Qwen. Its 42 rows cover the five new gains, four additional
defects, the prior defect, two historical rescues, three learned fit gains and
six clean controls, with and without Qwen. Both methods must preserve all ten
reference fault successes and six clean controls, introduce no new defective
case, and finish at most the one already defective case. Actual neural/camera
queries and actual Qwen calls are required. All 42 rows finish and pass their
audits: each variant completes ten of 15 faults and all six clean controls,
scraps the four additional defective cases without adaptation, and retains the
one existing defective finish. No reference success is lost. Each variant makes
104,468 neural and 2,280 boundary queries; Qwen makes 60 actual calls and ties
neural-only. This passes the targeted gate and releases the full rerun, not
acceptance. Historical full regression and untouched evaluation remain required.

All 159 repository tests pass, with three optional modules skipped. Tests cover
default behavior, negative/nonfinite versus zero uncertainty margins, preserved
continuations and joint plans, and audit rejection of missing or bypassed guards.
Re-auditing the completed 576-row joint-camera grid preserves every earlier
audit field except the auditor digest, including all 527 actual Qwen calls.

`robust-full-v33` starts after verifying the complete 42-row gate passed. It runs 576
new rows on all three profiles, four parts, five faults plus clean controls,
and seeds 1001–1004. The same frozen source, model weights and policy options
are retained; twelve workers replace the prior six after checking CPU and memory
availability. Its 6,336-row comparison retains all 20 earlier configurations
under their own labels and adds the two guarded variants. All earlier summary
values must remain identical. It is running in tmux window
`neural-recovery:robust-full-v33`, with a 35 MB artifact allowance checked again
before execution. A failed targeted gate skips it with zero new rows.

`robust-historical-v34` waits for that full comparison. Before starting, the
candidate must have the best completion count, no higher defect count, all clean
controls passing and no lost active-recovery baseline successes. Any remaining
ignore losses stay visible and do not count as acceptance. A separate storage
check reserves 16 MB. This follow-up runs 240 new rows on all 100 historical
fault cases and 20 clean controls with and without Qwen. Eleven earlier full-domain
configurations remain in the resulting 1,560-row comparison. A separate 480-row
comparison retains all six prior fit-only configurations on 50 faults and ten
clean cases. Splitting new jobs by declared part domain supplies both complete
grids without filtering or fabricating old results. All historical source
snapshots, model hashes and the shared simulation contract have been verified.
Its tmux window is `neural-recovery:robust-history-v34`. The former held-out seeds
22–23 are explicitly historical development data; seeds 2001–2004 remain untouched.

## Earlier joint-camera and measurement studies

The optional `--boundary-joint-forecast` extends `--boundary-fallback`. When the
classical-camera continuation fails but a correction is feasible, both camera
histories propose corrections and assess each other's proposals. A jointly
passing correction can displace one rejected by the other view. If none passes
both views, retain the original classical proposal. A passing classical
continuation remains unchanged. Histories remain independent and every frame is
observed. This compares two interpretations of one image, not two physical cameras.

The frozen development protocol `camera-joint-dev-v1-protocol.json` specifies
36 runs over four failure cases and two clean controls, including matched old
fallback and learned-camera-only ablations. The completed shifted-cup case
(Cartesian, seed 1002) now finishes successfully with one adaptation for both
neural-only and actual Qwen, versus a scrap for contour. Ignore also passes;
this is preservation of a recoverable print, not a rescue of an otherwise harmful
fault. The completed candidate comparison scores 3/4 failures versus contour's
2/4 and ignore's 4/4, with no defective finishes and both clean controls passed.
Qwen makes 19 validated requests and matches neural-only outcomes. The false-gap
thinning case still scraps; this loses an ignore success and fails acceptance.
The completed matched ablations score 2/4 for the previous fallback and 1/4 for
the learned camera alone, which also finishes one defective print. All six methods
pass both clean controls. All 36 planned rows are retained and audited.

The original wrapper exits unsuccessfully after the thinning case because its
shard auditor requires a Qwen request even when no eligible choice exists. The
completed raw case is retained, including zero requests. An explicit
`--allow-unreached-agent` audit option now verifies that no eligible decision was
bypassed and reports `agent_review_status: not_reached`; it never counts the shard
as actual supervision. The aggregate still requires actual agent requests. The
failure log and frozen runtime remain unchanged, and `camera-joint-completion-v2`
executes only the originally planned remaining jobs. No print is rerun or relabeled
as successful to repair the audit.

The repaired contour conversion's 36-group, 2,040-frame camera validation is
complete. Among 1,176 wall frames, 57 truly contain an opening. Classical
segmentation has 66 false openings and three misses; the original neural camera
has 63 and five, expanded-data central prediction 49 and six, upper prediction
39 and seven, and the shared-query decoder 51 and six. The newer networks detect
more thinning but introduce 259, 270 and 165 false base-problem frames respectively,
versus zero for classical and original neural segmentation across 864 base frames.
There are no positive base-fault examples in this sample, so base-fault recall
cannot be estimated. These are correlated per-frame geometry errors, not print
success rates. None of these replacement networks is promoted.

A scene-raster diagnostic finds that the known cup-thinning layer still encloses
its cavity at the existing scene resolution. It does not support blaming the
scene's binary rasterization for that false opening; rendering and functional
specifications are unchanged. The next test, `context-camera-v8-protocol.json`,
adds a stride-four 9x9 grayscale patch beside the original fine patch. Both come
from the existing camera image. A fine-only network is trained on exactly the
same rows, targets, split, seed and optimizer schedule to isolate the context
change. The resource-bounded dataset uses all 66 training and 36 validation
container trajectories, yielding 86,400 sampled patches (55,800 train and 30,600
validation). Both models complete 160 epochs, with verified NumPy export. On the
same validation points, boundary MAE is 0.06493 mm for the matched fine-only
model and 0.04817 mm with wider context. The existing central and shared-query
models score 0.06575 and 0.06876 mm on those same points; these values are not their
earlier full-dataset scores. Full-stack diagnostics and the 2,040-frame geometric
validation are separate from patch MAE. The completed cup diagnostic still reports
two false wall gaps and a 24 mm2 false support violation. Its conditional recovery
run is skipped. The complete 2,040-frame geometry check also favors the matched
fine-only model over wider context: 48 versus 51 false wall gaps, five versus six
missed gaps, and 55 versus 134 false base alarms. Wider context is therefore not
an improvement in the relevant geometry despite its lower patch error. A retained image/geometry
plot shows the visible thin arc fragmented by the prediction. In that diagnostic
arc, only 48.2% of true positive samples within 0.15 mm of the boundary receive a
positive prediction, versus 88.3% elsewhere. This identifies a local failure hidden
by the average metric; it is not an independent benchmark.

`context-query-camera-v9` combines the wider inputs with a shared subpixel distance
decoder, using the identical data, split and 160-epoch schedule. It completes
training at 0.04462 mm validation boundary MAE, with verified export. The full-stack
cup diagnostic still reports two false gaps, a 26 mm2 false support violation and
one false base-problem layer. Recovery and wider geometry checks are skipped as
predeclared. This model is also rejected; lower patch error did not repair the gap.
Seeds 2001--2004 remain untouched.

The matched `occupancy-camera-v10` trial adds a fixed material/background
classification loss to distance regression (weight 0.1, distance-to-logit scale
0.1 mm). The scale is not calibrated uncertainty. It uses the fine-only inputs,
identical data and 160-epoch schedule. Its zero-weight control reproduces every
array of the earlier fine-only checkpoint exactly, and both NumPy exports pass.
The auxiliary objective worsens near-boundary sign accuracy from 82.09% to 81.35%
and boundary MAE from 0.06493 to 0.06604 mm. The full 160-layer diagnostic retains
both false gaps and increases false overhang from 15 to 23 mm2. Recovery and wider
geometry checks are skipped by the predeclared condition; this model is rejected.

`dihedral-query-camera-v11` rotates and reflects both the image patch and its
continuous subpixel query coordinates, preserving each signed-distance label.
The unaugmented wider-query control reproduces every earlier v9 checkpoint array
exactly. Both models train for 160 epochs with verified exports. Augmentation
worsens boundary MAE from 0.04462 to 0.05504 mm and near-boundary sign accuracy
from 86.94% to 83.09%. The completed cup diagnostic keeps both false gaps and
increases false base-problem layers from one to eight. Its conditional follow-ups
are skipped and the model is rejected. The original protocol retains stale v10
descriptive fields; the separately recorded metadata correction identifies them
and the correct trainer hash. Frozen commands, data, source and outcomes were
unchanged; the original protocol remains available beside that correction.

The `temporal-camera-v12` experiment adds the immediate preceding camera image
to the current fine patch. The first layer duplicates the current image; later
layers retain the preceding observed image. No nominal geometry, injected fault,
simulator truth or future image enters the network. Legacy checkpoints still use
one image. Offline validation explicitly supplies the immediate predecessor,
because its selected frames are not necessarily consecutive.

A three-trajectory replay check covers 165 current frames and 156 preceding
frames across all three views. Every current input and target is exactly equal
to the retained dataset; cold starts duplicate current pixels. The full study
reuses the 66 training and 36 validation trajectories and verifies the same
equality across all 86,400 patches. Its current-only control exactly reproduces
every earlier fine-only checkpoint array. Both models completed 160 epochs.
Boundary MAE improves from 0.06493 to 0.06174 mm, and near-boundary sign accuracy
from 82.09% to 82.64%. The full-stack diagnostic still reports both false wall
gaps and one false hard-thin layer, despite removing the false support violation.
Recovery and broad geometry validation were skipped under the predeclared gate;
the temporal checkpoint is rejected.

The [raw-mask diagnostic](validation/neural/camera-contour-topology-diagnostic-v13.json)
finds that the centre air region already reaches the exterior in both models'
native pixel masks, under both 4- and 8-neighbor background connectivity. These
gaps precede contour conversion. Its separate provenance record identifies the
actual current/previous images and the retrospective diagnostic procedure.
Sampling the true deposited geometry at those exact native coordinates preserves
the closed wall under both connectivity definitions (enclosed cavity 1704.50 mm²),
so this frame's output grid is capable of representing the missing connection.

The [dense-topology study](validation/neural/dense-topology-camera-v13-protocol.json)
therefore tests adjacent image tiles with an auxiliary soft-Dice/soft-clDice loss,
following [Shit et al., CVPR 2021](https://arxiv.org/abs/2003.07311). Both variants
use the same two-image runtime network, initialization, data, split and 40-epoch
schedule. The control uses distance loss alone. One 24-by-24 input tile per
selected frame supplies a 64-by-64 distance target at the runtime's native pixel
positions. The fixed auxiliary weight is 0.1, with equal Dice/clDice contributions,
20 soft-skeleton iterations and a 0.1 mm distance-to-logit scale. Homogeneous tiles
omit clDice. Truth supplies training labels and sampling locations only. No runtime
gap filling or changed functional thresholds are introduced.

Both 40-epoch runs complete and pass full validation-tile export checks. The
distance control trains in 46 seconds; connectivity training takes 746 seconds.
Matched boundary MAE is 0.07839 versus 0.07911 mm, and near-boundary sign accuracy
78.73% versus 77.73%. The full 160-layer diagnostic still finds the same two false
openings (layers 85 and 86) for both models. The distance control also introduces
one false base alarm and a false support violation. Connectivity training removes
those extra alarms, but the wall gaps remain: it is rejected and its predeclared
recovery and wider geometry follow-ups are skipped. These measurements supply no
new successful-recovery result.

The current complete source check passes 139 tests, with two optional modules
skipped. The focused run passes 18 checks in the Torch environment,
including commanded-mask export parity, image history, dense tile coordinates,
training-only sampling, and penalties for missing walls and invented bridges. Long runs
remain in tmux session `neural-recovery`; attach with
`tmux attach -t neural-recovery`. Full evidence stays on the data volume, with
compact manifests and completed audits under `docs/validation/neural`.

To retain room for the final comparison under the shared 100 GB allocation,
[verified archival](validation/neural/completed-report-archive-v2.json) compresses
1,157 duplicate CSV exports and case journals from the completed first frozen
benchmark, reclaiming 48.66 MB. Every journal result equals its retained aggregate
record, and each gzip file reproduces its original bytes. Primary `runs.jsonl`,
audits, manifests, source and checkpoints remain unchanged. The full 4,320-run
comparison reproduces all outcome counts and gained/lost cases after archival.
The linked receipt identifies a quota-checked restore tool and a verified sample
restoration. An initial wrapper-format verification aborted before changing files.

The later [v3 archive](validation/neural/completed-report-archive-v3-summary.json)
and [v4 archive](validation/neural/completed-report-archive-v4-summary.json)
retain 867 additional duplicate journals and CSV exports from completed v22,
v23 and v26 runs: 37.66 MB compressed to 2.55 MB. Every journal row equals its
retained aggregate, every archive member reproduces its original bytes, and
both restore utilities pass an actual sample restoration. Raw result JSON,
manifests and audits retain identical hashes. Receipts and quota-checked restore
scripts remain under the corresponding data-volume archive directories.

## Comparing actual earlier implementations

`tools/compare_neural_experiments.py --configurations PATH` accepts an explicit
per-report-label provenance document instead of one shared training manifest.
Each label pins its raw method, source directory and digest, every behavioral
argument, checkpoint and training-manifest hashes, and supervisor provenance.
Unknown future behavioral arguments are pinned too. Runtime source and weights
are checked from disk; changed versions cannot be pooled under one label.

All configurations must share the same simulator, configuration, deposition and
toolpath code, controller/action limits, geometry scoring, part specifications,
camera projection/resolution, packages and printer profiles. Pixel-mask conversion
is allowed to differ as part of the explicitly pinned measurement implementation.
All other camera code remains fixed. Test seeds must be disjoint from every model's
training and validation groups. The existing single-version mode remains strict.

The retained `cross-version-v2-*` documents exercise this path on actual completed
earlier/current fit runs. Historical completion ties at 21/30 in the main slice;
validation improves from 12/20 to 13/20 without lost successes. This is a retrospective
harness check, not newly generated print evidence or a new frozen test. Final
configuration pins must be established before the untouched comparison starts.

The user requested a free model without an API key instead of Gemini. The first
supervisor is Qwen3-0.6B Q8_0, served on localhost by llama.cpp on CPU. This is a
resource-constrained starting point whose decisions must be evaluated, not an
assumption that a language model improves recovery. Official references:
[Qwen model](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) and
[llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

## Learning experiment

Train a compact temporal neural network to forecast the command-to-deposit
displacement from recent camera contour residuals. Keep separate inner and outer
boundaries. The network predicts how the observed deformation changes over the
remaining print; the analytical baseline extrapolates a median trend for three
layers and then holds it constant. Training labels may use simulated geometry;
policy inputs may use only images, past commands, the part spec and action budget.
No failure type, random seed, onset, severity, actual deposited geometry, or final
truth verdict may enter a runtime model or supervisor request.

Start with synthetic camera trajectories for affordable training. Report that
domain explicitly and evaluate transfer with the existing MuJoCo camera and
persistent-failure printer. Split entire trajectories by seed, not random frames.
Training seeds: 60–67. Validation/development seeds: 70–73 and previously published
cases. Reserve seeds 1001–1004 for a frozen final comparison; do not inspect them
for tuning. If a final comparison fails, retain it as a failed test and reserve
new untouched seeds before another final comparison.

## Supervisor and evaluation

The local agent receives camera-derived state and tool-computed candidate
assessments. It chooses an executable recovery among candidates that pass the
unchanged functional specification. Log requests, responses, selected actions,
rejections, latency, model/checkpoint identity and actual request counts. Model
errors must be visible and must not be counted as successful agent supervision.
Neither the agent nor the neural network may alter the already printed prefix.

Compare neural-only and neural-plus-agent ablations against ignore, scrap, alert,
ours, measured, predictive, functional, guarded and contour. Preserve existing
checks, continuous camera monitoring, maximum three adaptations, and persistent
failure causes. Report completion, defective finishes, scrapping, harmful-case
rescues, lost baseline successes, material and runtime. Include clean cases.

Completion requires a genuine trained checkpoint and actual local-model decisions
in the printing loop, strictly more functional completions than the strongest
prior method on the frozen fresh comparison, and a historical regression run
against contour's published 62/100. Any worse defect rate or lost passing case
must be resolved or explicitly reported as an unmet criterion. The agent ablation
must show whether supervision helps; do not attribute the network's gains to it.
Simulation evidence is not a physical-printer validation.

Long runs use tmux with logs, manifests and exit codes on the data volume. Reuse
the existing Torch environment for training and export small NumPy weights for
inference in the project's existing environment. Respect the shared 100 GB quota
and avoid taking over the GPU already used by other jobs.

## Original frozen comparison matrix (completed and rejected)

The original final grid used seeds 1001--1004, all three printer profiles, all four
parts, all five failure kinds and clean controls: 240 failure cases and 48 clean
cases per method. Freeze source, weights and options before the first case. The
required earlier baselines remain `ignore`, `scrap`, `alert`, `ours`, `measured`,
`predictive`, `functional`, `guarded` and `contour`; none is removed after seeing
results.

The additional comparisons are the analytical preventive planner (`contour_risk`),
learned camera on all parts (`neural_camera`), the combined model without
preventive recovery (`neural`), the complete model without Qwen (`neural_risk`),
and the earlier fit-only forecast hybrid using classical measurements everywhere.
The latter retains its raw `neural_risk` records and receives the explicit report
label `neural_risk_classical`, with its configuration and raw hashes preserved.
The candidate is the complete model with actual Qwen decisions
(`neural_risk_agent`). This is fifteen configurations, including the candidate.

The candidate must strictly exceed every earlier baseline and also attain the
best completion count among the included neural variants. A tie with the same
model without the agent is reported as a tie, not a benefit from supervision.
The audit lists all gained and lost paired cases and checks defects, clean
controls, real model calls, both trained checkpoint identities and disjoint
training/validation/test seeds. Raw records are never relabeled or edited to
combine variants; report labels are explicit input metadata. Historical
regression and the other acceptance requirements above remain necessary.

Seeds 1001–1004 are now consumed development evidence. The next acceptance run
must freeze the final candidate before using reserved seeds 2001–2004 on the same
three-profile, four-part, five-failure-kind plus clean grid. It must include the
required earlier baselines and separately pinned actual strongest prior neural
implementations. A newer implementation under an old method label does not
substitute for those earlier versions. No fresh cases have been run yet.

An independent [checkpoint recheck](validation/neural/training-provenance-check.json)
loads the retained datasets and frozen NumPy weights, verifies their hashes and
whole-trajectory splits, and reproduces the recorded validation errors exactly:
0.19210984 mm temporal displacement and 0.07919747 mm boundary distance. Both
models use 80 training groups and 40 validation groups, with no overlap with
test seeds 1001–1004. These metrics verify the trained artifacts, not recovery.

The test started at 2026-11-02 21:13:45 UTC from commit
`a27847b324da18b6c40b9b8108b6240d022c8892`, with runtime/specification digest
`4ed28f4355b9cc923b727ca90a67cee279e524b10c6764b4d5ec5585a60f49f3`.
All 4,320 runs use the frozen source and copied checkpoints under
`/mnt/data/ishaan/hri-workspace/failure-detection/neural-frozen-test-v1`.
The [launch protocol](validation/neural/frozen-test-v1-protocol.json) records
every command, model hash and comparison label. Seeds 1001–1004 are now consumed
by this test and must not be reused as untouched data for a tuned replacement.
All 4,320 runs and both final audits complete successfully at 23:46 UTC. The
outcome criteria fail: the candidate completes 126/240 failures, versus contour's
128/240 and preventive analytical recovery's 133/240. It loses ten contour
successes and thirteen preventive-analytical successes, while gaining eight and
six respectively. It finishes five defective prints, versus six and three.
All methods pass their 48 clean controls. Qwen makes 488 actual requests but
reduces the unsupervised combined policy's 129 completions to 126, losing three
cases and adding one defective finish. The complete ranking and every paired
gain/loss are retained in `frozen-test-v1-comparison.json`. Successful audit exits
confirm integrity, not superiority; `outcome_criteria_passed` is false.

A supplementary integrity audit was scheduled at 21:23 UTC, without changing
the frozen policy, weights, data or outcome criteria. It verifies that a finished
print contains every nominal layer, successful prints have no final violations,
raw persistence flags match the protocol, clean controls contain no injected
failure, and paired runs agree on onset, severity and nominal layer count across
profiles and separately launched jobs. The same checks pass all 600 historical
combined-model records; nine corruption/validity tests cover the added checks.
The [audit addendum](validation/neural/frozen-test-v1-integrity-addendum.json)
pins both audit scripts. Its completed `test-integrity` check
retains separate outputs under `integrity-v2`, and explicitly
distinguishes completed audits from passed outcome criteria.

### First frozen candidate rejected; revision under development

At 21:38 UTC, completed paired records already contradicted the no-lost-success
criterion: Cartesian cup collapse, seed 1001, finishes under contour but is
scrapped by the combined candidate. The
[retained counterexample](validation/neural/frozen-test-v1-early-rejection.json)
pins both records and their matched failure conditions. Candidate records also
contain defective bracket and mug finishes. The original 4,320-run benchmark
has now completed unchanged and confirms the early rejection. Seeds 1001–1004 are now diagnostic data,
never untouched evidence for a revision. Seeds 2001–2004 are
[reserved](validation/neural/next-test-seed-reservation.json) for the next frozen
test; they have not been launched.

One bracket failure exposed an action-identifier ambiguity: Qwen described the
best-ranked hole-clearance plan but returned candidate ID 1, which referred to a
different plan. Policy requests now number executable choices in rank order so
rank 1 and ID 1 identify the same plan. Two actual response replays select the
intended clearance plan after the fix. This is diagnostic evidence, not a new
closed-loop success count; the response still determines the executed plan.

A 64/32-unit camera network trained with rotations and reflections worsened
validation boundary error from 0.0792 to 0.0887 mm. Its weights, training record
and logs are retained in `neural-boundary-v2`; it is not selected as an improvement.
The next policy instead tests `--boundary-fallback`: preserve any feasible plan
from classical camera measurements, and consult a separate learned-camera
observation history only when the primary pipeline finds no feasible option.
Both views measure every actual frame; neither rewrites earlier observations.
Alternative plans must pass the original complete specification and action
constraints. The trained fit forecast remains paired with classical measurements.
This revision still needs closed-loop validation and a new frozen comparison.

The fallback run in `neural-camera-fallback-dev-v1` includes 128 diagnostic runs
on consumed seeds 1001–1002, followed by 144 validation runs on seeds 70–71.
Completed records restore the lost cup-collapse case with and without Qwen.
The bracket-collapse case at seed 1001 now completes with two adaptations under
both variants, whereas the original agent finished it defective. However, the
fallback loses the cup-shift success at seed 1002. The full study is complete;
these mixed results do not qualify this revision as an improvement.

An equivariant model trained with current-state calibration reduced validation
displacement error to 0.1611 mm versus 0.3043 mm analytically on the same
calibration dataset. It still scrapped the recoverable cup thinning case at seed
1002, as did contour; ignore finished successfully. This failed diagnostic is
retained separately and the model is not promoted on its lower point error.

The original camera dataset divided selected frames among the three profiles;
brief thinning events therefore had sparse coverage in each view. The next
dataset preserves the first ten eligible frames densely and renders every
selected frame with all three cameras. Additional cup/mug clean and thinning
trajectories use training seeds 64–67 and validation seeds 72–73, combined with
all 120 retained original trajectories. The
[dense-camera protocol](validation/neural/dense-camera-v3-protocol.json) records
the 144-group collection and training job in `neural-dense-camera-v3`.
Training is complete. On the same 676,800 validation patches from 48 trajectories,
the old model has 0.0837853 mm boundary error and the dense model has 0.0730932 mm.
The [matched comparison](validation/neural/dense-camera-v3-comparison.json)
retains both hashes and all per-fault metrics. False-negative material pixels
increase from 12,360 to 12,974 despite the lower average distance error.

`tools/inspect_camera_case.py` then follows the unchanged nominal commands for
the consumed cup-under-extrusion case at seed 1002. Every segmenter receives the
same actual camera frames. Truth is used only for diagnostic scoring, never as
a policy input. The actual 160-layer deposit passes; classical, original neural,
and dense neural measurements all report two falsely opened walls and a false
support violation. Raw images and geometry snapshots at layers 84–87 are retained
in `neural-dense-camera-check-v3`. This is a measurement diagnostic, not a
closed-loop recovery win. The dense checkpoint is not promoted on its patch error.

### Native subpixel output under development

The original camera network predicts a single signed distance at each camera
pixel centre. Upsampling those predictions cannot represent material between
centres if all surrounding values are negative. An opt-in checkpoint format now
supports a 4-by-4 set of independently trained distances per input patch, with
explicit image-coordinate offsets. These native predictions represent a
0.0625 mm grid using the existing 0.25 mm camera pixels; they do not increase
the camera's physical information. Old checkpoints retain their prior behavior.
Training targets use the true distance at each exact subpixel position. No
morphological rule fills gaps or alters earlier observations.

The [native-output protocol](validation/neural/native-camera-v4-protocol.json)
uses the same 144 disjoint whole trajectories, 12 dense frames per trajectory
under all three camera profiles, and 120 boundary samples per frame. A 64/32-unit
network learns 16 distance outputs. The pipeline in tmux window `native-camera-v4`
builds data, trains, checks its NumPy export, and repeats the consumed camera
diagnostic. That pipeline is complete: native-grid validation error is 0.07372 mm,
but the diagnostic still falsely opens both thin walls and introduces three
false base-hole reports. This model is not promoted. The
[matched pixel-centre comparison](validation/neural/native-camera-v4-comparison.json)
separates common-centre errors from the native subpixel metric. The source suite
passes 94 tests with
one optional module skipped; new checks cover native output alignment, material
between negative pixel centres, and checkpoint-grid mismatches. Test seeds
2001–2004 remain untouched.

An isolated [analytical-inverse ablation](validation/neural/analytical-inverse-results.json)
retained the trained forward forecast while constructing corrections with the
analytical residual field. It restored two consumed spacer failures, but lost
all three earlier spacer-shift gains at historical seed 10. Both completed
studies pass integrity audits. This ablation remains outside the production
runtime and is rejected as a replacement; changing the inverse alone is insufficient.

The next [joint-forecast diagnostic](validation/neural/joint-inverse-results.json)
retains both inverse constructions and checks each eligible plan under both
forward forecasts. When jointly passing plans exist, it ranks those using the
smaller fit slack from the two forecasts. When none exists, it retains the
original neural candidates. Both estimators receive the same camera history;
neither sees the injected failure or actual deposit. This restores both consumed
spacer regressions and retains all three historical spacer-shift gains: all seven
failure cases and both clean controls pass, with no defective finishes. Contour
passes three of the seven failure cases in this small diagnostic.

`--neural-joint-forecast` now enables this experimental fit check. The default
remains disabled. Rows record actual analytical candidate checks; agent requests
include whether both forecasts pass. Tests verify that a disagreement excludes
a plan only when a jointly passing alternative exists, and that neither the
original candidates nor measured prefix are mutated. The
[270-run broader study](validation/neural/joint-fit-dev-v2-protocol.json) is now
complete in `neural-joint-fit-dev-v2`, covering the 50 historical fit failure cases
and ten clean controls, consumed diagnostics, and validation with actual Qwen
decisions. Its source snapshot is pinned to commit `97c3b38` and runtime digest
`7fc3434ac95611a4d62784f1c902c5f965d858692b2b23b523332142529e6dbf`.
The historical fit results are 33/50 for both neural variants versus 30/50 for
contour, with zero versus one defective finish and all ten clean controls passed.
Qwen made 40 actual requests in these historical runs. On the separate validation
slice, both neural variants achieve 12/20 versus 10/20, zero versus one defective
finish, and all four clean controls; Qwen makes 32 actual requests. The three
gains are spacer collapse at seeds 70 and 71 and spacer shift at 71. However,
both variants lose bracket collapse at seed 71. Aggregate gains do not satisfy
the no-lost-success criterion. Qwen ties the neural-only outcome in both sets.
Compact audits and CSV rows are retained under `joint-fit-v2-*` in the evidence
directory. This is not a full-project improvement or an untouched-test result.

The [paired timing regression](validation/neural/joint-fit-v2-validation-regression.json)
shows why: the neural forecast accepts continuation at layer index 27 and skips
the analytical assessment because it has only one passing candidate. By the next
layer the measured prefix has already lost its fit. Contour intervenes at 27 and
completes. The revised optional joint check evaluates this continuation under the
analytical forecast before taking that fast path. If it signals failure or low
fit slack, both inverse constructions are considered; replacement still needs
a jointly passing alternative. When continuation passes both forecasts, an early
replacement needs more than half a camera pixel of additional worst-case central
fit slack. With no jointly passing alternative, the original neural choices remain.

The [early-check protocol](validation/neural/early-joint-fit-dev-v3-protocol.json)
freezes this revision and starts 39 diagnostic runs with actual Qwen on localhost
8089. Only if all candidate prints in those selected cases complete functionally
does it proceed to the full 50-case historical fit comparison and 20-case fit
validation, each with clean controls. The first bracket stage is complete: both
failure cases and both clean controls pass for contour, neural and actual Qwen
(four model requests). The complete diagnostic then fails its gate: spacer shift
at seed 1002 finishes defective with and without Qwen. The analytical warning
triggers a second correction at layer 28, although its worst fit bound is weaker
than the already passing neural continuation. All three historical spacer-shift
gains remain, but this new defective finish rejects the variant. The broad rerun
is skipped as predeclared; the failed gate and all 39 audited rows are retained.

The [robustness revision](validation/neural/robust-joint-fit-dev-v4-protocol.json)
permits an early, disagreement-triggered replacement only if its fit bound under
both forecasts is at least as good as the original passing neural continuation's
bound. A jointly passing plan alone is insufficient. If no such plan exists,
the original neural choice remains. The bracket and spacer stages now complete:
both neural variants pass all six failure diagnostics and all four clean controls,
with no defective finishes, versus contour's five of six failures. Qwen makes
20 actual requests. This restores the one-adaptation spacer success while keeping
the bracket rescue. All three historical gains also pass. The completed broader
fit comparison scores 33/50 historical and 13/20 validation, versus contour's
30/50 and 10/20, preserving all its successes with zero defective finishes.
An explicit comparison against the actual earlier neural source confirms the
validation gain from 12/20 to 13/20 without losses. `fit-consumed-grid-v5` completes
all 576 runs over 120 failures and 24 clean controls per method across all three
profiles. Both neural variants score 78/120, versus 76/120 for analytical preventive
recovery and 71/120 for contour. Both neural variants and analytical preventive
recovery have zero defective finishes; contour has three. Every method passes all
24 clean controls. Qwen makes 158 validated requests and ties neural-only outcomes.
Against analytical preventive recovery there are three gained successes and one
lost success: CoreXY bracket shift, seed 1001. Thus the completion count improves,
but the baseline-success preservation requirement fails. The neural policy waits
at layer index 26 and scraps at 27 after the measured hole becomes irreparable.
The analytical baseline adapts at 26 and completes. A candidate-only replay shows
that the analytical correction passes both central forecasts but has a neural
fit uncertainty bound of -0.150 mm, below the original continuation's -0.014 mm;
the existing early-intervention guard rejects it. This is a development regression,
not a fresh acceptance test.
Reserved seeds 2001--2004 remain untouched.

The optional `--neural-hole-envelope` experiment retains both inverse proposals'
hole openings in future commands while keeping the neural proposal's exterior.
It only expands the search when an early neural continuation passes and the
existing recovery candidates fail the unchanged joint robustness guard. It never
edits observations or the printed prefix. Every new command still undergoes both
forward forecasts, the original functional and executable checks, and the same
fit-bound comparison. It requires `--neural-joint-forecast` and remains off by
default. Candidate-only diagnostics reject simple mixed inverses, but the shared
hole proposal improves the limiting bound to +0.047 mm on the lost bracket.

The frozen `hole-envelope-fit-dev-v9` protocol completes its 51-run gate: that bracket,
earlier spacer and bracket regressions, three historical gains, a validation gain
and clean controls, with actual Qwen. The lost CoreXY bracket now completes with
one adaptation for both neural variants, matching analytical preventive recovery.
Both neural variants pass all 11 selected failures and six clean controls, with
no defective finishes and 38 actual Qwen requests. The broader neural-only grid
also completes: 79/120 failures versus analytical preventive recovery's 76/120
and the actual preceding neural implementation's 78/120. There are no lost
successes, no defective finishes, and all 24 clean controls pass. The paired
Qwen grid also finishes at 79/120 with 161 actual calls and the same preserved
successes and controls. This remains consumed development evidence.

The same frozen source completes `hole-envelope-fit-history-v10`: 252 new runs
cover all 50 historical fit failures, 20 validation fit failures and their clean
controls. Neural-only and Qwen tie the previous best at 33/50 historical and
13/20 validation, versus analytical preventive recovery's 31/50 and 12/20.
No baseline or previous-neural success is lost; there are zero defective finishes,
and all ten historical and four validation clean controls pass. Qwen makes 40
historical and 32 validation requests. Comparisons pin the actual earlier
implementation separately. These ties are preservation evidence, not new gains
over the previous neural policy. Reserved test seeds remain untouched.

### Camera uncertainty and additional deformation coverage did not resolve the diagnostic

An 80th-percentile signed-distance network attained 80.6% boundary-point coverage
on the native-output validation set. The recoverable 160-layer cup diagnostic
still reports two false wall gaps and six false base-hole layers. Point coverage
is not a calibrated whole-print success probability; the checkpoint is rejected.

The next dataset adds 18 whole trajectories at a previously missing range of
under-extrusion severity: training seeds 313, 335, 344, 355, 379 and validation
seeds 400, 410, 417, 420, for cup and mug. Seeds were chosen prospectively from
the generator's severity range, without inspecting printing outcomes. Training
uses all base frames and extra boundary samples near deposited deformation;
only grayscale patches enter the model. Combined splits contain 106 training
and 56 validation trajectories, with no group overlap. Two-dataset training
records both input hashes and rejects repeated groups.

The central estimate reaches 0.07203 mm native-grid validation error, while the
upper estimate reaches 77.2% boundary-point coverage and 0.09465 mm error on this
expanded dataset. Neither passes the fixed cup diagnostic: the central model
still reports two false wall gaps and three false base-hole layers; the upper
model reports one gap, five base-hole layers and two thin-wall layers. The
predeclared conditional recovery trials were therefore skipped, with explicit
records. No checkpoint is promoted. The initial retained-geometry validation
fails after recording 960 frames from 18 groups: an invalid pixel contour causes
GEOS to reject the polygon union. The failure manifest remains in the evidence.
Individual contours are now made valid before union, preserving polygonal pieces
and the original area and outline corrections. A deterministic crashing-mask
regression and a visible-hole/island preservation test pass. The full 36-group,
2,040-frame validation restarts from a new frozen snapshot in `camera-geometry-v7`;
the original failed run is not overwritten.

This conversion repair also changes some tiny base-hole classifications. An
[identical-mask comparison](validation/neural/camera-contour-repair-paired-mask.json)
reproduces a 0.000335 mm2 hole under the old conversion and none under the repair.
Its raw background touches the exterior diagonally (8-connectivity), but not
through pixel edges (4-connectivity). This is a pixel-topology ambiguity at the
boundary, not proof that geometry conversion has no effect on measurements.
Raw masks, hashes and crops are retained; all models undergo the same revised
validation. The original functional checks are unchanged.

A [shared-query camera experiment](validation/neural/query-camera-v7-protocol.json)
reuses the same 106/56 trajectory split. Its 81-to-64-to-32 network adds the
subpixel XY coordinate before a shared scalar distance decoder, replacing 16
independent heads. This represents a continuous function within each pixel;
continuity across different pixel patches is not guaranteed. Training completes
80 epochs and exports matching NumPy weights, with 0.07351 mm validation error
versus the independent-head model's 0.07203 mm on the same dataset. The completed
160-layer cup diagnostic still reports two false wall gaps and a 27 mm2 false
overhang; the conditional recovery trial is skipped. Under the repaired contour
conversion neither this model nor the earlier central model reports a base hole
in this diagnostic, so that improvement cannot be attributed to the new network.
The shared model remains unaccepted; its 36-group geometric validation continues.
It uses no new test seeds or datasets.

The camera-fallback diagnostic also finishes at 16/24 for both neural variants,
matching contour's 16/24 but with one versus zero defective finishes. This is
another rejected replacement. Its completed 40-case validation scores 21/40
versus contour's 19/40 and ignore's 20/40, with zero versus one versus twenty
defective finishes; all eight clean controls pass. That validation excludes Qwen
and does not erase the diagnostic regression.

### Smaller actual Qwen requests preserve eligible choices

The supervisor now sends every eligible choice while keeping rejected and
dominated candidates in the full controller audit. The independent auditor
checks equality of the schema, transmitted and raw eligible-ID sets. In an
[actual replay](validation/neural/compact-agent-replay-v1.json), a 6,531-character
user payload became 944 characters, retained both eligible choices, and selected
the same plan. Its 6.65-second response is interface evidence, not a completed
print; different server concurrency prevents attributing all latency improvement
to payload size.

## Development record (2026-11-02)

Qwen3-0.6B Q8_0 and llama.cpp b12057 are downloaded with pinned SHA-256 checksums.
The CPU server listens on `127.0.0.1:8089`. A real structured decision and its
request/response are retained in the data volume's `local-supervisor-v1/smoke.json`.
The printing-loop evaluation also records real model calls in each action audit.
No Gemini or other remote inference credentials are needed.

The first temporal network was trained on 388,692 synthetic-camera examples from
80 trajectories, with 155,924 validation examples from 40 separate trajectories.
Validation mean displacement error: analytical 0.3501 mm, persistence 0.3077 mm,
neural 0.2209 mm. The checkpoint is
`dbb507d434674476c8acb6fadc5f6c4d7435e4c80839819c3379193dd91c91a8`.
These are point-displacement metrics, not successful-print rates.

That first model is not accepted as an improvement. On eight previously published
MuJoCo trajectories, both contour and neural models made zero false failure calls
in 70 snapshots of successful prints. However, the neural model missed 19 of 42
snapshots of ultimately failing prints, versus 11 for contour. Those snapshots
are correlated observations, not 42 independent trials. The same-frame diagnostic
and first closed-loop development comparison are retained, including failures.

Two follow-up experiments use actual MuJoCo camera observations: future-only
forecast training, and current-frame calibration plus forecasting. Calibration
is opt-in, requires explicit current-state training targets in the checkpoint,
moves an observed boundary by at most one camera pixel, preserves its topology,
and never edits an earlier frame. It remains experimental until closed-loop
results support it. Full geometry trajectories are retained in the calibration
run so later training ablations can reuse them.

The first closed-loop development trial (32 runs: two parts, three failure kinds
plus clean, seed 70, four methods) tied the contour baseline: 2/6 functional
failure-case completions for contour, neural, and neural plus Qwen. All three
avoided defective finishes and passed both clean cases. The agent made 77/77
validated requests, with no recovery gain. Its failure-case runtime was 425 s,
versus 106 s for neural-only and 128 s for contour on the shared machine. Later
runs reuse a recent review for at most eight layers only when a freshly checked
continuation is the sole allowed action. Every actual recovery choice still
requires a model request; reused reviews are recorded and are not counted as
inference. This reduces redundant calls without skipping function checks.

Completed evidence is in [validation/neural](validation/neural). No model has yet
met the final superiority criterion, and the existing default policy is unchanged.

Artifacts live under `/mnt/data/ishaan/hri-workspace/failure-detection/`:
`neural-temporal-v1`, `neural-forecast-dev-v1`, `neural-closedloop-dev-v1`,
`neural-temporal-mujoco-v1`, and `neural-temporal-calibration-v1`.
Attach to all jobs with `tmux attach -t neural-recovery`. Source snapshots freeze
each run independently; ongoing code changes do not change an in-flight model.

## Camera, forecasting and preventive recovery follow-up

Training on actual MuJoCo images and adding optional current-frame calibration
both tied the initial 2/6 closed-loop result. Neither change established better
recovery. A separate camera network learned signed boundary distance from 9 by 9
grayscale patches, with whole-trajectory train/validation separation and camera
augmentation across the three rigs. Its validation boundary error was 0.0792 mm.
That pixel metric does not establish print success or transfer to real cameras.

On the 16 Cartesian seed-10 failure cases excluding detachment, the learned camera
and analytical forecast completed 14/16, matching contour. It stopped the one
defective spacer finish that contour missed. Qwen supervision tied those outcomes,
with 42 actual calls and 191 explicitly recorded reused continuation reviews. All
four clean controls passed. Detachment remains in the wider regression and final
comparison; its exclusion here was only a focused camera development trial.

The spacer's fit can become irrecoverable between the first and second shifted
layers while the first central forecast still passes. The preventive policy now
checks signed hole-clearance slack against observed residual uncertainty. It
considers an early intervention only when a feasible candidate improves central
clearance by more than half a camera pixel. Candidates include gradual enlargement
of an existing hole in future layers, bounded by the original allowed diameter
and the existing 0.25 mm/layer correction assumption. The printed prefix, function
specification and three-adaptation limit remain unchanged. Buffered forecast
scenarios rank candidates by signed fit clearance; these scores are not
calibrated probabilities or physical guarantees.

Earlier intervention alone failed all three seed-10 spacer-shift cases. Training
the temporal network in coordinates aligned with observed motion then reduced
validation displacement error to 0.1921 mm (classical measurements, versus the
analytical 0.3435 mm) and 0.1718 mm (learned camera, versus 0.2929 mm). Rotating an
observed scene rotates this model's forecast; zero motion produces zero predicted
motion. Checkpoints declare their measurement source, which the evaluator checks
to prevent mixing a model with a different camera pipeline.

With classical camera measurements, the rotation-equivariant network and bounded
fit-reserve planner completed 3/3 selected spacer cases, versus 0/3 for contour
and the same network without preventive recovery. The learned-camera combination
completed 2/3. These are previously inspected development cases, not an unbiased
success-rate estimate. Raw Qwen choices reduced those results to 1/3 and 0/3,
respectively: the small model chose weaker recovery options despite being given
their numerical assessments. The tool now ranks choices and excludes materially
weaker fit-risk candidates whose other functional differences are below explicit
measurement tolerances. The remaining genuine tradeoffs still go to Qwen. This
guard was then tested in a new actual-agent trial: the supervised policy completed
3/3, matching the neural policy without Qwen, with 10 validated model requests and
no defective finishes. Thus supervision preserved those gains; it did not add
another completed print. The model's prose is never numerical evidence of a
successful recovery. A matched `contour_risk` planner ablation separately tests
whether the new planner can recover the same cases using the analytical forecast.
That completed ablation recovered 1/3 with analytical forecasting and 3/3 with
learned forecasting, so the planner alone does not explain all three rescues.

The unrestricted learned forecast lost container successes in the wider
historical regression, including cup shift and cup/mug under-extrusion. It is not
an acceptable replacement everywhere: the completed 60-case main grid scored
35/60 for unrestricted neural recovery versus 40/60 for contour, with two versus
one defective finishes. There were three gained spacer cases and eight lost
container cases. An explicit `--neural-forecast-scope fits`
variant chooses the trained model only for parts whose original specification
contains through-hole fits; other parts retain analytical contour forecasting.
This choice uses the functional specification before observations, not fault
labels, simulator outcomes or seed identities. The full benchmark still contains
every part and failure kind. Each row records the selected forecast source and
actual neural calls, including zero neural calls for analytical cases. A new
full historical run compares this hybrid, with and without Qwen, against contour.
Its completed 60-case main grid now scores 43/60 for both hybrid variants versus
40/60 for contour, with zero versus one defective finish. Every contour success
is retained, all 12 clean controls pass, and the agent makes 121 validated calls
with 577 separately audited continuation-review reuses. The subsequent 40-case
historical slice completes at 22/40 for all three methods, yielding 65/100 for
both hybrids versus contour's 62/100 overall. The frozen fresh test remains required.

The unrestricted model's full 100-case historical run is now complete: 56/100
functional completions versus contour's 62/100, with three versus one defective
finishes. All nine lost contour successes were containers; all three gains were
spacer shifts. Both methods passed the 20 clean controls. This failed replacement
remains in the evidence rather than being hidden behind the successful subset.

A separate learned-camera check also restored the known bedslinger mug-shift
success that contour had lost to guarded. Across the three profiles, guarded and
the learned camera each completed 3/3 mug-shift cases; contour completed 2/3.
The next full variant therefore combines learned camera measurements for parts
without through-hole fits with the trained temporal forecast for fit parts.
Use `--boundary-scope nonfits --neural-forecast-scope fits`; this retains the
classical measurements on which the fit forecast was trained. Invalid camera/
forecast domain combinations are rejected before a run. Every row records its
actual measurement source, boundary inference count, forecast source and temporal
inference count. The original benchmark parts, failures and specs remain intact.

The combined comparison runs in `neural-combined-historical-v1` (deterministic)
and `neural-combined-agent-historical-v1` (actual Qwen), using the exact same source
snapshot and two checkpoints. A second CPU inference worker on localhost:8090
shares the verified Qwen model file so it does not queue behind the earlier
hybrid trial on localhost:8089. No second download or API access is required.
Both combined runs are complete: each scores 67/100, versus contour's 62/100,
guarded's 43/100 and ignore's 53/100. No passing case from those baselines is lost;
the 38 successes recorded for the original rule policy are also retained. The
combined candidate has zero defective finishes and passes all 20 clean controls.
Actual Qwen supervision makes 201 validated calls and matches the neural-only
result. The earlier classical-camera hybrid finishes at 65/100, also without
lost contour successes or defective finishes. These results qualify the combined
candidate for the frozen test; they do not replace that test. The full source test suite passes
77 tests, with one optional module skipped.
The operator-facing agent reason now states only the selected candidate and the
verified prediction check. Qwen's original prose remains in the raw audit;
incorrect numerical explanations are not presented as tool findings. This changes
display text, not model requests or selected plans.

One preceding reserve-planner experiment failed after a local-model timeout; its
failed manifest is retained. Requests now contain compact decision fields and
use a 120-second timeout. Per-case journals preserve completed methods and agent
errors even when a later method fails. Errors still fail the experiment; they do
not silently substitute an unsupervised decision.

All completed comparisons, including unsuccessful ablations, have compact audits
and training manifests in `docs/validation/neural`. Full source snapshots, model
weights, training data and request/response logs remain on the data volume. The
100-case historical regression has passed; the frozen fresh comparison is still
required before accepting the new policy as better overall.
