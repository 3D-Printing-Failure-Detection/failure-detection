# Learned recovery and local supervision

The objective is more functional completed prints and genuine recovery from
failures, compared on identical cases with every earlier policy. A trained model
or a successful language-model request alone does not satisfy that objective.

Current verdict: ahead on historical cases, not yet accepted as better overall.
The original combined candidate reached 67/100 versus contour's 62/100 but failed
its complete fresh test: 126/240 versus contour's 128/240 and the analytical
preventive planner's 133/240. It finished five defective prints versus three for
that strongest baseline. The newer paired-forecast fit policy reaches 33/50 versus
30/50 historical fit failures and 12/20 versus 10/20 validation failures, with
zero versus one defective finish in each set. It still loses a bracket case
that contour saves. The first earlier-continuation check restored that bracket
but harmed a spacer. A revised guard compares fit robustness before intervening.
It passes all nine selected failure cases and four clean controls with actual
Qwen, and its completed 50-case historical fit comparison scores 33/50 versus
30/50, with zero versus one defective finish, every contour success retained,
all ten clean controls passed and 40 actual Qwen requests. Separate fit validation
finishes at 13/20 versus contour's 10/20 and the earlier neural revision's 12/20,
with zero defective finishes, no lost successes against either, all four clean
controls passed and 32 actual Qwen requests. Qwen matches neural-only completion.
New camera models remain experimental.

## Current camera cross-check and context experiments

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

The current source passes 130 tests, with one optional module skipped. Long runs
remain in tmux session `neural-recovery`; attach with
`tmux attach -t neural-recovery`. Full evidence stays on the data volume, with
compact manifests and completed audits under `docs/validation/neural`.

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

## Frozen comparison matrix

The reserved final grid is seeds 1001--1004, all three printer profiles, all four
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
validation gain from 12/20 to 13/20 without losses. `fit-consumed-grid-v5` now
tests every consumed fit case across three profiles, including the strongest
analytical preventive baseline: 576 runs over 120 failures and 24 clean controls
per method. This is a development regression, not a fresh acceptance test.
Reserved seeds 2001--2004 remain untouched.

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
