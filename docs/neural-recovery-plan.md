# Learned recovery and local supervision

The objective is more functional completed prints and genuine recovery from
failures, compared on identical cases with every earlier policy. A trained model
or a successful language-model request alone does not satisfy that objective.

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
Complete results are pending, but the candidate has already failed acceptance
as recorded below. Attach with `tmux attach -t neural-recovery`; the
`test-status` window reports progress and `test-audit` runs the final comparison
after all jobs finish. A successful audit exit confirms integrity, not superiority:
all outcome criteria in `comparison.json` must also pass.

A supplementary integrity audit was scheduled at 21:23 UTC, without changing
the frozen policy, weights, data or outcome criteria. It verifies that a finished
print contains every nominal layer, successful prints have no final violations,
raw persistence flags match the protocol, clean controls contain no injected
failure, and paired runs agree on onset, severity and nominal layer count across
profiles and separately launched jobs. The same checks pass all 600 historical
combined-model records; nine corruption/validity tests cover the added checks.
The [audit addendum](validation/neural/frozen-test-v1-integrity-addendum.json)
pins both audit scripts. Its `test-integrity` tmux window waits for the original
comparison, retains separate outputs under `integrity-v2`, and explicitly
distinguishes completed audits from passed outcome criteria.

### First frozen candidate rejected; revision under development

At 21:38 UTC, completed paired records already contradicted the no-lost-success
criterion: Cartesian cup collapse, seed 1001, finishes under contour but is
scrapped by the combined candidate. The
[retained counterexample](validation/neural/frozen-test-v1-early-rejection.json)
pins both records and their matched failure conditions. Candidate records also
contain defective bracket and mug finishes. The original 4,320-run benchmark
continues unchanged; a partial counterexample proves rejection but does not
supply an overall success-rate estimate. Seeds 1001–1004 are now diagnostic data,
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
