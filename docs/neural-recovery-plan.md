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
