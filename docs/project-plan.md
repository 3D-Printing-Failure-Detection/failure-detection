# Project 2 plan: functional recovery from 3D print failures

*Oct 27, 2026 · @Sohum* (original: [project-plan.pdf](project-plan.pdf))

## Problem and goal
When a print fails mid-way, today's options are to scrap it or keep printing a defective part. Both waste hours and filament. Our goal is a feedback loop that catches the failure during the print and adapts the rest of the print so the part still works, even if it looks different.

Example from class: a mug's wall deforms so its circular cross-section becomes a teardrop. Instead of scrapping it, the system continues as a teardrop mug that still holds liquid and keeps its handle. Function is preserved; aesthetics change.

## The feedback loop
Every layer is imaged and checked. On a deviation, the system measures the as-printed shape, checks the part's function spec, and either continues, adapts the remaining layers, or scraps.

## Scope for 4 weeks
The 4-week version targets simple parts where function can be checked from cross-sections, and deformations that leave a usable base to build on.

| | In scope | Out of scope |
|---|---|---|
| Parts | 2.5D parts whose function is set by cross-sections: cups and mugs, containers, brackets, spacers | Complex organic shapes, multi-material, supports-heavy parts |
| Failures | Wall deformation, warping, partial layer shift, under-extrusion of a section | Spaghetti or a detached part (nothing to build on), so these go to scrap |
| Function | A short spec per part, e.g. holds liquid (closed walls, flat base), volume within a tolerance, minimum wall thickness, handle attached | Strength or load testing beyond simple checks |
| Printer | Pause, adapt remaining G-code, resume on one printer we have access to | Fully hands-off physical removal or ejection |

## Components and models
The trained models cover detection and measurement; adaptation is geometry code with an LLM or rules choosing the strategy.

| Stage | What it does | How |
|---|---|---|
| 1. Detect | Flag that a layer deviates and classify the failure type | Fine-tuned image classifier or detector on layer images (trained model) |
| 2. Measure | Extract the as-printed contour of the current layer and compare it to the expected contour from the G-code | Fine-tuned segmentation model on top-down images, registered to the rendered expected layer (trained model) |
| 3. Assess | Check whether the part can still meet its function spec | Rule checks on the measured shape: closed walls, wall thickness, volume estimate |
| 4. Decide | Continue, adapt or scrap | Policy over severity, functional slack, time and filament left; rules first, LLM planner as an option |
| 5. Adapt | Re-plan the remaining layers to start from the deformed contour and still meet the spec (circle becomes teardrop) | Morph remaining cross-sections from the measured contour, re-slice from the current height, resume |
| 6. Interface | Show the deviation, the decision and a preview of the adapted part | Gradio app; printer control through Klipper/Moonraker or OctoPrint |

## Data
Most training data will be synthetic, because printer time is limited; real prints are for validation.

| Source | What it gives | Size and caveats |
|---|---|---|
| Synthetic layers | Expected layer renders from sliced G-code with injected deformations (warp, shift, wall collapse); exact ground-truth contours | Unlimited; PrusaSlicer runs headless at under 0.5 s per slice. Gap to real camera images |
| Slice-100K | 100k models with G-code to slice and deform | Download location to confirm |
| CAXTON (Cambridge, 2022) | 1.2M real print images from 192 parts, labelled with process parameters | Labels are parameters, not geometry; public availability to confirm |
| Public failure images (Roboflow, Hugging Face) | Labelled failure types for the detector | Small (the Hugging Face set has 73 images); mixed licences |
| Our prints | Real layer images with deliberately induced failures, plus end-to-end demos | Roughly 5-10 prints given printer time |

## Evaluation
The main question: when a failure happens, how often does the loop end with a part that still meets its function spec, compared with scrapping or ignoring it?

- Detection and measurement: classification accuracy and false alarms; contour error (IoU, mm deviation) on held-out synthetic and real layers.
- Recovery in simulation: inject failures at random heights in sliced test parts, run the loop, and check the final geometry against the spec. Report the share of functional parts plus filament and time saved.
- Baselines: (a) ignore the failure and keep printing, (b) scrap and restart, (c) pause and alert a human.
- Physical validation: a few real prints with an induced failure, resumed with adapted G-code; check function directly (e.g. fill the mug with water).

## Week-by-week plan
1. **Week 1: problem and pipeline.** Interview a few makerspace or print-farm users about failures. Pick 3-5 test parts and write their function specs. Build the slicer-to-render-to-deform pipeline. Mount the camera and test pause, edit and resume on the printer with one manual example.
2. **Week 2: models.** Generate synthetic data; fine-tune the detector and the contour segmentation model. Collect the first real layer images with induced failures.
3. **Week 3: the loop.** Implement assess, decide and adapt (contour morphing and re-slicing from the current height). Run the loop end to end in simulation; first physical adapted print.
4. **Week 4: evaluate and ship.** Baseline comparisons, real-print validation, Gradio app, report and presentation.

Concept generation and selection for the product-development write-up (scrap-and-requeue agent vs. functional adaptation vs. alert-only) runs alongside week 1.

## Prior work and what's new
Existing closed-loop systems correct process parameters, not geometry. Changing the remaining design mid-print to keep the part functional appears to be open.

| Work | What it corrects | Gap for us |
|---|---|---|
| CAXTON (Cambridge, 2022) | Flow rate, speed, temperatures, Z offset, in real time | Parameters only; no geometry change |
| LLM-3D Print (2024) | GPT-4o agents read layer images and adjust parameters via Klipper | Never alters G-code geometry; layer shifts excluded as unfixable |
| Task-driven printability assistance (2026) | LLM + knowledge graph recommends design and material changes for a stated task | Before printing only; recommends, doesn't modify |
| Manual resume of a failed print | A maker trimmed the G-code and resumed by hand | Shows physical resume is possible but fiddly; layer bonding failed |

Our contribution: detect and measure the deformation, then adapt the remaining geometry to a function spec and resume.

## Risks and open questions

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

## Sources
- Generalisable 3D printing error detection and correction via multi-head neural networks (CAXTON), Nature Communications 2022
- Matta: Detecting and correcting 3D printing errors on the fly
- LLM-3D Print: Large Language Models To Monitor and Control 3D Printing
- Task-Driven 3D Printability Assistance via Geometry- and Knowledge-Grounded LLM Reasoning
- Fabbaloo: Maker experiments with manually resuming a failed 3D print
- Slice-100K: A Multimodal Dataset for Extrusion-based 3D Printing
- Roboflow: 3D print fail detection dataset
- Hugging Face: failures-3D-print dataset
