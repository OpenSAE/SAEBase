# Work Plan 07 inventory

Recorded before the Effects redesign. No credentials.

## Source of truth

- Repository branch at the start of this pass: `work-plan-06-real-evidence`, then `work-plan-07-final-redesign`.
- Live frontend is the Hugging Face Space checkout at `/workspace/opensae-saebase-space`, not `apps/research-lab/frontend`.
- Feature Atlas: `index.html`, `assets/style.css`, `assets/feature-atlas.js`, `assets/features.json`.
- Feature Effects, Context Lab, Compare models, and Research: `lab-experiments/`. The Atlas rail links to that directory. The older L2 views inside `index.html` are not the live route; `assets/app.js` sends those hashes to `lab-experiments`.
- Deployment is a static upload of the Space checkout. `features.json` checksum `f3bc09f9e9ee6528d3d52ceb511f383d038be288480f02a13175b6e556ceb41c`.

## Evidence used

- Eight authored next-token cases from run `effects-authored-20261021` (Pythia 410M layer 12 and Gemma 1B layer 13).
- Four more authored cases from run `effects-authored-20261021b`: “The color of the sky is” and “Once upon a” on the same two sites. Full-vocabulary softmax. Identity patch passed.
- Six GPU-04 score-only pairs. Prompt text was not stored. They stay in Experiment details.
- Context 64-state grid and position-corrected RSS aggregates are unchanged.

## Baseline

Screenshots before the redesign are in `outputs/research_lab_final_07/before/` for Atlas, Effects, Context, Compare models, and Research at 1440×900, 768×1024, and 390×844.
