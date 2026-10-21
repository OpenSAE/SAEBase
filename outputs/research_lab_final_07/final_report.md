# Research Lab final product

Feature Effects now opens on one measured sentence and one action, Remove this feature. The other labs use the same dark rail, light page, and card style as Feature Atlas. Atlas content is unchanged.

## What a visitor sees

Landing example, Gemma 1B: “The capital of France is”. Before removal the top next token is “ a” (23.0%). After removal it is “ Paris” (35.9%). The sentence says the top piece changed and that this does not mean a better answer or a known feature meaning.

Another example reaches four more measured prompts: a color question, an arithmetic question, an animal name, and a Pythia 410M story opening where the change is under one percentage point. Other measured edits (cut the activity in half, add 1) are behind Explore other changes. Protocol, feature id, fingerprints, and the six score-only records are behind Experiment details.

## What was reused and what was newly measured

Reused: the eight authored cases from `effects-authored-20261021`, the six GPU-04 score-only pairs, the 64-state context grid, the RSS aggregates, and Feature Atlas (`features.json` `f3bc09f9e9ee6528`).

Newly measured, same two local checkpoints, four prompt-site pairs, run `effects-authored-20261021b`: “The color of the sky is” and “Once upon a” on Pythia 410M layer 12 and Gemma 1B layer 13. The public gallery uses the Gemma color case and the Pythia story case. The other two new rows stay in the data and in Experiment details. No third-party prompt text was published. GPU-04 prompts remain unpublished because the wording was not stored.

## Checks

`apps/research-lab/tests/test_export_real_cases.py`: 3 passed. Local and live Chromium opened all five modules at desktop and mobile widths with no console errors and no failed assets. Horizontal layout overflow was 0 at 1440, 768, and 390. Restore original and three curated examples were exercised on the live host.

Screenshots: `outputs/research_lab_final_07/before/` and `after/`, five modules by three viewports.

## Commits and URLs

- Repository commit: `c1ea15c27585f9cfa19fce24ee62ad3940ca6269` on `work-plan-07-final-redesign`
- Space commit: `0ede94d213a70a7da2e5aede03dda79604c58555`
- Feature Atlas: https://opensae-saebase.static.hf.space/index.html?v=wp07#features
- Feature Effects: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07#effects
- Context Lab: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07#context
- Compare models: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07#models
- Research: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07#interactions

## Acceptance gates

1. PASS. First screen shows “The capital of France is” and Remove this feature.
2. PASS. Original and removed columns use the same candidates and full-vocabulary chances.
3. PASS. The takeaway states that the top piece changed from “ a” to “ Paris”.
4. PASS. Feature ids, protocol strings, and score-only rows are inside the closed details.
5. PASS. The page states what the input was, what was removed, what the top piece did, and that this is not a better answer.
6. PASS. The two actions are separate buttons with a visible gap at 390, 768, and 1440. Layout overflow is 0.
7. PASS. Another example visits the color, arithmetic, animal, and story prompts.
8. PASS. Cut its activity in half is a measured secondary edit and changes the shown result.
9. PASS. Each public example keeps its id, site, feature, prompt, and source hash.
10. PASS. Displayed chances match exp(logit − logsumexp) within 1e-5.
11. PASS. No generated sentence was invented. L2 is not shown as a probability.
12. PASS. Atlas hypotheses stay labeled under review and are not used as feature names.
13. PASS. The page calls the rows selected examples.
14. PASS. Before and after screenshots exist for five modules at three viewports. Atlas still shows Feature #20227 and ten prompts.
15. PASS. All five modules use the same rail, breadcrumb, width, type, and card border.
16. PASS. No horizontal overflow at the three widths. The mobile nav scrolls instead of clipping the active label.
17. PASS. Atlas opens Browse measured effects when no exact site, feature, and prompt match exists. It does not swap in a different example.
18. PASS. Context still has 64 cells and the six factor labels. Model comparison and Research load on the live host.
19. PASS. Remove, Restore, and Another example are at least 44px tall and have a visible focus outline.
20. PASS. Live desktop and mobile loads had no console errors and no failed asset requests.
21. PASS. Space revision `0ede94d213a70a7da2e5aede03dda79604c58555`. Live `features.json` still begins `f3bc09f9e9ee6528`.
22. PASS. Live Effects, Context, and Atlas were used, including three examples and Restore. All five routes loaded.
23. PASS. The new Effects page no longer concatenates the edit labels. Atlas screenshots still match the approved rail and prompt gallery.
24. PASS. Artifacts are under `outputs/research_lab_final_07/`.
