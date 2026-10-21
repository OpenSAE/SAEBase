# Research Lab final product

Feature Effects names the measured feature first, then uses one On/Off switch. Off means that feature's activity was set to zero for the prediction. The input and the model weights stay the same. The other labs use the same dark rail, light page, and card style as Feature Atlas. Atlas content is unchanged.

## What a visitor sees

The page opens on Gemma 1B and “The capital of France is”, with Feature #203 still on. The chart shows only the normal next-token list, topped by “ a” (23.0%). Switching Off shows the measured pair: the top next token becomes “ Paris” (35.9%). Switching On returns to the normal list. Another example changes the prompt and the feature together and puts the switch back to On.

Feature #203 was the strongest nonzero coordinate on that input. The page does not call it a fact feature. The numeric activation was not stored, so there is no activity meter. Protocol, fingerprints, and the six score-only records stay behind Experiment details.

## What was reused and what was newly measured

Reused: the eight authored cases from `effects-authored-20261021`, the six GPU-04 score-only pairs, the 64-state context grid, the RSS aggregates, and Feature Atlas (`features.json` `f3bc09f9e9ee6528`).

Newly measured, same two local checkpoints, four prompt-site pairs, run `effects-authored-20261021b`: “The color of the sky is” and “Once upon a” on Pythia 410M layer 12 and Gemma 1B layer 13. The public gallery uses the Gemma color case and the Pythia story case. The other two new rows stay in the data and in Experiment details. No third-party prompt text was published. GPU-04 prompts remain unpublished because the wording was not stored.

## Checks

`apps/research-lab/tests/test_export_real_cases.py`: 3 passed. Local and live Chromium opened all five modules at desktop and mobile widths with no console errors and no failed assets. Horizontal layout overflow was 0 at 1440, 768, and 390. The live page was switched Off and back On, then moved to the next example.

Screenshots: `outputs/research_lab_final_07/before/` and `after/`, five modules by three viewports.

## Commits and URLs

- Repository commit: `04b9d2dd6a5d318a9996e2e2e88512627693b423` on `work-plan-07-final-redesign`
- Space commit: `3b1a6524210fac9927cb8143edbb585f869aed73`
- Feature Atlas: https://opensae-saebase.static.hf.space/index.html?v=wp07b#features
- Feature Effects: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07b#effects
- Context Lab: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07b#context
- Compare models: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07b#models
- Research: https://opensae-saebase.static.hf.space/lab-experiments/index.html?v=wp07b#interactions

## Acceptance gates

1. PASS. First screen shows the prompt, Gemma 1B, and Feature #203 before the switch.
T01. PASS. Feature #203 and Gemma 1B are visible, with the reason this coordinate was selected.
T02. PASS. The switch sits next to the sentence that activity is set to zero and the weights stay the same.
T03. PASS. The old remove and restore buttons are gone. One On/Off switch toggles both ways.
T04. PASS. Another example changes the prompt and returns the switch to On.
T05. PASS. The arithmetic prompt is not labeled an arithmetic feature.
T06. PASS. No activity meter was drawn, because the numeric activation was not stored.
T07. PASS. The first view has no “A REAL INPUT”, no separate restore button, and no hash.
T08. PASS. Off shows aligned candidates. The France sentence names the change from “ a” to “ Paris”.
T09. PASS. The shared rail and card style are unchanged from the previous shell pass.
T10. PASS. Live check: identify Feature #203, switch Off, read the Paris comparison, switch On. Widths 1440, 768, and 390 had no horizontal overflow.
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
19. PASS. The On and Off buttons are 44px tall and have a visible focus outline.
20. PASS. Live desktop and mobile loads had no console errors and no failed asset requests.
21. PASS. Space revision `0ede94d213a70a7da2e5aede03dda79604c58555`. Live `features.json` still begins `f3bc09f9e9ee6528`.
22. PASS. Live Effects identified Feature #203, switched Off and On, and opened the next example. All five routes loaded in the previous shell check.
23. PASS. The new Effects page no longer concatenates the edit labels. Atlas screenshots still match the approved rail and prompt gallery.
24. PASS. Artifacts are under `outputs/research_lab_final_07/`.
