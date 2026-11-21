# Capture Shared Active Feature Filter

## Background and objectives

0.1% Density shows that there are five possible ways to activate the Zero feature into the candidate pool.

## Implementation and acceptance

- The formal formulation uses `task.capture.selection.candidate_policy=active_whiten` by default. The density parameters, entry thresholds and comparison of existing tasks are still read by the Capture configuration. The historical preservation configuration takes the old path when there is no candidate policy, and the historical retrospect script uses the old strategy.
- Saves undefined reasons, candidates, and model answers on a case-by-case basis. Saves results to read, interpret, refresh, and model answers to the same status as the report.
- Checks the default candidate ceiling 10 in the official CLI read-only configuration, as well as the threshold values of the current round covering 0.001 and rigorous comparison. The configuration, synthesis candidate, SAR and model answer test is passed.
- Use it. [0.1% Batch results obtained](../../../../experiment/batches/capture/density_comparison/comparison.md) The five models are the same on a case-by-case basis and the number is the same. 0,3,0,5,0,1,6,4,0,5.Three candidates for the Straight Threshold 9074,4148,5625.There are no candidates for the four questions, and five questions are not enough to calculate a pair of interactions..

## Result

The default comparison distinguishes between primary activation, albinos and three behaviors in the same range of candidates on the same subject. The deficit at the threshold of 0.1% should be read in conjunction with the numerical results of each model.

## Default threshold and candidate ceiling update

- The user confirms that the non-zero candidate for the original activation of the subject continues. Default strictly excludes the feature with a density greater than 1 per cent. The upper limit is changed to 20, less than the upper limit. The zero option retains the original record and model answers, text and HTML displays omitted.
- I ' s independent calculation ceiling is checked by the actual number of candidates on this topic. Checking with the same set of 10 issues has been activated, with 4 to 12 eligible candidates, all within the default budget of 16.
- The configuration and associated regression tests are passed, with 122 passes and 1 jump. Start the tmux session `saebase-capture-density1pct`, the first fully run check threshold and candidate ceiling. [start and path log](../../../../experiment/batches/capture/density_comparison/active_density1pct/launch.json)]
