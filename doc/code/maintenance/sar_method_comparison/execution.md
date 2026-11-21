# Whiten's I versus SAR.

## Purpose

Harmonize the selection of candidates for I and SAR, as requested by the users, rerun on 100 questions and deliver 10 models to answer the correct title and characterization.

## Scope

MMLLU-Pro, Llama 3.1 8B Base, Llama Spope L21 32K; results do not cover historical operations.

## Programme and conditions for completion

Using the standard 100 questions already selected in 500, soeed42; the same 100 questions for standardization and frequency filtering. The original model answers 31 questions by dataset labels, and the topics known to contradict the labels are shown out. The first two modes of semantic and density filtering, and then cross-issue frequency filtering, extracting 10 in absolute z-score; filtering without g(S) filtering, and not relaxing. I down 1024 states, SAR sample 384 groups, 256 is proposed to be set aside for a period of 128. After a formal execute_run precheck, configuration, follow-up, original response, Chinese interpretation, and question-by-issue answer.

## Implementation record

- The SAR phase was changed to a full rating of 1024 in current cycle I, which was still taken from 384 groups according to original thesis, and 256 was to be set aside for a period of time. The fit_responses, which were extracted to tasks/capture/response_surface.py in the same function, were used to measure SAR and rating; there was no copying of the paper estimate. The two real questions were re-referenced to zero directly to precheck indicators and recorded in the original paper, and 44 items were passed and one jump was made; multiple forms of stand-alone/matching factors were known to be passed.

- User confirms that the existing density rule is followed: 50% of the non-zero-activation ratio is excluded and universal activation is avoided. precheck and the I.SAR on the same 100th issue are restored; no further research parameters are available.

- 2026-11-19: Confirm that I has a Whitten candidate, changing the open default to a visible whitten; the old formula g_s retains the original syntax.
- Launcher launch.py calls the full running life cycle; each of the two models is automatically pre-screened and then fully operational, and the status.json records the real-time state.

- The precheck began and the check found that the previous dialogue had reversed the density filtering direction: the actual blocked _features were excluded at the endance > = 0.5; the user quoted text was excluded at <0.5. The user was advised and suspended from the self-owned process 580282(SIGTERM) pending the study semantic selection; I defaulted that Whiten and other screening parameters had been configured and 47 checks passed. Follow-up should resume the operation after confirmation and keep the precheck interrupted.

## Completed and used

- I have completed the 100/100 question, and SAR has reformulated the 100/100 question, all of which have failed to do so, 0 are undefined, and 0 are blocked. The public entry point line is read in two complete sets.Verification.json checks all 100 identical candidates, SAR 384 and I scores, all required documents and Hashi.g(S) both results are identical to the original characteristics, standardized identity and original answers of 100 questions of this round, and are validated for reuse.
- The IR experiment/MMLUPro/llama31_8b/capture/whiten100_i_42; the SAR-derived directory peer whiten100_sar_42, the execution method and source I reference is clearly stored in the list. The original long-term mission is terminated by the own end-of-the-life process after the I list is completed, avoiding the continuation of the duplicate SAR; no other tasks are affected.
- This 100th issue model option is the same as the data set label 31%. Displays 10 responses to the 10 questions, excluding 995 questions where the labels are known to contradict the electrons formula. MMUPro_right 10 questions_five configuration Top5.md includes 10 questions, 50 tables, 250 row rankings, all single features and pair interpretations have been translated into Chinese.
- 91 software check passed, 1 jump; configuration document sync checked and Ruff passed. Delaft_sar_pairs.json has been deleted for pre-checking translations only; original arrays, pre-checks and source records are maintained.
- Research opinion: For I to be the most measured interactive across all backgrounds, the AR Table is the second-tier regression factor for a sample of 384 groups, which cannot be directly considered the same indicator. The current candidate is screened by Whiten, allowing a zero-activated candidate, and does not automatically ease the filter.
