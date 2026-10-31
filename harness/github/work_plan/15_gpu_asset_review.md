# WP15 — GPU asset review

Date: 2026-10-23. This records the local pre-release assets for human review. Production GitHub `main` and the Hugging Face Space `OpenSAE/SAEBase` are unchanged. The raw SQLite trace store, model weights, and the other nineteen LIVE-02 prompts stay on the GPU host and are not in this commit.

Reviewer for the one published trace: Haotian Ma. Only that trace was marked redistributable.

## Approved trace

One trace was published into the local bundle `/workspace/wp15/staging/rc1`.

| Field | Value |
| --- | --- |
| Sample | `WP13-DEMO-01` |
| Site | `gemma270m:gemma270m_scope_resid:L5` |
| Prompt | The sensor reading rose from seven to twelve after calibration. |
| Prompt SHA256 | `8034d10b659c951e60f5abc774a8d44b7d777c1f27fa4b44a35c0d9ef4f29ef2` |
| Trace file SHA256 | `3f1c6ce045df8d30168635b11cbe383aeb53aa95f72a6dd67ba41042bc3a6d02` |
| API trace ID | `44e5ff88530599aa316b7ecb3839925ecb2294eb99e68dd84c4c0be5552e7d97` |
| Nonzero feature rows | 639 |

Against the integration Lab pointed at that bundle, the trace list, the trace, and one active feature each returned HTTP 200.

## Reviewed generation examples

Both prompts below were read before publication. The edited text is the ablation continuation from the existing G2 run, not a rewrite. Feature 1819, layer 5, 16 new tokens.

Prompt SHA256 `683d2275d77b908d4e18aa421764dbc0b1a7aa601d1b9702891ab050e1373790`:

> The night clerk counted the brass keys before locking the side door.

Baseline and ablation continuation were the same string, including the leading blank lines:

```
\n\nThe door was open, and the door was locked.\n\nThe door was
```

The first-step maximum logit change for that ablation was 7.65. The displayed continuation does not show it.

Prompt SHA256 `33f0195c5961825882fa08ffddb9b4de80bd572b728b9b87fc4ed772652c75e6`:

> Blue paper boats drifted under the stone bridge after the rain.

Baseline continuation:

```
\n\nThe boat was a blue paper boat, and the bridge was a stone bridge
```

Ablation continuation:

```
<start_of_image>The boat was a blue paper boat, and the boat was a blue paper
```

The ablation changed the first generated token. Its first-step maximum logit change was 9.80.

The checked-in `saebase/research_lab/data/wp15/generation.json` stays without prompt text. The reviewed copy is only in the local staging bundle. The Lab served that copy with HTTP 200.

## Private edit that changed logits

An earlier queue ablation of feature 1627 at the last token left `logit_delta_l2` at 0. That run is not the accepted edit.

The accepted job ablated feature 1819 on the second prompt above, where that feature is active at the last token.

| Field | Value |
| --- | --- |
| Operation | ablate |
| Wrong token on input | HTTP 401 |
| Authenticated result | HTTP 200 |
| Wrong token on result | HTTP 401 |
| `logit_delta_l2` | 1029.88818359375 |
| Maximum absolute logit change | 9.802836418151855 |
| Baseline argmax token | 108 |
| Edited argmax token | 255999 |
| Full logits stored | no |
| Wall time | 12.2 seconds |

The worker no longer writes the full logit vectors into the result file. Those vectors made the result larger than the 10 MB download limit. The result now keeps the scalar change and the two argmax token ids. The comparison function itself still returns the logits to its caller.

## Left unpublished

The G6 hypothesis stays unpublished. Its held-out score is 0.50 on three positive examples and one negative example, and no description was approved. No LIVE-03 public intervention cards were found on this host, so none were added. No browser screenshot was captured. Remote Docker staging and the production Space were not run.
