# LIVE-03 small summary

Status: teacher-forced logit measurements only. No text was generated.

The same saved prompt was edited at token position 1. L5 uses measured content feature 506. L9 uses measured content feature 95 at that same position, not the shared beginning-of-sequence feature. Scale replaces the live code with twice its measured value.

No-op deltas are 0. Ablation changes the next-token argmax at the edited position and leaves the final-token argmax unchanged. Full logit tensors stay on the GPU host and are not in this commit.
