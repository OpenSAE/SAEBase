from __future__ import annotations

from pathlib import Path

from src.schema import finite, require_keys


def execute(payload, workdir: Path):
    require_keys(payload, {"labels", "scores", "threshold"})
    labels, scores, threshold = payload["labels"], payload["scores"], payload["threshold"]
    if (
        not labels
        or len(labels) != len(scores)
        or any(type(label) is not int or label not in (0, 1) for label in labels)
        or not all(finite(score) for score in scores)
        or not finite(threshold)
    ):
        raise ValueError("labels or scores")
    positives = [scores[i] for i, label in enumerate(labels) if label == 1]
    negatives = [scores[i] for i, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC undefined for a single class")
    auc = sum(
        1 if pos > neg else 0.5 if pos == neg else 0 for pos in positives for neg in negatives
    ) / (len(positives) * len(negatives))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    groups = []
    true_pos = false_pos = average_precision = previous_recall = 0
    for score in sorted(set(scores), reverse=True):
        group = [i for i in order if scores[i] == score]
        true_pos += sum(labels[i] for i in group)
        false_pos += sum(1 - labels[i] for i in group)
        recall = true_pos / len(positives)
        precision = true_pos / (true_pos + false_pos)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        groups.append(
            {
                "score": score,
                "tp": true_pos,
                "fp": false_pos,
                "precision": precision,
                "recall": recall,
            }
        )
    selected = sum(label == 1 and score >= threshold for label, score in zip(labels, scores))
    return {
        "n": len(labels),
        "auc": auc,
        "average_precision": average_precision,
        "recall": selected / len(positives),
        "threshold": threshold,
        "curve": groups,
    }
