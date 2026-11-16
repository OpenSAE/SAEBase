"""Export fifty questions with answers and only the four requested Top1 bullets."""

import json
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent / "results"
    report = json.loads((directory / "report.json").read_text())
    assert report["completed_questions"] == 50
    lines = ["# 50道题：Whiten、Non-whiten、g(S)与交互的Top 1", ""]
    for number, question in enumerate(report["questions"], 1):
        white = question["whiten_top_feature"]
        raw = question["absolute_activation_top_feature"]
        single = question["max_g01_single"]
        pair = question["max_interaction_pair"]
        prompt = question["prompt"].rstrip()
        if prompt.endswith("Answer:"):
            prompt = prompt[: -len("Answer:")].rstrip()
        lines += [
            f"## {number}. {question['key']}",
            "",
            f"**题目：** {question['question_zh']}",
            "",
            "**完整原题（含选项）：**",
            "",
            *["> " + line for line in prompt.splitlines()],
            "",
            f"**标准答案：** {question['answer_zh']}",
        ]
        if question["answer_zh"] != question["answer"]:
            lines += ["", f"标准答案原文：{question['answer']}"]
        descriptions = "；".join(
            f"{feature['feature_id']}：{feature['explanation_zh']}" for feature in pair["features"]
        )
        lines += [
            "",
            f"- **Whiten Top 1：{white['feature_id']}**。{white['explanation_zh']}。"
            f"**zscore = {white['zscore']:.6f}**；activation = {white['activation']:.6f}。",
            f"- **Non-whiten Top 1：{raw['feature_id']}**。{raw['explanation_zh']}。"
            f"**activation = {raw['activation']:.6f}**；zscore = {raw['zscore']:.6f}。",
            f"- **g(S) Top 1（single）：{single['feature']['feature_id']}**。"
            f"{single['feature']['explanation_zh']}。仅清零这一项，其余保持原样："
            f"**g(S) = {single['g']['01']:.6f}**；相对原始状态Δg = {single['delta_g']:+.6f}。",
            f"- **交互 Top 1（pair）：{pair['pair_ids'][0]} × {pair['pair_ids'][1]}**。"
            f"{descriptions}。**I = {pair['interaction']:.6f}**。",
            "",
        ]
    target = directory / "50_questions_four_top1_zh.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    assert sum(line.startswith("## ") for line in lines) == 50
    assert sum(line.startswith("- **") for line in lines) == 200
    print(target)


if __name__ == "__main__":
    main()
