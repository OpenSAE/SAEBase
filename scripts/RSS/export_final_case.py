"""Export the measured same-case target/context features selected for display."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.capture.rss_mode import feature_row

TARGET_IDS = [12293, 28763, 9190]
CONTEXT_IDS = [17410, 5436, 32506, 3044, 30221]
TOPICS = ["Basketball", "Space", "Music", "Chess", "Animals"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    case_path = directory / "case_01/rss_case.json"
    case = json.loads(case_path.read_text())
    target = [next(r for r in case["rss_features"] if r["feature_id"] == f) for f in TARGET_IDS]
    context = [
        next(r for r in case["context_candidates"][v]["features"] if r["feature_id"] == f)
        for v, f in enumerate(CONTEXT_IDS)
    ]
    lines = [
        "# Llama 3.1：同一个案例中的目标重分配与新增背景特征",
        "",
        "本例在同一 clean、同一组五个 variants、同一模型和 SAE 上，同时取得三条目标相关 RSS 特征与五条对应新增背景、未被 RSS 选中的特征。",
        "",
        "模型：Llama 3.1 8B Base。SAE：`llama31_8b_lxr_8x`，layer 21，`blocks.21.hook_resid_post`，32,768 features。所有输入在最后一个相同 token（句号）捕获完整原始 SAE codes。",
        "",
        "RSS 参数：M=400、K=20、rho=kappa=0.8、tau=1e-5。变化值为 variant − clean。解释在激活选集确定后读取，不参与 RSS 选择。",
        "",
        "## 完整输入",
        "",
        f"Clean：{case['clean_prompt']}",
        "",
    ]
    for v, prompt in enumerate(case["variants"], 1):
        lines += [f"V{v} · {TOPICS[v - 1]}：{prompt}", ""]
    lines += [
        "## 表 A：RSS 选中的三个烹饪相关特征",
        "",
        "从原始 RSS 的20个实际选中特征中，按解释节选以下三条。F12293、F28763 减少，F9190 增加。",
        "",
        "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | 原始 Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [feature_row(row) for row in target]
    lines += [
        "",
        "## 表 B：五个背景各自对应、被 RSS 排除的特征",
        "",
        "每个特征都只在对应 variant 中增加，其他四个 delta 均为零；五条特征的 recurrence 均为 1/5=0.2，低于 rho=0.8。它们均进入 M=400 的候选池，因为不满足 recurrence 条件而不具备 RSS 入选资格。",
        "",
        "| Variant | Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | 原始 Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for topic, row in zip(TOPICS, context):
        lines.append(f"| {topic} | " + feature_row(row)[2:])
    lines += [
        "",
        "篮球场景对应球员与团队关系，太空场景对应太空旅行，小提琴场景对应乐器，棋赛对应国际象棋策略，非洲象场景对应野生动物。解释原文和原始页面链接均保留在表中。",
        "",
        "## 设计与选择记录",
        "",
        "用户允许修改 prompts 后，在 Llama 3.1 的第16、21层探索了96种背景与固定烹饪请求组合。随后固定本例的完整五个 backgrounds，重新执行 clean + 五个 variants；本报告所有行均来自此次同组实际结果，没有拼接先前案例的表格行。",
        "",
        "固定目标由最初的无背景烘焙问题改为‘根据 setting 改写香蕉面包配方’，因此这是背景相关的配方改写案例；不能将其描述为最初的无关事实实验。论文展示时应使用上述完整真实输入。该案例经探索筛选，尚未作为跨 prompts 的普遍性结果。",
        "",
        "## 结果读取",
        "",
        "- `case_01/rss_case.json`：全部20条 RSS 特征、全部数值 context 候选、原始解释、recurrence、候选池资格与排除原因。",
        "- `case_01/rss_activations.npz`：完整 clean、五个 variants 与 delta。",
        "- `case_01/config.yaml`、`case_01/task_result.json`、`run.json`：实际运行设置和正式任务返回。",
        "- `selected_case.json`：本次展示选用的8条特征及完整输入。",
    ]
    (directory / "case_report.md").write_text("\n".join(lines) + "\n")
    (directory / "selected_case.json").write_text(
        json.dumps(
            {
                "status": "matched",
                "selection_kind": "exploratory_case_study",
                "measurement": str(case_path),
                "identity": case["identity"],
                "clean": case["clean_prompt"],
                "variants": case["variants"],
                "rss_parameters": case["rss_parameters"],
                "table_a": target,
                "table_b": [
                    {"variant": v + 1, "topic": TOPICS[v], **row} for v, row in enumerate(context)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(directory / "case_report.md")
    for row in target:
        print("TARGET", row["feature_id"], row["delta"], row["recurrence"], row["consistency"])
    for row in context:
        print(
            "CONTEXT",
            row["feature_id"],
            row["delta"],
            row["recurrence"],
            "in_top_M",
            row["in_top_M"],
        )


if __name__ == "__main__":
    main()
