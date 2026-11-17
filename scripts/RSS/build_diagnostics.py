"""Export all selected feature rows and bounded qualitative interpretations for E1–E4."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def descriptions(row):
    return "; ".join(e["description"] for e in row["annotation"]["explanations"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    review = json.loads((ROOT / "scripts/RSS/diagnostic_review.json").read_text())
    (root / "review_decisions.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n"
    )
    summaries = []
    for decision in review["cases"]:
        eid = decision["id"]
        directory = root / eid
        case = json.loads((directory / "rss_case.json").read_text())
        inp = case["diagnostic_case"]
        lines = [
            f"# {eid} {decision['title_zh']}",
            "",
            f"研究问题：{' '.join(inp['scientific_question'].split())}",
            "",
            "## 完整输入",
            "",
            "以下代码块保留实际发送给模型的内部换行。",
            "",
            "Clean",
            "",
            "```text",
            inp["clean"],
            "```",
            "",
        ]
        for v, prompt in enumerate(inp["variants"], 1):
            lines += [f"V{v}", "", "```text", prompt, "```", ""]
        lines += [
            "## 实验设置与完整RSS表",
            "",
            "Llama 3.1 8B Base；SAE llama31_8b_lxr_8x；layer 21；blocks.21.hook_resid_post；32,768 features。捕获完整提示词最后一个token（目标问题末尾），使用完整原始SAE codes。",
            "",
            f"实际末尾token（clean、V1–V5）：{case['captured_last_tokens']}。位置均为-1。",
            "",
            "原RSS参数：M=400、K=20、rho=kappa=0.8、tau=1e-5。Delta为variant减clean。解释在RSS数值选择之后取得，未参与选集。下表完整保留全部选中特征，按原RSS顺序排列。",
            "",
            "| Feature ID | Clean activation | Delta V1 | Delta V2 | Delta V3 | Delta V4 | Delta V5 | Recurrence | RSS direction | Original Neuronpedia explanation | Neuronpedia URL |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in case["rss_features"]:
            fid = row["feature_id"]
            url = row["annotation"]["url"]
            vals = " | ".join(f"{x:+.6g}" for x in row["delta"])
            lines.append(
                f"| [F{fid}]({url}) | {row['clean_activation']:.6g} | {vals} | {row['recurrence']:.1f} | {'+1 ↑' if row['direction'] > 0 else '-1 ↓'} | {cell(descriptions(row))} | [Neuronpedia]({url}) |"
            )
        rows = {r["feature_id"]: r for r in case["rss_features"]}
        focus = [rows[f] for f in decision["focus_ids"]]
        lines += [
            "",
            "Recurrence统计非零变化比例；RSS direction是超过tau的主导符号，不表示每个变体都同向。原算法允许五个中四个同向，具体例外以表中delta为准。",
            "",
            "## 最清楚的候选语义关系（最多三条）",
            "",
            "本段选用："
            + ", ".join(f"F{f}" for f in decision["focus_ids"])
            + "。这是对完整选集的事后阅读，不改变RSS。",
            "",
        ]
        for i, paragraph in enumerate(decision["interpretation"], 1):
            lines += [f"{i}. {paragraph}", ""]
        lines += [
            f"**具体有用规则/动作是否出现：** {decision['useful_specific_rule']}",
            "",
            f"**作者审阅建议：** {decision['verdict']}",
            "",
            "## 输入范围与原始记录",
            "",
            "本轮遵照literal输入保留附件内部换行。各变体的目标问题词序和标点保持一致，但排版换行位置不完全一致；因此观测同时包含这些字面换行差异，不能将激活变化完全单独归因于新增误导信息或曝光渠道。没有进行额外规范化、答案生成或定量评测。",
            "",
            "- 完整raw clean/variants/delta：[rss_activations.npz](rss_activations.npz)",
            "- 全部20条RSS、原解释、方向、token IDs与完整来源：[rss_case.json](rss_case.json)",
            "- 本例候选解释与3条展示关注行：[interpretation.json](interpretation.json)",
            "- 模型/SAE身份、精度和软件版本：[config.json](../config.json)",
            "- 全部实际输入：[inputs.json](../inputs.json)",
        ]
        (directory / "case_report.md").write_text("\n".join(lines) + "\n")
        payload = {
            "review": decision,
            "focus_rows": focus,
            "input_case": inp,
            "interpretation_status": "post_selection_qualitative_reading",
            "rss_selection_count": len(case["rss_features"]),
        }
        (directory / "interpretation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        summaries.append(
            {
                "id": eid,
                "title": decision["title_zh"],
                "finding": decision["finding"],
                "specific_rule": decision["useful_specific_rule"],
                "verdict": decision["verdict"],
                "priority": decision["priority"],
                "report": f"{eid}/case_report.md",
            }
        )
    lines = [
        "# E1–E4诊断案例比较",
        "",
        "四组均已按附件literal输入执行，共24次末尾token捕获，每组完整保存原RSS选中的20条特征。模型、SAE和原RSS参数固定，未生成答案、未做定量benchmark。全部四组完成后进行下列定性比较。",
        "",
        "**E1是相对最值得作者审阅的部分线索；四组都没有明确支持预设的具体正确规则、机制或行动转移。**",
        "",
        "| 案例 | 增减特征的实际语义关系 | 增加方是否包含更具体有用规则 | 作者审阅判断 |",
        "| --- | --- | --- | --- |",
    ]
    for c in summaries:
        lines.append(
            f"| [{c['id']} {c['title']}]({c['report']}) | {c['finding']} | {c['specific_rule']} | {c['verdict']} |"
        )
    lines += [
        "",
        "## 作者先看什么",
        "",
        "先看E1的F21772（数学函数和概念，主导减少）、F9736（数学证明，增加）与F29126（推理或因果解释，增加）。其标签支持粗粒度的数学主题/论证方式差异，但减少方没有明确表示错误启发式，增加方也没有识别十进制位值或正确比较规则。不能把这组变化写成已证实的“从误导模式转向正确推理”。",
        "",
        "E2的增加方是一般技术支持问答，不是撤销或轮换凭据；E3的增加方是泛数学概念，不是共同原因或混杂推理；E4的增加方是配置对象与类，不是有序不变量。这三例保留为未成功解释的案例，不因增加方向而赋予它们更高任务价值。",
        "",
        "每份报告都包含全部20条而非节选表，之后才聚焦最多三条候选关系，逐项讨论减少方、增加方、任务意义与标签是否足以支持解释。最终是否采用由作者查看原始页面后决定。本次没有起草rebuttal。",
        "",
        "## 字面输入说明",
        "",
        "附件正文有内部换行，四组均原样保留；目标问题词序与标点不变，但clean和variant中的换行位置不完全相同。报告明确记载这一范围，不把包含换行差异的结果当作纯前缀变换的单独因果效应。E1/E2均在问号捕获，E3/E4均在句号捕获。",
        "",
        "## 下载与记录",
        "",
        "- [仅含四份完整案例报告的ZIP](../rss_diagnostic_four_reports.zip)",
        "- [原始输入](inputs.json) · [模型/SAE配置与版本](config.json) · [运行完成状态](run.json)",
        "- [AI定性阅读记录](review_decisions.json) · [机器可读比较](comparison.json)",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    (root / "comparison.json").write_text(
        json.dumps(
            {
                "status": "four_cases_completed_and_reviewed",
                "cases": summaries,
                "preferred_partial_case": "E1",
                "specific_mechanism_transition_established": False,
                "final_rebuttal_drafted": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    archive = root.parent / "rss_diagnostic_four_reports.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for c in summaries:
            z.write(root / c["id"] / "case_report.md", f"{c['id']}_report.md")
    print(root / "comparison.md")
    print(archive)


if __name__ == "__main__":
    main()
