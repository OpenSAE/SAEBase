"""Build eight literal-input qualitative reports from recorded features and explicit reviews."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.capture.rss_mode import feature_row

TOPICS = ["Basketball", "Space", "Music", "Chess", "Animals"]
HEADER = "| Status | Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | Recurrence | Neuronpedia explanation |"
RULE = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    inputs = json.loads((ROOT / "scripts/RSS/qualitative_batch_inputs.json").read_text())
    review = json.loads((ROOT / "scripts/RSS/qualitative_batch_review.json").read_text())
    write_json(root / "inputs.json", inputs)
    write_json(root / "review_decisions.json", review)
    cases = []
    for i, (inp, decision) in enumerate(zip(inputs["cases"], review["cases"]), 1):
        directory = root / f"case_{i:02d}"
        captured = json.loads((directory / "rss_case.json").read_text())
        candidates = json.loads((directory / "context_candidates.json").read_text())
        selected = {r["feature_id"]: r for r in captured["rss_features"]}
        target = [selected[f] for f in decision["rss_ids"]]
        contexts = []
        for v, f in enumerate(decision["context_ids"]):
            contexts.append(
                None
                if f is None
                else next(r for r in candidates["groups"][v]["features"] if r["feature_id"] == f)
            )
        match_count = sum(r is not None for r in contexts)
        lines = [
            f"# {inp['case_id']} {decision['title_zh']} · {inp['title']}",
            "",
            "## 完整输入与实验设置",
            "",
            f"Clean：{inp['clean']}",
            "",
        ]
        for variant in inp["variants"]:
            lines += [f"V{variant['number']} · {variant['topic']}：{variant['prompt']}", ""]
        last_ids = [ids[-1] for ids in captured["token_ids"]]
        lines += [
            "模型：Llama 3.1 8B Base；SAE：llama31_8b_lxr_8x；layer 21；hook：blocks.21.hook_resid_post；32,768 features。捕获每条完整输入的最后一个 token（末尾句号），position=-1。",
            "",
            f"实际最后 token IDs（clean、V1–V5）：{last_ids}。RSS 使用原始选择器，M=400、K=20、rho=kappa=0.8、tau=1e-5；完整 raw codes，不作语义预过滤。",
            "",
            "## 合并特征表",
            "",
            "下表保留有解释价值的 RSS 行，并明确标注相邻或宽泛标签。语义不足时不将其当作已确认目标特征；没有清楚背景匹配的行显示缺口，不以无关特征凑足八条。全部20条RSS与所有数值候选见文末原始记录。",
            "",
            HEADER,
            RULE,
        ]
        for label, row in zip(decision["rss_labels"], target):
            lines.append(f"| {label} | " + feature_row(row)[2:])
        for v, row in enumerate(contexts):
            if row is None:
                lines.append(
                    f"| RSS excluded · V{v + 1} {TOPICS[v]}：未找到清楚匹配 | — | — | — | — | — | — | — | — |"
                )
            else:
                lines.append(f"| RSS excluded · V{v + 1} {TOPICS[v]} | " + feature_row(row)[2:])
        lines += [
            "",
            "## 科学解释与判断",
            "",
            decision["interpretation"],
            "",
            decision["context_note"],
            "",
            f"**目标相关性：** {decision['target_finding']}",
            "",
            f"**方向关系：** {decision['opposition']}",
            "",
            f"**背景匹配：** {match_count}/5。该数字仅表示这张定性表中有多少背景得到清楚解释对应，不是性能评测。",
            "",
            f"**展示适用性：** {decision['suitability']}",
            "",
            "**用户提出的待检查解释：** "
            + inp["proposed_interpretation_to_inspect"]
            + " 本批未将该提议当作已知事实。",
            "",
        ]
        if match_count:
            lines += ["### 已展示背景特征的排除依据", ""]
            for v, row in enumerate(contexts):
                if row:
                    lines.append(
                        f"- V{v + 1} F{row['feature_id']}：in_top_M={row['in_top_M']}，recurrence={row['recurrence']:.1f}，consistency={row['consistency']:.1f}；排除条件：{', '.join(row['exclusion_reasons'])}。"
                    )
            lines += [""]
        lines += [
            "## 完整原始记录",
            "",
            "- [全部20条RSS与全部context候选原始解释](all_candidates.md)",
            "- [完整clean/variants/delta数组](rss_activations.npz)",
            "- [完整字典RSS统计与入选资格](rss_feature_statistics.npz)",
            "- [原始Capture结果及所有20条RSS](rss_case.json)",
            "- [全部数值特有候选及排除原因](context_candidates.json)",
            "- [实际配置](config.yaml) · [正式任务返回](task_result.json)",
            "- [本例展示行与人工阅读判断](qualitative_case.json)",
        ]
        (directory / "case_report.md").write_text("\n".join(lines) + "\n")
        all_lines = [
            f"# {inp['case_id']} 全部实测候选",
            "",
            "## 原始RSS选中的全部特征",
            "",
            HEADER,
            RULE,
        ]
        for row in captured["rss_features"]:
            all_lines.append(
                f"| RSS {'↑' if row['direction'] > 0 else '↓'} | " + feature_row(row)[2:]
            )
        all_lines += [
            "",
            "## 全部数值特有候选",
            "",
            candidates["candidate_rule"],
            "",
            "候选由激活确定，公开解释仅供阅读；候选不等于通过语义验收。原始strict-recurrence候选也保留在rss_case.json。",
            "",
        ]
        for v, group in enumerate(candidates["groups"]):
            all_lines += [
                f"### V{v + 1} {TOPICS[v]}",
                "",
                HEADER + " Exclusion reason |",
                RULE + " --- |",
            ]
            for row in group["features"]:
                reason = ", ".join(row["exclusion_reasons"])
                all_lines.append(f"| RSS excluded | " + feature_row(row)[2:] + f" {reason} |")
            if not group["features"]:
                all_lines += ["", "没有符合数值候选条件的特征。"]
            all_lines += [""]
        (directory / "all_candidates.md").write_text("\n".join(all_lines) + "\n")
        payload = {
            "case": inp,
            "review": decision,
            "context_match_count": match_count,
            "display_rss_rows": target,
            "display_context_rows": contexts,
            "capture_position": captured["capture_position"],
            "last_token_ids": last_ids,
            "identity": captured["identity"],
            "rss_parameters": captured["rss_parameters"],
            "neuronpedia": candidates["neuronpedia"],
        }
        write_json(directory / "qualitative_case.json", payload)
        cases.append(
            {
                "case_id": inp["case_id"],
                "title": decision["title_zh"],
                "target": decision["target_finding"],
                "opposition": decision["opposition"],
                "context_match_count": match_count,
                "suitability": decision["suitability"],
                "priority": decision["priority"],
                "report": f"case_{i:02d}/case_report.md",
                "rss_count": len(captured["rss_features"]),
                "capture_count": len(captured["token_ids"]),
            }
        )
    summary = [
        "# 八个RSS定性案例对照",
        "",
        "指定的八组literal inputs均已执行：Llama 3.1 8B Base / Llama Scope第21层32k SAE，在clean与五个variants的末尾句号捕获，共48条输入。每例保存完整codes和delta，以及原RSS实际选出的20条特征。全部8例完成后才填写阅读判断与排序。",
        "",
        "**本批没有同时满足三项完整展示标准的案例。** 有些数学、AI责任或安全相邻特征值得人工查看，但尚未得到两到三条机制明确、具有连贯双向关系的目标特征，也未得到任何一个案例中五类背景全部匹配的组合。",
        "",
        "## 八例比较",
        "",
        "| 案例 | 目标相关RSS特征 | 相反方向是否有连贯解释 | 清楚背景匹配 | OpenReview适用性 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in cases:
        summary.append(
            f"| [{c['case_id']} {c['title']}]({c['report']}) | {c['target']} | {c['opposition']} | {c['context_match_count']}/5 | {c['suitability']} |"
        )
    summary += [
        "",
        "## 建议人工先看",
        "",
        "1. **C06 证据与引用**：AI负责任使用特征下降，文档格式特征上升，存在内容/形式的阅读线索；但公开标签尚不能把二者具体解释为证据可靠性与引用表面形式。背景只匹配太空一项。",
        "2. **C01 十进制比较**：两条数学概念特征在五个背景中共同下降；这是较清楚的数学主题关联，但没有十进制大小、数字表面形式的区分，也没有目标主题内的双向组合。",
        "3. **C02 凭据安全**：背景的得分/比赛特征与太空旅行特征各只对相应variant响应，排除依据清楚；目标侧只有相邻的黑客/恶意软件标签，不能替代API凭据暴露机制。",
        "",
        "这些是本批相对值得复核的线索，不是已推荐进入rebuttal的完整证据。本次未起草最终rebuttal。",
        "",
        "## 如何阅读这些判断",
        "",
        "用户给出的八种潜在对照均作为待检查假设保留。特征解释是公开标签，不能仅从标签相近或数值一增一减推断已经识别具体机制。C03、C07、C08的合并表明确将宽泛参考行标为未确认目标，未以其凑成合格案例。",
        "",
        "所有背景数值候选在查询解释之前由完整delta确定：未被RSS选中、本变体正变化>tau、至少为其他四个最大绝对变化的五倍，并按本变体幅度排序，保留所有符合条件的候选。本批将recurrence失败作为优先展示条件，同时记录其他可能排除原因。三个实际背景匹配（C02两条、C06一条）均在M候选池内，recurrence=0.2，其他四个delta均零，明确因recurrence不足排除。",
        "",
        "本批没有微调指定prompts、模型、层或RSS参数来追逐预设解释；判断范围就是这八组末尾句号的实测激活与公开解释。",
        "",
        "## 产物入口",
        "",
        "- [全部原始输入及待检查解释](inputs.json)",
        "- [逐例AI阅读判断](review_decisions.json) · [机器可读比较摘要](comparison.json)",
        "- [本批特征原始解释与来源](batch_annotations.json)",
        "- [运行参数与完成状态](run.json)",
        "- 每例的case_report.md含完整输入、一张合并表、简短科学解释和全部原始记录链接。",
    ]
    (root / "summary.md").write_text("\n".join(summary) + "\n")
    write_json(
        root / "comparison.json",
        {
            "status": "eight_cases_measured_and_reviewed",
            "case_count": len(cases),
            "input_count": sum(c["capture_count"] for c in cases),
            "full_scientific_example_found": False,
            "cases": cases,
            "manual_review_order": ["C06", "C01", "C02"],
            "final_rebuttal_drafted": False,
        },
    )
    (root / "index.md").write_text(
        "# 八个定性案例\n\n[先看：八例比较摘要与人工复核建议](summary.md)\n\n"
        + "\n".join(f"- [{c['case_id']} {c['title']}]({c['report']})" for c in cases)
        + "\n"
    )
    print(root / "summary.md")


if __name__ == "__main__":
    main()
