"""Export transparent explanation navigation and a same-case report from measured results."""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.capture.rss_mode import feature_row

# These patterns only navigate post-selection explanations, never select RSS features.
PATTERNS = [
    r"basket|sports?|athlet|ball",
    r"astronaut|space|orbit|cosmi|nasa|rocket|planet|galax|aeronaut|astronom",
    r"violin|music|symphon|orchestra|instrument|concert",
    r"chess|knight|board.game",
    r"elephant|animal|africa|savanna|wildlife|mammal",
]


def navigation(root):
    cases = []
    for path in sorted(root.glob("case_*/rss_case.json")):
        c = json.loads(path.read_text())
        groups = []
        for v, group in enumerate(c["context_candidates"]):
            matched = [
                r
                for r in group["features"]
                if re.search(
                    PATTERNS[v],
                    "; ".join(e["description"] for e in r["annotation"]["explanations"]),
                    re.I,
                )
            ]
            groups.append(
                {
                    "variant": v + 1,
                    "candidate_count": len(group["features"]),
                    "keyword_navigation_matches": matched,
                }
            )
        cases.append({"case": path.parent.name, "prompt": c["clean_prompt"], "groups": groups})
    report = {
        "purpose": "解释关键词仅用于定位阅读，不代表语义验收，也不影响 RSS 选集。所有数值候选保存在原案例中。",
        "patterns": PATTERNS,
        "cases": cases,
    }
    (root / "explanation_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return cases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    args = parser.parse_args()
    original, alternative = args.original.resolve(), args.alternative.resolve()
    navigation(original)
    navigation(alternative)
    c = json.loads((alternative / "case_04/rss_case.json").read_text())
    lines = [
        "# 第二轮：同一个真实案例的对照结果",
        "",
        "**已得到三条与目标相关、双向变化的 RSS 特征；尚未找到五条分别对应新增上下文的 RSS-excluded 特征，完整验收条件未满足。**",
        "",
        "GPT-2 Small，layer 8，SAE `gpt2small_fs_24576_l8`（Neuronpedia `gpt2-small/8-res_fs24576-jb`）。两张表均来自这一个 SAE、同一个 clean 与同一组五个 variants，在最后一个目标 token（句号）提取完整原始 SAE codes。",
        "",
        f"Clean：{c['clean_prompt']}",
        "",
    ]
    lines += [f"- V{i}：{p}" for i, p in enumerate(c["variants"], 1)]
    lines += [
        "",
        "## 表 A：原始 RSS 选中的三个目标相关特征",
        "",
        "完整 RSS support 为 20，以下按公开解释节选其中三条。三条 recurrence 和 consistency 均为 1.0；解释未参与 RSS 选择。保留原始 Neuronpedia explanation。",
        "",
        "| Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    selected = []
    for fid in (3130, 8406, 15492):
        row = next(r for r in c["rss_features"] if r["feature_id"] == fid)
        lines.append(feature_row(row))
        selected.append(row)
    lines += [
        "",
        "F3130 在五个 variants 中均减少，F8406 和 F15492 均增加；三者解释均指向食材、食谱或烹饪步骤，适合展示保留目标相关特征的双向变化。",
        "",
        "## 表 B：五个上下文的筛选结果（五条合格 context features 尚未取得）",
        "",
        "下面保留最相关或变化最大的未通过候选，明确展示缺口。列出的候选均未被 RSS 选中且 recurrence < 0.8；它们不能被标成五个上下文语义已匹配的特征。V1 列语义最接近的体育候选，V2、V5 列变化最大的候选；V3、V4 在全字典中没有符合数值筛选条件的候选。",
        "",
        "| Variant | Feature | ΔV1 | ΔV2 | ΔV3 | ΔV4 | ΔV5 | recurrence | Neuronpedia explanation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    partial = []
    for v, fid in [(1, 3959), (2, 15550), (3, None), (4, None), (5, 12314)]:
        if fid is None:
            lines.append(f"| V{v} | 无合格数值候选 | — | — | — | — | — | — | — |")
        else:
            row = next(
                r for r in c["context_candidates"][v - 1]["features"] if r["feature_id"] == fid
            )
            lines.append(f"| V{v} | " + feature_row(row)[2:])
            partial.append({"variant": v, "row": row})
    lines += [
        "",
        "V1 的解释偏向足球等宽泛体育，变化仅 +0.136；V2 的解释是准备就绪，与宇航员不直接对应；V5 的最大独有响应反而被解释为烹饪。这里的数值集中性与上下文语义对应并未一致。V3、V4 的空集不是解释缺失。",
        "",
        "## 案例筛选过程与范围",
        "",
        "1. 首先对原 SAE `gpt2small_res_jb_l8` 运行香蕉面包和其余 23 个 demo prompts。香蕉面包的 F2215、F567 是明确的烹饪 RSS 特征，五次均减少；未找到五个新增上下文分别对应的特有特征。",
        "2. 在同一个 GPT-2 Small layer 8 改用已注册且有 Neuronpedia 来源的 `gpt2small_fs_24576_l8`，再次运行全部 24 个 prompts。香蕉面包得到表 A 的三条目标相关双向 RSS 特征，但表 B 仍存在上述缺口。",
        "3. 共保存 48 个完整候选案例，使用相同五个上下文、相同 RSS 参数和同一个目标末尾位置；不把不同 prompts、不同 SAE 或不同 token 位置拼成一个完整案例。",
        "4. context 数值筛选先遍历所有未被 RSS 选中的特征，要求 recurrence < 0.8、当前变体增加 > 1e-5、至少是其他变体最大绝对变化的五倍，按变化幅度排序保留全部。之后才查询公开解释。没有把微小独有响应直接称作强上下文编码。",
        "5. 为帮助逐项阅读，`explanation_review.json` 保存五组语义关键词定位到的候选与原始解释；关键词匹配只是导航，人工判断仍需看具体内容。额外查看完整激活矩阵发现：两组 SAE 的国际象棋相关解释特征均没有出现“V4 正变化且 recurrence < 0.8”的情况，单纯放宽五倍规则也不能补上该项。",
        "",
        "## 当前可以使用的结论",
        "",
        "表 A 可作为保留目标相关特征发生双向活动变化的案例。当前结果尚不支持‘RSS 同时区分五个新增上下文各自的编码’这一完整论断。末尾 token 的观测不足以说明模型在其他位置是否编码这些上下文；改变捕获位置属于另一组实验，应单独呈现。",
        "",
        f"[原 SAE 全部案例]({original / 'index.md'}) · [Feature Splitting SAE 全部案例]({alternative / 'index.md'}) · [本例全部 RSS 与数值候选]({alternative / 'case_04/rss_case.md'})",
    ]
    report_path = original / "contrast_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    (original / "case_selection.json").write_text(
        json.dumps(
            {
                "status": "acceptance_not_met",
                "searched_case_count": 48,
                "selected_partial_case": str(alternative / "case_04"),
                "target_features": selected,
                "context_partial_rows": partial,
                "missing": "No same case with five semantically aligned, RSS-excluded context features was found in these two SAEs and 24 targets.",
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    for directory in (original, alternative):
        index = directory / "index.md"
        text = index.read_text()
        marker = f"[先读：本轮对照结果与缺口]({report_path})"
        if marker not in text:
            text = text.replace("\n\n", "\n\n" + marker + "\n\n", 1)
        index.write_text(text)
    print(report_path)


if __name__ == "__main__":
    main()
