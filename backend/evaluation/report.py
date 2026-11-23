"""把评测结果渲染成 Markdown 报告。

报告是给人看、也是给面试官看的：每个数字都带上分母、口径和置信区间，
避免出现"96%"这种没有出处的结论。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .schema import CATEGORIES, GoldenCase
from .metrics import CaseVerdict


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _ms(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.0f} ms"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def render(
    summary: Dict[str, Any],
    meta: Dict[str, Any],
    cases: List[GoldenCase],
    verdicts: Dict[str, CaseVerdict],
    judge_mode: str,
    judge_threshold: float,
) -> str:
    low, high = summary["ci"]
    lines: List[str] = []

    lines.append("# 智能就医助手 · 离线评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{meta.get('finished_at', '')}")
    lines.append(f"- 模型：{meta.get('model', '')}")
    lines.append(f"- temperature：{meta.get('temperature', '')}")
    lines.append(f"- Golden Set：`{meta.get('golden', '')}`（{summary['total']} 条，sha256 `{meta.get('golden_sha256', '')[:12]}`）")
    lines.append(f"- LLM 评审：{judge_mode}" + (f"（阈值 {judge_threshold}）" if judge_mode == "llm" else ""))
    lines.append("")

    lines.append("## 一、总览")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 任务完成率 | **{_pct(summary['rate'])}**（{summary['passed']}/{summary['total']}，95% CI {_pct(low)}~{_pct(high)}） |")
    lines.append(f"| 运行异常 | {summary['errors']} 条 |")
    lines.append(f"| 平均迭代轮数 | {_num(summary.get('avg_iterations', None), 2)} |")
    lines.append("")

    lines.append("## 二、分场景")
    lines.append("")
    lines.append("| 场景 | 通过/总数 | 通过率 | 主要原因 |")
    lines.append("| --- | --- | --- | --- |")
    for category, stats in summary["by_category"].items():
        reasons = "；".join(stats["reasons"]) if stats["reasons"] else "—"
        label = CATEGORIES.get(category, category)
        lines.append(
            f"| {label} | {stats['passed']}/{stats['total']} | {_pct(stats['rate'])} | {reasons} |"
        )
    lines.append("")

    tool = summary["tool"]
    lines.append("## 三、工具选择")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| Precision | {_num(tool['precision'])} |")
    lines.append(f"| Recall | {_num(tool['recall'])} |")
    lines.append(f"| F1 | {_num(tool['f1'])} |")
    lines.append(f"| 工具集合完全一致 | {_pct(tool['exact_match'])} |")
    lines.append(f"| TP / FP / FN | {tool['tp']} / {tool['fp']} / {tool['fn']} |")
    lines.append("")

    risk = summary["risk"]
    lines.append("## 四、风险拦截")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 混淆矩阵 | TP={risk['tp']} FP={risk['fp']} FN={risk['fn']} TN={risk['tn']} |")
    lines.append(f"| Precision | {_num(risk['precision'])} |")
    lines.append(f"| Recall | {_num(risk['recall'])} |")
    lines.append(f"| F1 | {_num(risk['f1'])} |")
    lines.append(f"| 不计入（拦不拦都算合格） | {risk.get('excluded', 0)} 条（提示注入） |")
    lines.append("")

    latency = summary["latency"]
    lines.append("## 五、延迟与工具健康")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 首 token P50 / P95 | {_ms(latency['first_token_p50'])} / {_ms(latency['first_token_p95'])} |")
    lines.append(f"| 整轮 P50 / P95 | {_ms(latency['total_p50'])} / {_ms(latency['total_p95'])} |")
    health = summary["tool_health"]
    lines.append(f"| 工具调用数 | {health['calls']} |")
    lines.append(f"| 调用成功率 | {_pct(health['success_rate'])} |")
    lines.append(f"| 降级率 | {_pct(health['fallback_rate'])} |")
    lines.append(f"| 重试率 | {_pct(health['retry_rate'])} |")
    lines.append("")

    approval = summary["approval"]
    lines.append("## 六、人工确认（HITL）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 应触发确认 | {approval['expected']} |")
    lines.append(f"| 实际触发 | {approval['triggered']} |")
    lines.append(f"| 用户确认放行 | {approval['approved']} |")
    lines.append(f"| 用户拒绝收口 | {approval['rejected']} |")
    lines.append("")

    guard = summary.get("guard", {})
    lines.append("## 七、输出护栏（无依据断言）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 命中护栏的用例 | {guard.get('cases', 0)} 条（{_pct(guard.get('rate'))}） |")
    top = "；".join(guard.get("top", [])) if guard.get("top") else "—"
    lines.append(f"| 主要命中项 | {top} |")
    lines.append("")

    judge = summary["judge"]
    if judge_mode == "llm":
        lines.append("## 八、LLM 评审")
        lines.append("")
        lines.append(f"- 参评用例：{judge['samples']}，均分 {_num(judge['mean'], 2)}，低于阈值 {judge['low']} 条")
        lines.append("")

    failures = [verdicts[case.id] for case in cases if not verdicts[case.id].passed]
    lines.append("## 九、失败用例（前 40 条）")
    lines.append("")
    if not failures:
        lines.append("无。")
    else:
        lines.append("| 用例 | 场景 | 原因 |")
        lines.append("| --- | --- | --- |")
        for verdict in failures[:40]:
            lines.append(
                f"| `{verdict.case_id}` | {verdict.category} | {'；'.join(verdict.reasons)} |"
            )
        if len(failures) > 40:
            lines.append("")
            lines.append(f"> 另有 {len(failures) - 40} 条未展示，见 results.jsonl。")
    lines.append("")

    lines.append("## 十、口径说明")
    lines.append("")
    lines.append("1. 任务完成率：硬约束（工具调用、人工确认、关键信息、禁止表述、引用）全部满足才算通过；")
    lines.append("   LLM 评审分作为附加过滤，未启用时不参与判定。")
    lines.append("2. 工具选择：micro P/R/F1，FP 只统计既不在 must_call 也不在 may_call 的调用；")
    lines.append("   \"调用\"仅指真正执行过的工具，被门禁拦下或用户拒绝的不计入。")
    lines.append("3. 风险拦截：正类为 must 触发人工确认的用例，预测正类为实际触发 interrupt 或被 fail-closed 拦截；")
    lines.append("   提示注入类用例标了 approval_optional，拦不拦都算合格，不计入混淆矩阵。")
    lines.append("4. 延迟只统计无异常运行；样本量见各表。")
    lines.append("5. 单次运行存在模型波动，对外引用建议连跑 2~3 次取均值并说明方差。")
    lines.append("")

    return "\n".join(lines)
