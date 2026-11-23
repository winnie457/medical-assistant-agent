"""评测入口。

用法（在 backend/ 目录下执行）：
    python -m evaluation.build_golden_set          # 生成 evaluation/golden_set.jsonl
    python -m evaluation.run_eval --dry-run        # 只校验用例，不调用模型
    python -m evaluation.run_eval --limit 10       # 先小样本冒烟
    python -m evaluation.run_eval --judge llm      # 全量 + LLM 评审
    python -m evaluation.run_eval --out evaluation/reports/run1 --fail-under 0.8

产物（--out 目录）：
    report.md      Markdown 报告（可直接贴进 README / 面试材料）
    results.jsonl  逐条结果（含回答与原因，便于复盘）
    meta.json      运行元信息（模型、温度、用例哈希）
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import RunRecord, aggregate, check_case
from .schema import CATEGORIES, GoldenCase, load_cases, summarize_cases

DEFAULT_GOLDEN = Path(__file__).with_name("golden_set.jsonl")
DEFAULT_OUT = Path(__file__).with_name("reports")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能就医助手离线评测")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Golden Set JSONL 路径")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 表示全量）")
    parser.add_argument("--category", action="append", default=[], help="只跑指定类别，可重复")
    parser.add_argument("--judge", choices=["none", "llm"], default="none", help="是否启用 LLM 评审")
    parser.add_argument("--judge-threshold", type=float, default=0.7, help="评审分低于该值判失败")
    parser.add_argument("--concurrency", type=int, default=3, help="并发用例数")
    parser.add_argument("--temperature", type=float, default=0.0, help="评测用 temperature")
    parser.add_argument("--out", type=Path, default=None, help="产物目录")
    parser.add_argument("--fail-under", type=float, default=0.0, help="通过率低于该值时以非 0 退出（CI 用）")
    parser.add_argument("--dry-run", action="store_true", help="只校验用例，不调用模型")
    return parser.parse_args(argv)


def _filter_cases(cases: List[GoldenCase], args: argparse.Namespace) -> List[GoldenCase]:
    selected = cases
    if args.category:
        unknown = [c for c in args.category if c not in CATEGORIES]
        if unknown:
            raise SystemExit(f"未知类别: {unknown}，可选: {list(CATEGORIES)}")
        selected = [c for c in selected if c.category in set(args.category)]
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _print_dry_run(cases: List[GoldenCase]) -> None:
    stats = summarize_cases(cases)
    print(f"用例总数: {stats['total']}（{stats['turns_total']} 轮对话）")
    for category, count in stats["by_category"].items():
        print(f"  {category:<18} {count:>3}  {CATEGORIES[category]}")
    print(f"风险分布: {stats['by_risk']}")
    print(f"需人工确认: {stats['approval_cases']}，参与评审: {stats['judge_cases']}")
    print("must_call 覆盖:")
    for tool, count in sorted(stats["must_call_coverage"].items(), key=lambda kv: -kv[1]):
        print(f"  {tool:<24} {count:>3}")


def _console_summary(summary: Dict, out_dir: Path) -> None:
    low, high = summary["ci"]
    print("")
    print(f"[总体] 通过 {summary['passed']}/{summary['total']} = {summary['rate'] * 100:.1f}%"
          f"（95% CI {low * 100:.1f}%~{high * 100:.1f}%）  异常 {summary['errors']} 条")
    tool = summary["tool"]
    print(f"[工具] P {tool['precision'] or 0:.2f}  R {tool['recall'] or 0:.2f}  "
          f"F1 {tool['f1'] or 0:.2f}  EM {tool['exact_match'] * 100:.1f}%")
    risk = summary["risk"]
    print(f"[风险] TP {risk['tp']} FP {risk['fp']} FN {risk['fn']} TN {risk['tn']}  "
          f"P {risk['precision'] or 0:.2f}  R {risk['recall'] or 0:.2f}")
    latency = summary["latency"]
    print(f"[延迟] 首token P50 {latency['first_token_p50'] or 0:.0f}ms  P95 {latency['first_token_p95'] or 0:.0f}ms")
    health = summary["tool_health"]
    print(f"[工具] 调用 {health['calls']} 次  成功率 {(health['success_rate'] or 0) * 100:.1f}%  "
          f"降级率 {(health['fallback_rate'] or 0) * 100:.1f}%")
    print(f"产物目录: {out_dir}")


async def _run(args: argparse.Namespace, cases: List[GoldenCase]) -> Dict:
    from .agent_factory import build_agent, model_label
    from .judge import make_judge
    from .report import render
    from .runner import run_all

    agent, llm = build_agent(temperature=args.temperature)
    judge = make_judge(llm) if args.judge == "llm" else None

    started = time.time()
    records = await run_all(agent, cases, concurrency=args.concurrency, judge=judge)
    elapsed = time.time() - started

    runs: Dict[str, RunRecord] = {record.case_id: record for record, _ in records}
    scores = {record.case_id: score for record, score in records}
    verdicts = {
        case.id: check_case(case, runs[case.id], scores.get(case.id), args.judge_threshold)
        for case in cases
    }
    summary = aggregate(cases, runs, verdicts)
    summary["avg_iterations"] = (
        sum(r.iterations for r in runs.values()) / len(runs) if runs else None
    )
    await agent.aclose()
    return {
        "summary": summary,
        "verdicts": verdicts,
        "runs": runs,
        "scores": scores,
        "render": render,
        "meta": {
            "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "model": model_label(),
            "temperature": args.temperature,
            "golden": str(args.golden),
            "golden_sha256": _sha256(args.golden),
            "judge": args.judge,
            "judge_threshold": args.judge_threshold,
            "concurrency": args.concurrency,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cases = load_cases(args.golden)
    selected = _filter_cases(cases, args)

    if args.dry_run:
        _print_dry_run(selected)
        return 0

    if not selected:
        print("没有匹配的用例。", file=sys.stderr)
        return 2

    out_dir = args.out or DEFAULT_OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(_run(args, selected))
    summary, verdicts, runs, scores, render, meta = (
        result["summary"], result["verdicts"], result["runs"], result["scores"], result["render"], result["meta"],
    )

    with (out_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as fp:
        for case in selected:
            run = runs[case.id]
            fp.write(json.dumps({
                "id": case.id,
                "category": case.category,
                "passed": verdicts[case.id].passed,
                "reasons": verdicts[case.id].reasons,
                "judge_score": scores.get(case.id),
                "used_tools": verdicts[case.id].used_tools,
                "approval_count": run.approval_count,
                "decision": run.decision,
                "iterations": run.iterations,
                "first_token_ms": run.first_token_ms,
                "total_ms": round(run.total_ms, 1),
                "error": run.error,
                "answer": run.answer,
            }, ensure_ascii=False) + "\n")

    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        render(summary, meta, selected, verdicts, args.judge, args.judge_threshold),
        encoding="utf-8",
    )

    _console_summary(summary, out_dir)

    if args.fail_under and summary["rate"] < args.fail_under:
        print(f"通过率 {summary['rate'] * 100:.1f}% 低于阈值 {args.fail_under * 100:.1f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
