"""并发基准：N 路并发流式会话的 P50 / P95 与错误率。

用法（在 backend/ 下）：
    python -m evaluation.concurrency_bench                    # 默认 50 并发
    python -m evaluation.concurrency_bench --concurrency 20 --repeat 2
    python -m evaluation.concurrency_bench --warmup 2         # 先跑几条热身

口径：
    - 只测"链路本身"，用的是无需工具的问题，避免外部 API（高德/搜索）拖慢统计；
    - 首 token 延迟 = 请求开始到第一个 answer_delta；
    - 失败包括：抛异常、没有收到 done、答案为空；
    - 会话按 user_id 隔离（bench-<i>），避免共享健康画像造成写竞争。
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List

from .metrics import percentile

PROMPTS = [
    "你好",
    "你是谁？",
    "你能做什么？",
    "谢谢",
    "介绍一下你自己",
    "感冒了要注意什么？",
    "平时怎么预防高血压？",
    "喝水有什么讲究？",
]


async def _one(agent: Any, text: str, index: int) -> Dict[str, Any]:
    started = time.perf_counter()
    first_token_ms = None
    done = False
    answer = ""
    try:
        stream = agent.chat_stream(message=text, city="北京", user_id=f"bench-{index}")
        try:
            async for event in stream:
                etype = event.get("type")
                if etype == "answer_delta" and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                elif etype == "answer":
                    answer = event.get("content") or answer
                elif etype == "done":
                    done = True
        finally:
            await stream.aclose()
    except Exception as exc:  # noqa: BLE001 - 压测要统计失败而不是中断
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "first_token_ms": first_token_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
        }

    return {
        "ok": done and bool(answer.strip()),
        "error": "" if done else "未收到 done 事件",
        "first_token_ms": first_token_ms,
        "total_ms": (time.perf_counter() - started) * 1000,
    }


async def _run(concurrency: int, repeat: int, warmup: int, temperature: float) -> Dict[str, Any]:
    from .agent_factory import build_agent

    agent, _ = build_agent(temperature)
    await agent.ensure_graph()
    kind = agent._checkpointer_provider.kind  # noqa: SLF001 - 基准脚本需要报告实现

    for index in range(warmup):
        await _one(agent, PROMPTS[index % len(PROMPTS)], index)

    results: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()
    for round_index in range(repeat):
        tasks = [
            _one(agent, PROMPTS[(round_index * concurrency + i) % len(PROMPTS)], i)
            for i in range(concurrency)
        ]
        results.extend(await asyncio.gather(*tasks))
    wall = time.perf_counter() - wall_start

    ok = [r for r in results if r["ok"]]
    first_tokens = [r["first_token_ms"] for r in ok if r["first_token_ms"] is not None]
    totals = [r["total_ms"] for r in results]
    await agent.aclose()

    return {
        "checkpointer": kind,
        "requests": len(results),
        "success": len(ok),
        "error_rate": 1 - len(ok) / len(results) if results else 0.0,
        "first_token_p50": percentile(first_tokens, 0.5),
        "first_token_p95": percentile(first_tokens, 0.95),
        "total_p50": percentile(totals, 0.5),
        "total_p95": percentile(totals, 0.95),
        "wall_seconds": wall,
        "qps": len(results) / wall if wall > 0 else 0.0,
        "errors": [r["error"] for r in results if not r["ok"]][:5],
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="并发压测（流式会话）")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=1, help="重复轮数")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("reports") / "concurrency")
    args = parser.parse_args(argv)

    summary = asyncio.run(_run(args.concurrency, args.repeat, args.warmup, args.temperature))

    print(f"checkpointer={summary['checkpointer']}  请求 {summary['requests']} 条")
    print(f"成功率 {(1 - summary['error_rate']) * 100:.1f}%  错误 {summary['requests'] - summary['success']} 条")
    print(
        f"首 token P50 {summary['first_token_p50'] or 0:.0f}ms  P95 {summary['first_token_p95'] or 0:.0f}ms"
    )
    print(f"整轮 P50 {summary['total_p50'] or 0:.0f}ms  P95 {summary['total_p95'] or 0:.0f}ms")
    print(f"墙钟 {summary['wall_seconds']:.1f}s  吞吐 {summary['qps']:.2f} req/s")
    if summary["errors"]:
        print("错误样本:", summary["errors"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已写入 {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
