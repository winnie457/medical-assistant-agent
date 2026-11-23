"""重启恢复基准：验证"确认前挂起 → 进程退出 → 新进程续跑"是否成立。

用法（在 backend/ 下）：
    python -m evaluation.recovery_bench                  # 一键：子进程 start + 本进程 resume
    python -m evaluation.recovery_bench --cases 8
    python -m evaluation.recovery_bench --phase start    # 只跑前半段（制造挂起）
    python -m evaluation.recovery_bench --phase resume   # 只跑后半段（续跑）

原理：
    start 阶段跑若干高危用药问题，触发 interrupt 后**不做确认**，直接结束进程；
    resume 阶段在另一个进程里按 run_id 从 checkpoint 恢复并继续。
    只有 checkpoint 真正落盘（AsyncSqliteSaver）才能成功；
    用 InMemorySaver 时这段必然失败——这正是改造前后要对比的数字。

注意：
    - start 与 resume 必须是两个进程，否则测不出"重启"；
    - 需要配置好 .env（模型 Key），会真实调用模型。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_STATE = Path(__file__).with_name("reports") / "recovery" / "state.json"

# 全部是高危用药问题：一定会触发人工确认，适合用来制造中断
CASES = [
    "我想问下布洛芬，一天吃几次、一次吃多少？",
    "阿莫西林饭前吃还是饭后吃？",
    "头孢和酒能一起吗？",
    "哺乳期能吃氯雷他定吗？",
    "孩子发烧能不能吃布洛芬？",
    "奥美拉唑能长期吃吗？",
    "二甲双胍能不能和降压药一起吃？",
    "蒙脱石散一次吃几袋？",
]


async def _start(state_file: Path, count: int, temperature: float) -> int:
    from .agent_factory import build_agent

    agent, _ = build_agent(temperature)
    records: List[Dict[str, Any]] = []
    for text in CASES[:count]:
        run_id = None
        conversation_id = None
        stream = agent.chat_stream(message=text, city="北京", user_id="recovery-bench")
        try:
            async for event in stream:
                if event.get("type") == "approval_required":
                    # 记录但不提前退出：让事件流自然结束，确保 interrupt 写入 checkpoint
                    run_id = event.get("run_id")
                    conversation_id = event.get("conversation_id")
        finally:
            await stream.aclose()
        records.append({"case": text, "run_id": run_id, "conversation_id": conversation_id})

    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    kind = agent._checkpointer_provider.kind  # noqa: SLF001 - 基准脚本需要报告实现
    await agent.aclose()  # aiosqlite 连接不关，进程不会退出
    hit = sum(1 for record in records if record["run_id"])
    print(f"[start] checkpointer={kind}  挂起 {hit}/{len(records)} 条 → {state_file}")
    return 0 if hit == len(records) else 1


async def _resume(state_file: Path, temperature: float) -> Tuple[int, int]:
    from .agent_factory import build_agent

    if not state_file.exists():
        raise SystemExit(f"状态文件不存在: {state_file}（先跑 --phase start）")

    agent, _ = build_agent(temperature)
    records = json.loads(state_file.read_text(encoding="utf-8"))
    pending = [record for record in records if record.get("run_id")]
    ok = 0
    for record in pending:
        answer = ""
        done = False
        stream = agent.resume_stream(
            record["run_id"], {"approved": True}, record.get("conversation_id")
        )
        try:
            async for event in stream:
                if event.get("type") == "answer":
                    answer = event.get("content") or ""
                elif event.get("type") == "done":
                    done = True
        finally:
            await stream.aclose()
        if done and answer.strip():
            ok += 1
        else:
            print(f"  [失败] {record['case'][:20]}... run_id={record['run_id']} 未完成")

    await agent.aclose()
    rate = ok / len(pending) if pending else 0.0
    print(f"[resume] 跨进程恢复成功 {ok}/{len(pending)} = {rate * 100:.1f}%")
    return ok, len(pending)


def _run_phase(phase: str, state_file: Path, count: int, temperature: float) -> int:
    if phase == "start":
        return asyncio.run(_start(state_file, count, temperature))
    ok, total = asyncio.run(_resume(state_file, temperature))
    return 0 if total and ok == total else 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重启恢复基准（跨进程续跑）")
    parser.add_argument("--phase", choices=["all", "start", "resume"], default="all")
    parser.add_argument("--cases", type=int, default=5, help="制造挂起的用例数")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args(argv)

    if args.phase != "all":
        return _run_phase(args.phase, args.state_file, args.cases, args.temperature)

    # all：start 放进子进程，resume 留在本进程 —— 两个进程，才算"重启"
    backend_dir = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable, "-m", "evaluation.recovery_bench",
        "--phase", "start",
        "--state-file", str(args.state_file),
        "--cases", str(args.cases),
        "--temperature", str(args.temperature),
    ]
    started = subprocess.run(cmd, cwd=str(backend_dir))
    if started.returncode != 0:
        print("[all] start 阶段失败，跳过 resume", file=sys.stderr)
        return started.returncode
    ok, total = asyncio.run(_resume(args.state_file, args.temperature))
    return 0 if total and ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
