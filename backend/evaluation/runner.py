"""评测运行器：把 Golden Case 跑在真实的 LangGraph Agent 上。

两条链路都覆盖：
- stream：chat_stream → 若出现 approval_required 则用 resume_stream 续跑，
  用来测人工确认（HITL）的"确认/拒绝"两条分支；
- sync：chat，没有确认通道，用来测高风险工具的 fail-closed。

只观测、不改业务逻辑；所有异常都被收敛成 RunRecord.error，避免一条用例炸掉整轮评测。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage

from .metrics import RunRecord
from .schema import GoldenCase

JudgeFn = Callable[[GoldenCase, str], Awaitable[Optional[float]]]
EVAL_USER_ID = "eval"


def _decision(case: GoldenCase) -> Dict[str, Any]:
    """评测里模拟的用户决策。

    只有用例明确要求"确认"时才放行；其余情况一律按拒绝处理——这与真实系统
    的 fail-closed 取向一致，也能防止意外触发的高危调用被评测悄悄放行。
    """
    if case.expect.expect_approval and case.expect.approval_decision == "approve":
        return {"approved": True, "reason": "评测：用户确认"}
    return {"approved": False, "reason": "评测：用户拒绝"}


async def _run_turn(
    agent: Any,
    case: GoldenCase,
    turn: str,
    history: List[Any],
    record: RunRecord,
    started: float,
) -> str:
    """跑一轮（含可能的确认续跑），返回本轮最终答案。"""
    answer = ""
    pending: Optional[Dict[str, Any]] = None

    stream = agent.chat_stream(
        message=turn,
        history=list(history),
        city=case.city,
        user_id=EVAL_USER_ID,
    )
    try:
        async for event in stream:
            etype = event.get("type")
            if etype == "answer_delta" and record.first_token_ms is None:
                record.first_token_ms = (time.perf_counter() - started) * 1000
            elif etype == "approval_required":
                # 不要 break：提前关闭生成器会让 interrupt 的 checkpoint 来不及落盘，
                # 换进程续跑时就找不到这次挂起。这里只记录，继续消费到流自然结束。
                pending = event
            elif etype == "answer":
                answer = event.get("content") or answer
            elif etype == "done":
                record.tool_events.extend(event.get("tool_calls_history") or [])
                record.iterations += int(event.get("iteration_count") or 0)
                record.guard_issues = list(event.get("guard_issues") or record.guard_issues)
    finally:
        await stream.aclose()

    if pending is None:
        return answer

    record.approval_count += 1
    decision = _decision(case)
    record.decision = "approve" if decision["approved"] else "reject"

    resumed = agent.resume_stream(
        run_id=pending.get("run_id", ""),
        decision=decision,
        conversation_id=pending.get("conversation_id"),
    )
    try:
        async for event in resumed:
            etype = event.get("type")
            if etype == "answer_delta" and record.first_token_ms is None:
                record.first_token_ms = (time.perf_counter() - started) * 1000
            elif etype == "answer":
                answer = event.get("content") or answer
            elif etype == "done":
                record.tool_events.extend(event.get("tool_calls_history") or [])
                record.iterations += int(event.get("iteration_count") or 0)
                record.guard_issues = list(event.get("guard_issues") or record.guard_issues)
    finally:
        await resumed.aclose()

    return answer


async def run_case(agent: Any, case: GoldenCase) -> RunRecord:
    started = time.perf_counter()
    record = RunRecord(case_id=case.id, category=case.category)
    history: List[Any] = []

    try:
        if case.mode == "sync":
            result = await agent.chat(
                message=case.turns[0],
                history=list(history),
                city=case.city,
                user_id=EVAL_USER_ID,
            )
            record.answer = result.get("answer", "")
            record.tool_events = list(result.get("tool_calls_history") or [])
            record.iterations = int(result.get("iteration_count") or 0)
        else:
            for turn in case.turns:
                answer = await _run_turn(agent, case, turn, history, record, started)
                history.append(HumanMessage(content=turn))
                history.append(AIMessage(content=answer))
                record.answer = answer
    except Exception as exc:  # noqa: BLE001 - 评测要吞掉单条用例的异常
        record.error = f"{type(exc).__name__}: {exc}"

    record.total_ms = (time.perf_counter() - started) * 1000
    return record


async def run_all(
    agent: Any,
    cases: List[GoldenCase],
    concurrency: int = 3,
    judge: Optional[JudgeFn] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Tuple[RunRecord, Optional[float]]]:
    """并发跑完全部用例；返回 (RunRecord, judge_score) 列表（顺序与 cases 一致）。"""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total = len(cases)
    done = 0

    async def _one(case: GoldenCase) -> Tuple[RunRecord, Optional[float]]:
        nonlocal done
        async with semaphore:
            record = await run_case(agent, case)
            score: Optional[float] = None
            if judge is not None and case.judge and not record.error:
                try:
                    score = await judge(case, record.answer)
                except Exception:  # noqa: BLE001 - 评审失败不应影响硬指标
                    score = None
            done += 1
            if progress:
                progress(done, total, case.id)
            return record, score

    return await asyncio.gather(*(_one(case) for case in cases))
