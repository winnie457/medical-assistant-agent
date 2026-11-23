"""Agent 运行记录：run_id ↔ 会话 ↔ 状态。

为什么需要它：
    中断态原先只活在进程内存（InMemorySaver + 客户端手里的 run_id），
    进程一重启就查不到这次运行到底是在跑、在等确认、还是已完成。
    把运行记录落库后，`GET /api/chat/runs/{run_id}` 可以在任意进程里回答
    "这次运行什么状态、要不要继续确认"，跨进程续跑才有入口。

状态机：
    running   → 已开始，尚未结束
    interrupted → 等待用户确认（高风险门禁挂起）
    completed → 正常结束
    failed    → 异常结束
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..storage.db import get_connection

RUNNING = "running"
INTERRUPTED = "interrupted"
COMPLETED = "completed"
FAILED = "failed"

STATUSES = (RUNNING, INTERRUPTED, COMPLETED, FAILED)


def record_run(
    run_id: str,
    conversation_id: Optional[str] = None,
    status: str = RUNNING,
    detail: str = "",
) -> None:
    """写入（或覆盖）一条运行记录。"""
    if not run_id:
        return
    now = datetime.now().isoformat()
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO agent_runs (run_id, conversation_id, status, detail, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id) DO UPDATE SET"
            "   conversation_id = COALESCE(excluded.conversation_id, agent_runs.conversation_id),"
            "   status = excluded.status, detail = excluded.detail, updated_at = excluded.updated_at",
            (run_id, conversation_id, status, detail, now, now),
        )


def update_run(run_id: str, status: str, detail: str = "") -> None:
    """更新状态（记录不存在时补一条）。"""
    record_run(run_id, None, status, detail)


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    row = get_connection().execute(
        "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return dict(row) if row else None


def list_runs(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    if status:
        rows = get_connection().execute(
            "SELECT * FROM agent_runs WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = get_connection().execute(
            "SELECT * FROM agent_runs ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
