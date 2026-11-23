"""SQLite 连接与建表（应用侧数据：会话、消息、运行记录）。

为什么不用 ORM：
    这里只有三张表，操作都是主键点查和按会话列举，sqlite3 足够；
    真正需要高频并发写入的是 Agent 的 checkpoint，由 LangGraph 的 saver 独立管理，
    两边共用一个数据库文件（WAL 模式），互不干扰。

连接策略：
    进程内单例 + 可重入锁。FastAPI 的处理器是异步的，但这里的 SQL 都是毫秒级
    点查，直接同步执行比引入 aiosqlite 更简单，也不会把连接绑定到某个事件循环。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

# backend/ 目录（app/storage/db.py -> parents[2]）
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BASE_DIR / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_message    TEXT NOT NULL DEFAULT '',
    message_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_calls      TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id          TEXT PRIMARY KEY,
    conversation_id TEXT,
    status          TEXT NOT NULL,
    detail          TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status, updated_at);
"""

_connection: Optional[sqlite3.Connection] = None
_lock = threading.RLock()


def db_path() -> Path:
    """数据库路径，可用 APP_DB_PATH 覆盖（评测/测试用临时库）。"""
    raw = os.getenv("APP_DB_PATH")
    path = Path(raw) if raw else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    """进程内单例连接（WAL + 外键约束）。"""
    global _connection
    with _lock:
        if _connection is None:
            conn = sqlite3.connect(str(db_path()), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            conn.commit()
            _connection = conn
        return _connection


def reset_connection() -> None:
    """关闭并丢弃单例连接（切换数据库 / 测试用）。"""
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None
