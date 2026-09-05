"""
memory/db.py
============
SQLite 数据库层：负责建表与基础 CRUD。

设计原理：
    - 为什么用 SQLite 而非 MySQL/PostgreSQL？
        本项目为单机 CLI/容器部署，SQLite 是「零配置、单文件、事务完备、进程内」的
        嵌入式数据库，完全满足对话记忆的持久化需求，且无需额外启动数据库服务，
        天然契合 Docker 单容器部署（数据文件挂载卷即可持久化）。
    - 线程安全：
        数据库访问只发生在 Agent 图节点（agent/tools 节点）与 run_agent 主流程，
        均为串行调用；工具并行执行（ThreadPoolExecutor）不触碰数据库。
        连接使用 check_same_thread=False + 每次操作短开短关，规避跨线程复用风险。
    - WAL 模式：
        开启 Write-Ahead Logging，读不阻塞写、写不阻塞读，提升并发读写体验。

表结构：
    conversations      会话表（短期记忆的容器）
    messages          消息表（每轮 用户提问 + AI 回答，Q/A 对）
    long_term_memories 长期记忆表（跨会话的持久事实 / 用户偏好）
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)

# 建表 DDL（幂等：IF NOT EXISTS，重复执行安全）
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,              -- 会话 ID（UUID）
    title       TEXT NOT NULL DEFAULT '',      -- 会话标题（首问前 N 字）
    created_at  TEXT NOT NULL,                 -- 创建时间（ISO8601 UTC）
    updated_at  TEXT NOT NULL                  -- 最后更新时间
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,             -- 所属会话
    role            TEXT NOT NULL,             -- user / assistant / system
    content         TEXT NOT NULL,             -- 消息正文
    meta            TEXT NOT NULL DEFAULT '{}',-- 扩展元信息（JSON：工具调用次数/思考链等）
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS long_term_memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL DEFAULT 'default', -- 用户维度（当前单用户）
    content     TEXT NOT NULL,                   -- 长期记忆内容
    source      TEXT NOT NULL DEFAULT '',        -- 来源（会话 ID）
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON long_term_memories(user_id);
"""


def _now() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串，用于各表时间戳字段。"""
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    """打开数据库连接并应用 PRAGMA 优化。

    - check_same_thread=False：允许跨线程使用（配合短开短关策略）。
    - journal_mode=WAL：写读互不阻塞。
    - foreign_keys=ON：启用外键约束。
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row          # 行按列名访问
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path) -> None:
    """初始化数据库：创建目录、执行建表 DDL（幂等，可重复调用）。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
    logger.info("数据库就绪: %s", db_path)


@contextmanager
def db_connection(db_path: Path):
    """提供带事务的数据库连接上下文。

    用法：
        with db_connection(db_path) as conn:
            conn.execute(...)   # 提交在退出时自动完成
    任何异常会回滚事务并向上抛出，保证数据一致性。
    """
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
