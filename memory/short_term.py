"""
memory/short_term.py
====================
短期对话记忆（会话内记忆）。

设计原理：
    - 「短期记忆」指当前会话内可被直接引用的对话上下文（用户说了什么、Agent 答了什么）。
      与「长期记忆」的差异：
          * 短期记忆：会话内的 Q/A 轮次，逐轮追加，跨轮加载，帮助 Agent 理解当前话题语境；
          * 长期记忆：跨会话沉淀的用户偏好 / 关键事实，注入到系统提示词作为背景知识。
    - 存储粒度：每轮保存「用户提问 + Agent 最终回答」成对（Q/A 对）。
      不保存中间的工具调用轨迹（体积大、恢复复杂且对下一轮无增益），
      最终回答本身已包含工具结果，足以支撑多轮上下文。
    - 加载时按时间顺序重建 LangChain 消息列表（HumanMessage / AIMessage 交替），
      直接拼接到 Agent 的状态消息中，实现跨轮记忆。

本模块只负责短期记忆的读写，不依赖图结构，便于单元测试。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.logging import get_logger
from memory.db import _connect, _now

logger = get_logger(__name__)

# 单次加载的最大历史轮次（Q/A 对数），防止上下文无限膨胀
_MAX_HISTORY_TURNS = 20


def _conv_exists(db_path: Path, conv_id: str) -> bool:
    """判断会话是否存在（用于幂等创建）。"""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return row is not None


def create_conversation(db_path: Path, session_id: Optional[str] = None) -> str:
    """创建一个新会话，返回会话 ID。

    若传入 session_id 且已存在，则直接复用（幂等），用于恢复历史会话。
    """
    conv_id = session_id or uuid.uuid4().hex
    if _conv_exists(db_path, conv_id):
        logger.info("复用已有会话: %s", conv_id)
        return conv_id
    now = _now()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, "", now, now),
        )
    logger.info("创建会话: %s", conv_id)
    return conv_id


def save_turn(
    db_path: Path,
    conv_id: str,
    user_text: str,
    assistant_text: str,
    meta: Optional[dict] = None,
) -> None:
    """保存一轮对话（用户提问 + Agent 回答）。

    Args:
        user_text: 用户本轮提问。
        assistant_text: Agent 本轮最终回答。
        meta: 扩展信息（JSON 序列化），如工具调用次数、思考链步数。
    """
    now = _now()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
            "VALUES (?, 'user', ?, '{}', ?)",
            (conv_id, user_text, now),
        )
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
            "VALUES (?, 'assistant', ?, ?, ?)",
            (conv_id, assistant_text, meta_json, now),
        )
        # 会话标题：首次对话时用提问前 30 字作为标题
        conn.execute(
            "UPDATE conversations SET updated_at = ?, title = CASE WHEN title = '' "
            "THEN substr(?, 1, 30) ELSE title END WHERE id = ?",
            (now, user_text, conv_id),
        )
    logger.info("已保存对话轮次: conv=%s, 用户%d字/回答%d字", conv_id, len(user_text), len(assistant_text))


def load_history(db_path: Path, conv_id: str, max_turns: int = _MAX_HISTORY_TURNS) -> list[BaseMessage]:
    """加载会话历史，重建为 LangChain 消息列表（供 Agent 状态拼接）。

    按时间顺序取最近 max_turns 轮 Q/A，交替重建 HumanMessage / AIMessage。
    返回 [] 表示无历史（新会话或记忆未启用）。
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? AND role IN ('user','assistant') "
            "ORDER BY id DESC LIMIT ?",
            (conv_id, max_turns * 2),
        ).fetchall()

    # 逆序拿的是「最近在前」，需反转回时间正序
    rows = list(reversed(rows))
    messages: list[BaseMessage] = []
    for row in rows:
        if row["role"] == "user":
            messages.append(HumanMessage(content=row["content"]))
        else:
            messages.append(AIMessage(content=row["content"]))
    if messages:
        logger.info("加载短期记忆: conv=%s, %d 轮", conv_id, len(messages) // 2)
    return messages


def list_conversations(db_path: Path, limit: int = 20) -> list[dict]:
    """列出最近会话（标题 + 会话 ID），供会话恢复/展示使用。"""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
