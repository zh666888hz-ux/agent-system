"""
memory/repository.py
====================
记忆仓库（Repository）门面。

设计原理：
    - 短期记忆与长期记忆的读写细节由各自模块实现，这里提供一个「统一入口」，
      让 Agent 图 / CLI 只需调用 MemoryRepository 一个类，
      不需要关心数据库路径、表结构、序列化细节（门面模式）。
    - 依赖注入：构造时传入 db_path，便于测试时指向临时数据库。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage

from core.logging import get_logger
from memory import db as db_mod
from memory.long_term import (
    build_memory_context,
    extract_long_term_memories,
    load_long_term_memories,
    save_long_term_memories,
)
from memory.short_term import (
    create_conversation,
    list_conversations,
    load_history,
    save_turn,
)

logger = get_logger(__name__)


class MemoryRepository:
    """短期 + 长期记忆的统一访问门面。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        db_mod.init_db(self.db_path)

    # ---------- 短期记忆 ----------
    def new_session(self, session_id: Optional[str] = None) -> str:
        """创建（或复用）会话，返回会话 ID。"""
        return create_conversation(self.db_path, session_id)

    def save_qa(self, session_id: str, question: str, answer: str, meta: Optional[dict] = None) -> None:
        """保存一轮 Q/A 到短期记忆。"""
        save_turn(self.db_path, session_id, question, answer, meta)

    def load_session_history(self, session_id: str, max_turns: int = 20) -> list[BaseMessage]:
        """加载会话历史消息。"""
        return load_history(self.db_path, session_id, max_turns)

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """列出最近会话。"""
        return list_conversations(self.db_path, limit)

    # ---------- 长期记忆 ----------
    def consolidate_memories(self, question: str, answer: str, source: str = "") -> int:
        """提炼并保存长期记忆，返回新增条数。"""
        memories = extract_long_term_memories(question, answer)
        if not memories:
            return 0
        return save_long_term_memories(self.db_path, memories, source=source)

    def load_memory_context(self, limit: int = 20) -> str:
        """加载长期记忆并组装为注入上下文文本。"""
        memories = load_long_term_memories(self.db_path, limit)
        return build_memory_context(memories)
