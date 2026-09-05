"""
memory/long_term.py
===================
长期记忆（跨会话持久记忆）。

设计原理：
    - 「长期记忆」沉淀的是跨会话仍有价值的持久信息：用户身份、偏好、关注领域、
      重要结论等。它不随会话结束而消失，在新会话开始时作为背景知识注入，
      让 Agent 在冷启动时也能"记得"用户是谁。
    - 提炼机制（memory consolidation）：
        每轮对话结束后，用 LLM 从「用户提问 + Agent 回答」中抽取值得长期记住的事实，
        输出结构化 JSON 数组。抽取时机放在对话完成后（而非每轮实时），
        既避免每轮都调 LLM 的额外开销，也保证提炼基于完整一轮上下文。
    - 注入机制：
        新会话开始 / 每轮开始时，读取最近 N 条长期记忆，拼成一段提示词追加到
        系统提示词之后（作为 SystemMessage），让模型在回答时自然参考。
    - 去重：写入前按内容精确匹配去重（同一事实只存一条），避免长期记忆膨胀。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from core.exceptions import LLMError
from core.llm import get_chat_model
from core.logging import get_logger
from memory.db import _connect, _now

logger = get_logger(__name__)

# 单次提炼 / 注入的上限，防止长期记忆表无限膨胀
_MAX_INJECT_LIMIT = 20

# 长期记忆提炼的系统提示词：要求模型只抽取「跨会话仍有价值」的事实，避免把
# 一次性闲聊内容写入长期记忆。输出必须为 JSON 数组。
_EXTRACT_PROMPT = (
    "你是一个记忆提炼助手。请从下面的「用户提问与助手回答」中，提炼出"
    "【值得跨会话长期记住的信息】，例如：用户的身份/职业/偏好/关注领域、"
    "明确的个人事实、反复出现的主题、需要长期遵循的要求。\n"
    "规则：\n"
    "1. 只提炼真正有长期价值的信息；普通闲聊、一次性问答内容不要收录。\n"
    "2. 每条记忆用一句通顺的中文短句表达，简洁明确。\n"
    "3. 如果没有任何值得长期记住的信息，返回空数组 []。\n"
    "4. 必须只输出 JSON 数组（不要任何解释文字），例如："
    '["用户是一名软件工程专业的学生，正在学习大模型应用开发", "用户关注 Java 与 Python 后端技术"]\n\n'
    "对话内容：\n{conversation}"
)


def extract_long_term_memories(question: str, answer: str) -> list[str]:
    """用 LLM 从一轮对话中提炼值得长期记住的事实。

    Args:
        question: 用户提问。
        answer: Agent 最终回答。

    Returns:
        提炼出的记忆字符串列表（可能为空）。
    """
    if not question or not answer:
        return []
    conversation = f"【用户提问】\n{question}\n\n【助手回答】\n{answer[:2000]}"
    prompt = _EXTRACT_PROMPT.format(conversation=conversation)

    try:
        response = get_chat_model().invoke(prompt)
        content = str(response.content).strip()
    except Exception as exc:
        # 记忆提炼失败不影响主流程：记录后跳过，避免记忆功能拖垮对话
        logger.warning("长期记忆提炼失败（已跳过）: %s", exc)
        return []

    memories = _parse_json_array(content)
    if memories:
        logger.info("提炼到 %d 条长期记忆", len(memories))
    return memories


def _parse_json_array(text: str) -> list[str]:
    """容错解析模型输出的 JSON 数组（容忍被 ```json 包裹等情况）。"""
    text = text.strip()
    # 去掉可能的 markdown 代码块围栏
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        logger.warning("长期记忆 JSON 解析失败，原始内容: %s", text[:200])
    return []


def save_long_term_memories(
    db_path: Path, memories: list[str], source: str = "", user_id: str = "default"
) -> int:
    """保存长期记忆（按内容去重，返回实际新增条数）。"""
    if not memories:
        return 0
    now = _now()
    inserted = 0
    with _connect(db_path) as conn:
        for mem in memories:
            # 精确去重：同内容已存在则跳过（不重复沉淀）
            exists = conn.execute(
                "SELECT 1 FROM long_term_memories WHERE user_id = ? AND content = ?",
                (user_id, mem),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                "INSERT INTO long_term_memories (user_id, content, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, mem, source, now, now),
            )
            inserted += 1
    if inserted:
        logger.info("已写入 %d 条新长期记忆", inserted)
    return inserted


def load_long_term_memories(
    db_path: Path, limit: int = _MAX_INJECT_LIMIT, user_id: str = "default"
) -> list[str]:
    """加载最近的长期记忆（最新在前），用于注入 Agent 上下文。"""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT content FROM long_term_memories WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [r["content"] for r in rows]


def build_memory_context(memories: list[str]) -> str:
    """把长期记忆列表拼成一段提示词文本（注入系统提示词用）。"""
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return (
        "【长期记忆：关于用户的一些持久信息，请在回答时自然参考，"
        "但不要刻意提及这些内容来自记忆】\n"
        + lines
    )
