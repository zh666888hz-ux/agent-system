"""
agent/state.py
==============
ReAct Agent 的图状态定义。

设计原理：
    - LangGraph 的状态是一个「不可变快照 + Reducer 归约器」模型：
        每个节点返回的 dict 会通过对应字段的 Reducer 合并进状态。
    - messages 字段使用 add_messages Reducer：每轮新增的消息（AI 思考、
      工具调用请求、工具返回结果）会被**追加**到历史列表，天然累积出
      完整的「思考 → 行动 → 观察」轨迹，供模型多轮推理与最终作答使用。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """ReAct 图的状态。

    Attributes:
        messages: 对话消息序列（用户输入、AI 思考/工具调用、工具结果），
                  由 add_messages Reducer 自动追加。
    """

    messages: Annotated[list, add_messages]
