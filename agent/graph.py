"""
agent/graph.py
==============
LangGraph ReAct Agent 图的构建与运行。

设计原理（ReAct 图结构）：
    - 本图是一个带条件回边的状态机：
        START → agent → tools → agent → tools → ... → agent → END

        节点说明：
        * agent  节点：把「系统提示词 + 全部历史消息」交给绑定了工具的 LLM，
                  模型要么输出最终回答（无 tool_calls），要么输出工具调用请求。
        * tools  节点：执行模型请求的工具调用，把执行结果以 ToolMessage 回填状态。
        * 条件边：agent 的输出若含 tool_calls → 走 tools；否则 → 走 END。

    - 为什么用「图 + 循环」而非单次调用？
        复杂问题往往需要多轮工具调用（先查资料 → 再计算 → 再总结），
        图结构天然支持这种迭代，且每一轮的轨迹都保留在状态里，可审计。

    - 思考过程的可观测性：
        通过 graph.stream(..., stream_mode="updates") 逐节点拿到增量输出，
        把每一步「模型思考 / 选择的工具 / 工具结果」记录为结构化 chain，
        满足「记录每一步思考过程」的需求。

    - 异常捕获与兜底：
        * 工具执行失败 → ToolNode 把异常包装为 ToolMessage 回传模型，Agent 继续规划；
        * 图整体执行异常 → 在此统一捕获并抛出 LLMError/AgentError，由上层（CLI）兜底。

    - 防无限循环（本模块核心安全设计）：
        * 状态中维护 tool_calls 计数器，每执行一轮工具调用就自增；
        * 条件边在「模型又请求调用工具」时先检查计数器是否已达 AGENT_MAX_ITERATIONS 上限，
          达到则进入 finalize 节点——用「未绑定工具」的模型追加提示后生成最终答案，
          从根本上杜绝模型反复调用工具造成的死循环（而非依赖框架层 recursion_limit 兜底）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from config.settings import get_settings
from core.exceptions import LLMError
from core.llm import get_chat_model
from core.logging import get_logger
from tools.base import get_tools

logger = get_logger(__name__)

# 工具列表：模块加载时解析一次（不依赖 API Key）
_tools = get_tools()

# 达到工具调用上限时，追加给模型的提示：强制收敛、不再调用工具
_LIMIT_PROMPT = (
    "注意：本次会话已到达工具调用次数上限，请【立即停止调用任何工具】。"
    "仅基于已有的工具返回结果和对话历史，直接给出最终答案；"
    "若信息不足，请如实说明当前已获得的信息与局限。"
)


@lru_cache(maxsize=1)
def _get_bound_model():
    """懒加载并缓存「绑定工具」的模型实例。

    设计原因：绑定工具需要读取配置（含 API Key）。若在模块导入时绑定，
    则未配置 Key 时连 `--help` 都会启动失败。改为首次调用时才绑定，
    使 CLI 帮助、参数校验等场景无需 Key 也能运行（fail-late 而非 fail-always）。
    """
    return get_chat_model().bind_tools(_tools)


def _agent_node(state: AgentState) -> dict[str, list]:
    """agent 节点：调用绑定工具的 LLM 生成「回答 或 工具调用请求」。

    Args:
        state: 当前图状态（含全部历史消息）。

    Returns:
        {"messages": [AI 消息]}，由 add_messages Reducer 追加到状态。
    """
    # 仅当这是第一轮时才注入系统提示词（避免历史消息中重复出现系统提示）
    if not any(isinstance(m, SystemMessage) for m in state["messages"]):
        state = {**state, "messages": [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]}

    try:
        response = _get_bound_model().invoke(state["messages"])
    except Exception as exc:
        logger.exception("agent 节点调用 LLM 失败")
        raise LLMError(f"LLM 调用失败: {exc}", cause=exc) from exc

    # ---------- 记录思考过程 ----------
    if response.content:  # 模型产出的推理文本
        logger.info("[思考] %s", str(response.content).replace("\n", " "))
    for tc in getattr(response, "tool_calls", []) or []:
        logger.info("[选择工具] %s, 参数=%s", tc["name"], tc.get("args"))
    if not (getattr(response, "tool_calls", []) or []):
        logger.info("[给出答案] %s", str(response.content).replace("\n", " ")[:200])

    return {"messages": [response]}


def _tools_node(state: AgentState) -> dict[str, Any]:
    """tools 节点：执行模型请求的所有工具调用，结果以 ToolMessage 回填，并自增计数器。

    Returns:
        {"messages": [ToolMessage...], "tool_calls": 递增后的累计次数}。
    """
    last: AIMessage = state["messages"][-1]
    calls = getattr(last, "tool_calls", []) or []
    for tc in calls:
        logger.info("→ 正在执行工具: %s%s", tc["name"], tc.get("args"))

    result = _tool_node.invoke(state)
    new_count = state.get("tool_calls", 0) + len(calls)
    logger.info("工具调用累计次数: %d", new_count)
    return {"messages": result.get("messages", []), "tool_calls": new_count}


def _finalize_node(state: AgentState) -> dict[str, list]:
    """finalize 节点：达到工具调用上限后强制收敛。

    使用「未绑定工具」的模型（无法再发起工具调用），追加 _LIMIT_PROMPT 提示后，
    让模型仅基于已有工具结果生成最终答案——确保即使模型此前一直想调用工具，
    也能在上限处停止并给出结论，不会无限循环。

    注意：达到上限时，最后一条 AI 消息可能带 tool_calls，但这些调用「并未执行」
    （被条件边拦截），消息序列中不存在对应 ToolMessage。直接将其传给 LLM 会违反
    「带 tool_calls 的 assistant 消息必须紧跟对应 tool 消息」的协议约束（400 错误），
    因此必须先清理这条「孤儿」tool_calls 消息，仅保留其思考文本。
    """
    limit = get_settings().max_iterations
    logger.warning("已达到工具调用上限 %d 次，强制进入最终作答（不再允许调用工具）", limit)

    msgs = list(state["messages"])
    last = msgs[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        logger.info("清理未执行的孤儿 tool_calls（%d 个）", len(last.tool_calls))
        # 重建一条不含 tool_calls 的 AI 消息，保留其思考文本，保证消息协议合法
        msgs[-1] = AIMessage(content=last.content or "", name=last.name)

    try:
        response = get_chat_model().invoke(
            [*msgs, SystemMessage(content=_LIMIT_PROMPT)]
        )
    except Exception as exc:
        logger.exception("finalize 节点调用 LLM 失败")
        raise LLMError(f"LLM 调用失败（finalize）: {exc}", cause=exc) from exc
    logger.info("[最终作答(上限触发)] %s", str(response.content).replace("\n", " ")[:200])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    """条件边路由：决定 agent 之后的去向。

    路由优先级：
        1. 模型没有工具调用请求 → "end"（正常给出最终答案）；
        2. 工具调用累计次数已达上限 → "limit"（强制收敛，防无限循环）；
        3. 否则 → "tools"（继续执行工具调用）。

    Returns:
        "tools" / "limit" / "end"。
    """
    last: AIMessage = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "end"
    if state.get("tool_calls", 0) >= get_settings().max_iterations:
        logger.warning("检测到工具调用已达上限 %d 次，强制终止循环", get_settings().max_iterations)
        return "limit"
    return "tools"


# 工具执行节点：模块加载时构建一次（不依赖 API Key）
_tool_node = ToolNode(_tools)


def build_agent():
    """构建并编译 ReAct Agent 图。

    图结构：
        START → agent ─(tools)→ tools → agent → ... ─(limit)→ finalize → END
                          └──────(end)─────────── END

    Returns:
        langchain.graph.state.CompiledStateGraph: 可被 invoke/stream 的编译图。
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.add_node("finalize", _finalize_node)

    # 连边：START → agent；agent 条件路由 tools/limit/end；tools 循环回 agent
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "limit": "finalize", "end": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(question: str, show_chain: bool = True) -> dict[str, Any]:
    """执行一次 Agent 问答，返回最终答案与完整思考链。

    Args:
        question: 用户问题。
        show_chain: 是否在日志中打印完整思考链（默认 True）。

    Returns:
        {
            "answer": 最终回答文本,
            "chain":  思考链列表，每项为 {"node": str, "summary": str},
            "tool_calls": 实际调用的工具次数统计,
        }

    Raises:
        AgentError / LLMError: 图执行失败。
    """
    settings = get_settings()
    question = (question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    graph = build_agent()
    chain: list[dict[str, str]] = []
    tool_call_count = 0
    answer = ""

    logger.info("=" * 60)
    logger.info("开始处理问题: %s", question)
    logger.info("=" * 60)

    # 通过 stream 逐节点消费，记录每一步思考过程
    # tool_calls=0：工具调用计数器初始值（防无限循环）
    # recursion_limit：需容纳「每轮 agent+tools ≈ 2 个 superstep」+ 末尾 finalize/END 收尾，
    #   取 max_iterations*3 + 5，确保达到上限后 finalize 分支能完整走完而不被框架层拦截。
    for step in graph.stream(
        {"messages": [HumanMessage(content=question)], "tool_calls": 0},
        config={"recursion_limit": settings.max_iterations * 3 + 5},
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            # 统计工具调用
            for msg in update.get("messages", []) or []:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    tool_call_count += len(msg.tool_calls)
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content:
                    answer = str(msg.content)  # 无工具调用的 AI 消息即最终答案

            summary = _summarize_update(node_name, update)
            if summary:
                chain.append({"node": node_name, "summary": summary})
                if show_chain:
                    logger.info("[%s] %s", node_name, summary)

    return {
        "answer": answer,
        "chain": chain,
        "tool_calls": tool_call_count,
    }


def _summarize_update(node_name: str, update: dict[str, Any]) -> str:
    """把某节点的一次输出压缩为一行摘要，供思考链展示。"""
    messages = update.get("messages", []) or []
    if not messages:
        return ""
    last = messages[-1]

    if node_name == "agent":
        if getattr(last, "tool_calls", None):
            calls = ", ".join(tc["name"] for tc in last.tool_calls)
            return f"思考后决定调用工具: {calls}"
        content = str(last.content or "")
        return f"生成回答（{len(content)} 字符）" if content else "（模型无文本输出）"

    if node_name == "tools":
        # ToolMessage：展示工具名与返回摘要
        tool_name = getattr(last, "name", "?")
        content = str(last.content or "")[:120]
        return f"工具 {tool_name} 返回: {content}"

    if node_name == "finalize":
        content = str(last.content or "")
        return f"已达工具调用上限，强制最终作答（{len(content)} 字符）"

    return f"{node_name} 节点输出"
