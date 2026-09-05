"""
agent/graph.py
==============
LangGraph ReAct Agent 图的构建与运行。

设计原理（ReAct 图结构）：
    - 本图是一个带条件回边的状态机：
        START → agent → tools → agent → tools → ... → agent → END

        节点说明：
        * agent   节点：把「系统提示词 + 长期记忆 + 历史消息」交给绑定了工具的 LLM，
                  模型要么输出最终回答（无 tool_calls），要么输出工具调用请求。
        * tools   节点：执行模型请求的工具调用；失败自动重试，连续失败则写入 abort_reason
                  并路由到 abort 节点，给出友好提示而非让 Agent 无限重试。
        * finalize 节点：工具调用次数达上限时，用「未绑定工具」的模型强制收敛作答。
        * abort    节点：工具连续失败时，生成确定性友好提示（不依赖 LLM，保证可靠）。

    - 为什么用「图 + 循环」而非单次调用？
        复杂问题往往需要多轮工具调用（先查资料 → 再计算 → 再总结），
        图结构天然支持这种迭代，且每一轮的轨迹都保留在状态里，可审计。

    - 思考过程的可观测性：
        通过 graph.stream(..., stream_mode="updates") 逐节点拿到增量输出，
        把每一步「模型思考 / 选择的工具 / 工具结果」记录为结构化 chain。

    - 异常捕获与兜底：
        * 工具执行失败 → 自动重试（指数退避）；连续失败 → abort 节点友好提示；
        * 图整体执行异常 → 在此统一捕获并抛出 LLMError/AgentError，由上层（CLI）兜底。

    - 防无限循环（双重保障）：
        * 图内计数器：state.tool_calls 每轮自增，达 AGENT_MAX_ITERATIONS 上限 → finalize；
        * 框架层 recursion_limit 动态放大兜底。

    - 记忆系统集成：
        * 短期记忆：run_agent 加载指定会话的历史 Q/A 对（HumanMessage/AIMessage），
          拼接进初始状态，实现跨轮上下文；
        * 长期记忆：加载后作为 SystemMessage 追加到系统提示词，让 Agent 记得用户背景；
        * 对话结束后：保存本轮 Q/A 到短期记忆，并用 LLM 提炼可长期记住的事实入库。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from config.settings import get_settings
from core.exceptions import LLMError, RetryExhaustedError
from core.llm import chat_invoke
from core.logging import get_logger
from core.retry import retry_call
from memory.repository import MemoryRepository
from tools.base import get_tools

logger = get_logger(__name__)

# 工具列表与映射：模块加载时解析一次（不依赖 API Key）
_tools = get_tools()
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _tools}

# 达到工具调用上限时，追加给模型的提示：强制收敛、不再调用工具
_LIMIT_PROMPT = (
    "注意：本次会话已到达工具调用次数上限，请【立即停止调用任何工具】。"
    "仅基于已有的工具返回结果和对话历史，直接给出最终答案；"
    "若信息不足，请如实说明当前已获得的信息与局限。"
)


def _agent_node(state: AgentState) -> dict[str, list]:
    """agent 节点：调用绑定工具的 LLM 生成「回答 或 工具调用请求」。

    - 首次进入时注入系统提示词 + 长期记忆上下文（只注入一次，避免历史中重复）。
    - 统一走 chat_invoke（内部自动限流 + 记录耗时/token + 完整日志）。
    """
    if not any(isinstance(m, SystemMessage) for m in state["messages"]):
        system_content = SYSTEM_PROMPT
        memory_ctx = state.get("memory_context", "")  # 长期记忆注入
        if memory_ctx:
            system_content = f"{system_content}\n\n{memory_ctx}"
        state = {**state, "messages": [SystemMessage(content=system_content), *state["messages"]]}

    try:
        response = chat_invoke(state["messages"], bind_tools=_tools, caller="agent")
    except Exception as exc:
        logger.exception("agent 节点调用 LLM 失败")
        raise LLMError(f"LLM 调用失败: {exc}", cause=exc) from exc

    # ---------- 记录思考过程 ----------
    if response.content:
        logger.info("[思考] %s", str(response.content).replace("\n", " "))
    for tc in getattr(response, "tool_calls", []) or []:
        logger.info("[选择工具] %s, 参数=%s", tc["name"], tc.get("args"))
    if not (getattr(response, "tool_calls", []) or []):
        logger.info("[给出答案] %s", str(response.content).replace("\n", " ")[:200])

    return {"messages": [response]}


def _execute_one_tool(name: str, args: dict) -> str:
    """执行单个工具调用，带「失败自动重试 + 指数退避」。

    用 retry_call 包装：工具失败后按 AGENT_TOOL_MAX_RETRIES 重试，
    每次等待时间指数翻倍；连续失败则抛出 RetryExhaustedError 交给上层终止。
    """
    settings = get_settings()
    tool = _TOOL_MAP.get(name)
    if tool is None:
        # 模型请求了不存在的工具（幻觉）：直接返回错误信息，让模型重新规划
        raise ValueError(f"未知工具: {name!r}，可用工具: {sorted(_TOOL_MAP)}")

    # 闭包包装工具调用，供 retry_call 零参数调用
    def _call() -> str:
        result = tool.invoke(args)
        return str(result) if result is not None else "执行完成（无返回值）"

    logger.info("→ 执行工具 %s，参数=%s", name, args)
    started_at = __import__("time").time()
    result = retry_call(
        _call,
        description=f"工具 {name}",
        retries=settings.tool_max_retries,
        backoff=settings.tool_retry_backoff,
        logger=logger,
    )
    elapsed = __import__("time").time() - started_at
    logger.info("工具 %s 成功，耗时 %.2fs", name, elapsed)
    return result


def _tools_node(state: AgentState) -> dict[str, Any]:
    """tools 节点：执行模型请求的所有工具调用（逐个失败重试）。

    - 全部成功：返回 ToolMessage 列表 + 自增计数器 → 条件边回到 agent。
    - 任一工具连续失败：写入 abort_reason → 条件边路由到 abort 节点（终止任务）。
    """
    last: AIMessage = state["messages"][-1]
    calls = getattr(last, "tool_calls", []) or []
    settings = get_settings()

    tool_messages: list[ToolMessage] = []
    abort_reason = ""
    for tc in calls:
        name = tc["name"]
        args = tc.get("args") or {}
        try:
            content = _execute_one_tool(name, args)
        except RetryExhaustedError as exc:
            # 连续失败：终止任务，记录原因（不再把错误丢给模型继续试）
            logger.error("工具 %s 连续失败，终止任务: %s", name, exc)
            abort_reason = f"工具「{name}」连续失败 {settings.tool_max_retries + 1} 次，任务终止"
            tool_messages.append(
                ToolMessage(content=str(exc), tool_call_id=tc["id"], name=name)
            )
            break
        except Exception as exc:  # noqa: BLE001 - 未知工具等单次错误，回传模型重新规划
            logger.warning("工具 %s 调用异常（单次，不终止）: %s", name, exc)
            tool_messages.append(
                ToolMessage(content=f"工具调用失败: {exc}", tool_call_id=tc["id"], name=name)
            )
            continue
        tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"], name=name))

    new_count = state.get("tool_calls", 0) + len(calls)
    logger.info("工具调用累计次数: %d", new_count)

    update: dict[str, Any] = {"messages": tool_messages, "tool_calls": new_count}
    if abort_reason:
        update["abort_reason"] = abort_reason
    return update


def _finalize_node(state: AgentState) -> dict[str, list]:
    """finalize 节点：达到工具调用上限后强制收敛。

    使用「未绑定工具」的模型（无法再发起工具调用），追加 _LIMIT_PROMPT 提示后，
    让模型仅基于已有工具结果生成最终答案，不会无限循环。

    注意：达到上限时，最后一条 AI 消息可能带 tool_calls，但这些调用「并未执行」
    （被条件边拦截），消息序列中不存在对应 ToolMessage。直接传给 LLM 会违反
    「带 tool_calls 的 assistant 消息必须紧跟对应 tool 消息」的协议约束（400 错误），
    因此必须先清理这条「孤儿」tool_calls 消息，仅保留其思考文本。
    """
    limit = get_settings().max_iterations
    logger.warning("已达到工具调用上限 %d 次，强制进入最终作答（不再允许调用工具）", limit)

    msgs = list(state["messages"])
    last = msgs[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        logger.info("清理未执行的孤儿 tool_calls（%d 个）", len(last.tool_calls))
        msgs[-1] = AIMessage(content=last.content or "", name=last.name)

    try:
        response = chat_invoke(
            [*msgs, SystemMessage(content=_LIMIT_PROMPT)], caller="finalize"
        )
    except Exception as exc:
        logger.exception("finalize 节点调用 LLM 失败")
        raise LLMError(f"LLM 调用失败（finalize）: {exc}", cause=exc) from exc
    logger.info("[最终作答(上限触发)] %s", str(response.content).replace("\n", " ")[:200])
    return {"messages": [response]}


def _abort_node(state: AgentState) -> dict[str, list]:
    """abort 节点：工具连续失败后终止任务，生成友好的确定性提示。

    设计要点：
        - 不调用 LLM（避免二次失败），直接基于 abort_reason 生成结构化文案，
          保证无论何种情况下终止提示都稳定可靠。
        - 提示包含：失败工具名、已尝试次数、可能原因、用户可操作的建议。
    """
    reason = state.get("abort_reason", "工具执行遇到连续失败")
    logger.error("== 任务终止 ==")
    logger.error("终止原因: %s", reason)

    friendly = (
        f"抱歉，任务未能完成：{reason}。\n\n"
        "可能原因：\n"
        "- 网络暂时不可用或服务端超时（已按策略自动重试多次仍失败）；\n"
        "- 目标站点 / 服务临时故障或拒绝访问。\n\n"
        "建议：\n"
        "- 稍后重试一次；\n"
        "- 换个说法描述你的需求；\n"
        "- 若持续失败，可联系管理员检查网络与服务配置。"
    )
    return {"messages": [AIMessage(content=friendly)]}


def _should_continue(state: AgentState) -> str:
    """agent 节点之后的去向。

    路由优先级：
        1. 模型没有工具调用请求 → "end"（正常给出最终答案）；
        2. 工具调用累计次数已达上限 → "limit"（强制收敛，防无限循环）；
        3. 否则 → "tools"（继续执行工具调用）。
    """
    last: AIMessage = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "end"
    if state.get("tool_calls", 0) >= get_settings().max_iterations:
        logger.warning("检测到工具调用已达上限 %d 次，强制终止循环", get_settings().max_iterations)
        return "limit"
    return "tools"


def _after_tools(state: AgentState) -> str:
    """tools 节点之后的去向：工具连续失败 → abort；否则继续 agent。"""
    if state.get("abort_reason"):
        return "abort"
    return "agent"


def build_agent():
    """构建并编译 ReAct Agent 图。

    图结构：
        START → agent ─(tools)→ tools ─(abort)→ abort → END
                          │          │
                          │          └──(agent)→ agent
                          └──(limit)→ finalize → END
                          └──(end)────────────→ END

    Returns:
        langchain.graph.state.CompiledStateGraph: 可被 invoke/stream 的编译图。
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.add_node("finalize", _finalize_node)
    graph.add_node("abort", _abort_node)

    # 连边
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "limit": "finalize", "end": END},
    )
    graph.add_conditional_edges(
        "tools",
        _after_tools,
        {"agent": "agent", "abort": "abort"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("abort", END)

    return graph.compile()


def run_agent(
    question: str,
    session_id: str | None = None,
    repository: MemoryRepository | None = None,
    show_chain: bool = True,
) -> dict[str, Any]:
    """执行一次 Agent 问答，返回最终答案与完整思考链，并完成记忆读写。

    Args:
        question: 用户问题。
        session_id: 会话 ID（短期记忆载体）。None 时自动新建会话。
        repository: 记忆仓库（可复用）。None 且 memory_enabled 时自动创建。
        show_chain: 是否在日志中打印完整思考链（默认 True）。

    Returns:
        {
            "answer": 最终回答文本,
            "chain":  思考链列表,
            "tool_calls": 实际调用的工具次数统计,
            "session_id": 会话 ID,
            "memory_new": 本次新增长期记忆条数,
            "aborted": 是否因工具连续失败而终止,
        }

    Raises:
        AgentError / LLMError: 图执行失败。
    """
    settings = get_settings()
    question = (question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    # ---------- 记忆初始化 ----------
    repo = repository
    memory_on = settings.memory_enabled
    if memory_on and repo is None:
        repo = MemoryRepository(settings.memory_db_path)
    conv_id = session_id
    if memory_on and repo is not None:
        conv_id = repo.new_session(session_id)
        history = repo.load_session_history(conv_id)          # 短期记忆：历史 Q/A
        memory_context = repo.load_memory_context(settings.memory_inject_limit)  # 长期记忆
        logger.info("会话 %s 启动：历史 %d 条，长期记忆注入 %d 字符",
                    conv_id, len(history), len(memory_context))
    else:
        history, memory_context = [], ""
        logger.info("记忆未启用，使用一次性无记忆会话")

    # ---------- 构建图并执行 ----------
    graph = build_agent()
    chain: list[dict[str, str]] = []
    tool_call_count = 0
    answer = ""
    aborted = False

    logger.info("=" * 60)
    logger.info("开始处理问题: %s", question)
    logger.info("=" * 60)

    initial_messages: list = [HumanMessage(content=question)]
    if history:
        # 短期历史在前、当前问题在后，保持时间正序
        initial_messages = [*history, HumanMessage(content=question)]

    # tool_calls=0：防无限循环计数器；recursion_limit 需容纳每轮 2 superstep + 收尾
    for step in graph.stream(
        {
            "messages": initial_messages,
            "tool_calls": 0,
            "abort_reason": "",
            "memory_context": memory_context,
        },
        config={"recursion_limit": settings.max_iterations * 3 + 5},
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            if node_name == "abort":
                aborted = True
            for msg in update.get("messages", []) or []:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    tool_call_count += len(msg.tool_calls)
                if (
                    isinstance(msg, AIMessage)
                    and not getattr(msg, "tool_calls", None)
                    and msg.content
                ):
                    answer = str(msg.content)  # 无工具调用的 AI 消息即最终答案

            summary = _summarize_update(node_name, update)
            if summary:
                chain.append({"node": node_name, "summary": summary})
                if show_chain:
                    logger.info("[%s] %s", node_name, summary)

    # ---------- 对话结束后：写入记忆 ----------
    memory_new = 0
    if memory_on and repo is not None and answer:
        # 1) 短期记忆：保存本轮 Q/A
        repo.save_qa(conv_id, question, answer, meta={"tool_calls": tool_call_count})
        # 2) 长期记忆：LLM 提炼值得长期记住的事实
        if settings.memory_extract:
            memory_new = repo.consolidate_memories(question, answer, source=conv_id)

    logger.info("=" * 60)
    logger.info("会话 %s 本轮完成：工具调用 %d 次，思考链 %d 步，新增长期记忆 %d 条",
                conv_id, tool_call_count, len(chain), memory_new)
    logger.info("=" * 60)

    return {
        "answer": answer,
        "chain": chain,
        "tool_calls": tool_call_count,
        "session_id": conv_id,
        "memory_new": memory_new,
        "aborted": aborted,
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
        tool_name = getattr(last, "name", "?")
        content = str(last.content or "")[:120]
        return f"工具 {tool_name} 返回: {content}"

    if node_name == "finalize":
        content = str(last.content or "")
        return f"已达工具调用上限，强制最终作答（{len(content)} 字符）"

    if node_name == "abort":
        return "工具连续失败，任务终止（已生成友好提示）"

    return f"{node_name} 节点输出"
