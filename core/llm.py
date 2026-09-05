"""
core/llm.py
===========
LLM 客户端工厂 + 调用计量（限流 / 耗时 / token 消耗）。

设计原理：
    - get_chat_model()：创建底层 ChatOpenAI 实例（OpenAI 兼容网关），
      保留给需要 bind_tools 的场景使用；lru_cache 缓存连接池。
    - chat_invoke()：所有 LLM 调用的统一入口，串起三层生产能力：
        1. 限流：调用前经 RateLimiter 获取「次数令牌 + 估算 token 令牌」，
           防止突发流量打爆上游 API（RPM / TPM 双桶）；
        2. 计量：记录每次调用的耗时、prompt/completion/total tokens，
           累计到 LLMMetrics（进程内），并逐次打印结构化日志；
        3. 结算：调用返回后按实际 usage 校准令牌桶（估算偏差自校正）。
    - token 估算（estimate_tokens）：请求前无法预知实际消耗，用「中文字符权重」
      粗略估算：中文按 ~1.7 字符/token、英文按 ~4 字符/token 折算，
      用于限流预扣；最终以 API 返回的实际 usage 为准。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from core.exceptions import LLMError, RateLimitError
from core.logging import get_logger
from core.ratelimit import RateLimiter

logger = get_logger(__name__)


# ---------- token 估算 ----------
def estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数（请求前无法预知实际用量时的预扣依据）。

    规则：中文字符按 ~1.7 字符/token、其余按 ~4 字符/token 折算。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.7 + other / 4))


def estimate_messages_tokens(messages: Sequence) -> int:
    """估算一组消息的总 token 数（含消息正文与简单指令开销）。"""
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):  # 多模态内容（罕见）
            content = " ".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
        total += estimate_tokens(str(content))
        total += 4  # 每条消息的协议开销近似
    return total


# ---------- 调用统计 ----------
@dataclass
class LLMUsage:
    """单次 LLM 调用的用量与耗时。"""

    caller: str = ""            # 调用方标识（agent / finalize / memory_extract）
    duration: float = 0.0       # 耗时（秒）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""             # 失败时记录错误摘要


@dataclass
class LLMMetrics:
    """进程内 LLM 调用统计（可导出给健康检查 / 监控）。"""

    calls: list[LLMUsage] = field(default_factory=list)   # 保留最近全部调用明细
    _by_caller: dict[str, dict] = field(default_factory=dict)

    def record(self, usage: LLMUsage) -> None:
        """记录一次成功调用。"""
        self.calls.append(usage)
        agg = self._by_caller.setdefault(
            usage.caller, {"count": 0, "total_duration": 0.0, "total_tokens": 0, "errors": 0}
        )
        agg["count"] += 1
        agg["total_duration"] += usage.duration
        agg["total_tokens"] += usage.total_tokens

    def record_error(self, caller: str, duration: float, error: str) -> None:
        """记录一次失败调用（计入错误与耗时，token 计 0）。"""
        self.calls.append(LLMUsage(caller=caller, duration=duration, error=error))
        agg = self._by_caller.setdefault(
            caller, {"count": 0, "total_duration": 0.0, "total_tokens": 0, "errors": 0}
        )
        agg["count"] += 1
        agg["total_duration"] += duration  # 失败也占用时长，如实计入
        agg["errors"] += 1

    def snapshot(self) -> dict:
        """汇总统计快照（供 /api/metrics 或健康检查）。"""
        total_calls = sum(a["count"] for a in self._by_caller.values())
        total_duration = sum(a["total_duration"] for a in self._by_caller.values())
        total_tokens = sum(a["total_tokens"] for a in self._by_caller.values())
        total_errors = sum(a["errors"] for a in self._by_caller.values())
        return {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "total_duration_sec": round(total_duration, 3),
            "total_tokens": total_tokens,
            "avg_duration_sec": round(total_duration / total_calls, 3) if total_calls else 0.0,
            "by_caller": self._by_caller,
        }


# ---------- 客户端与统一调用 ----------
@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """创建（并缓存）支持工具调用的 Chat 模型实例（供 bind_tools 使用）。"""
    settings = get_settings()
    logger.info(
        "初始化 LLM 客户端: model=%s, base_url=%s, timeout=%ss, retries=%d",
        settings.chat_model,
        settings.openai_base_url,
        settings.llm_timeout,
        settings.llm_max_retries,
    )
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


@lru_cache(maxsize=1)
def get_limiter() -> RateLimiter:
    """全局 LLM 限流器（RPM / TPM 双令牌桶）。"""
    settings = get_settings()
    limiter = RateLimiter(
        requests_per_minute=settings.llm_rate_limit_rpm,
        tokens_per_minute=settings.llm_rate_limit_tpm,
        burst_seconds=settings.llm_rate_limit_burst_seconds,
    )
    logger.info(
        "LLM 限流器就绪: RPM=%d, TPM=%d",
        settings.llm_rate_limit_rpm,
        settings.llm_rate_limit_tpm,
    )
    return limiter


@lru_cache(maxsize=1)
def get_metrics() -> LLMMetrics:
    """全局 LLM 调用统计实例。"""
    return LLMMetrics()


def _extract_usage(response) -> tuple[int, int, int]:
    """从模型响应中提取 token 用量（OpenAI 兼容格式）。

    response_metadata 中通常包含 token_usage: {prompt_tokens, completion_tokens, total_tokens}。
    """
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage", {}) or {}
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        int(usage.get("total_tokens", 0) or 0),
    )


def chat_invoke(
    messages: Sequence,
    *,
    bind_tools: list | None = None,
    caller: str = "llm",
) -> object:
    """带「限流 + 计量 + 日志」的 LLM 统一调用入口。

    Args:
        messages: 消息序列（langchain 消息或字符串）。
        bind_tools: 需要绑定的工具列表（ReAct agent 节点用）；None 表示不绑定。
        caller: 调用方标识，用于日志与统计分组。

    Returns:
        模型响应对象（AIMessage / 字符串）。

    Raises:
        RateLimitError: 触发限流且等待超时。
        LLMError: 调用失败（含原因链）。
    """
    settings = get_settings()
    model = get_chat_model()
    if bind_tools:
        model = model.bind_tools(bind_tools)

    # 1) 限流预扣（次数 + 估算 token）
    estimated = estimate_messages_tokens(messages)
    if not get_limiter().acquire(estimated, timeout=settings.llm_rate_limit_timeout):
        raise RateLimitError(
            f"LLM 调用触发限流（caller={caller}），等待超时，请稍后重试"
        )

    # 2) 调用并计时
    start = time.monotonic()
    try:
        response = model.invoke(messages)
    except Exception as exc:
        duration = time.monotonic() - start
        get_metrics().record_error(caller, duration, str(exc)[:200])
        logger.error("[LLM] %s 调用失败，耗时 %.2fs: %s", caller, duration, exc)
        raise LLMError(f"LLM 调用失败（{caller}）: {exc}", cause=exc) from exc

    duration = time.monotonic() - start

    # 3) 提取实际用量并结算
    p, c, t = _extract_usage(response)
    get_limiter().settle(estimated, t)
    get_metrics().record(
        LLMUsage(caller=caller, duration=duration, prompt_tokens=p, completion_tokens=c, total_tokens=t)
    )

    # 4) 完整日志：调用方、耗时、token 明细
    logger.info(
        "[LLM] %s 耗时 %.2fs | prompt=%d completion=%d total=%d",
        caller, duration, p, c, t,
    )
    return response
