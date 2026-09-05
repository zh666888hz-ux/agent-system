"""
core/llm.py
===========
LLM 客户端工厂。

设计原理：
    - 通过 ChatOpenAI 对接任意 OpenAI 兼容网关（base_url 可指向 DeepSeek / OpenAI /
      通义 / vLLM / OneAPI 等），上层 Agent 无需关心具体厂商差异。
    - 关键工程参数集中在此配置：
        * timeout：单次请求超时，避免外部服务无响应时进程无限挂起；
        * max_retries：网络抖动自动重试（OpenAI SDK 内置指数退避）；
        * temperature=0：工具调用场景要求确定性，减少模型随机发挥。
    - 使用 lru_cache 缓存客户端实例：避免多次构建重复创建连接池。
    - 模型支持「工具调用」（function calling / tool calling），这是 ReAct 的基石：
      Agent 节点通过 bind_tools 把工具 schema 注入 LLM，模型才能输出结构化工具调用。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.settings import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """创建（并缓存）支持工具调用的 Chat 模型实例。

    Returns:
        ChatOpenAI: 绑定工具前的模型实例；工具绑定在 Agent 图构建阶段完成。
    """
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
