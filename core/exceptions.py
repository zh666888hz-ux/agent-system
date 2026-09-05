"""
core/exceptions.py
==================
统一异常体系。

设计原理：
    - 通过「基础异常类 + 细分子类」形成清晰的异常层级，调用方可按需捕获：
        1) 精确捕获某一子类做针对性处理；
        2) 捕获 AgentError 做统一兜底（对应 HTTP 层 / CLI 层的全局异常处理）。
    - 所有异常都携带 cause 参数，通过 `raise ... from exc` 保留底层异常链，
      便于生产环境排查根因（日志中会同时打印原始 Traceback）。
    - 工具执行失败统一包装为 ToolExecutionError，由 ReAct 图的 tools 节点捕获后
      转成 ToolMessage 反馈给模型，让模型「知道自己工具调用失败了」并重新规划，
      而不是让整个 Agent 直接崩溃——这是 ReAct 模式鲁棒性的关键设计。
"""

from __future__ import annotations


class AgentError(Exception):
    """Agent 应用的基础异常基类。"""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause


class ConfigurationError(AgentError):
    """配置错误：缺少必填配置 / 配置非法。"""


class LLMError(AgentError):
    """LLM 调用失败：网络错误、超时、鉴权失败、接口异常等。"""


class ToolExecutionError(AgentError):
    """工具执行失败：工具内部异常的统一包装。"""

    def __init__(self, message: str, tool_name: str = "", cause: Exception | None = None):
        super().__init__(message, cause)
        self.tool_name = tool_name


class SearchError(ToolExecutionError):
    """网络搜索失败：目标站点不可达、超时、返回异常等。"""


class DocumentReadError(ToolExecutionError):
    """文档读取/解析失败：文件不存在、编码异常、大小超限等。"""


class RetryExhaustedError(AgentError):
    """重试耗尽：工具/操作连续失败超过重试上限后抛出，用于终止任务并友好降级。"""

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        tool_name: str = "",
        cause: Exception | None = None,
    ):
        super().__init__(message, cause)
        self.attempts = attempts
        self.tool_name = tool_name
