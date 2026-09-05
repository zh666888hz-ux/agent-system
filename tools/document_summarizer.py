"""
tools/document_summarizer.py
============================
工具②：文档总结。

设计原理：
    - 支持两种输入：直接传入文本（text）或本地文件路径（file_path）。
    - 工具内部独立调用一次 LLM 生成摘要——注意这里使用**未绑定工具**的模型实例
      （get_chat_model()），避免与 Agent 主循环的模型递归绑定造成循环调用。
    - 生产级防御：
        * 参数二选一校验（text 与 file_path 同时给 / 都不给 → 报错）；
        * 文件大小限制（防止超大文件撑爆上下文）；
        * 编码自动探测（gbk / utf-8 兜底，兼容国内常见文档）；
        * 超长文本截断（保护 LLM 上下文窗口）；
        * 调用 LLM 失败时抛出可追踪的工具异常，由 ReAct 循环反馈给模型重规划。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from config.settings import get_settings
from core.exceptions import DocumentReadError, LLMError, ToolExecutionError
from core.llm import get_chat_model
from core.logging import get_logger

logger = get_logger(__name__)

_MAX_FILE_BYTES = 2 * 1024 * 1024  # 文件上限 2MB


def _read_text_from_file(file_path: str) -> str:
    """读取本地文本文件，自动探测常见编码。

    Raises:
        DocumentReadError: 文件不存在 / 超限 / 编码无法解析。
    """
    path = Path(file_path)
    if not path.is_file():
        raise DocumentReadError(
            f"文件不存在或不是普通文件: {file_path}", tool_name="document_summarizer"
        )
    size = path.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise DocumentReadError(
            f"文件过大（{size} 字节 > {_MAX_FILE_BYTES} 字节上限），"
            "请先拆分或只传入文本片段",
            tool_name="document_summarizer",
        )
    # 依次尝试常见编码，全部失败则报错
    for encoding in ("utf-8", "gbk", "gb18030", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentReadError(
        f"无法识别文件编码: {file_path}（已尝试 utf-8/gbk/gb18030/utf-16）",
        tool_name="document_summarizer",
    )


@tool
def document_summarizer(text: str | None = None, file_path: str | None = None) -> str:
    """总结一段文本或一个本地文本文件的内容，返回结构化摘要（要点 + 结论）。

    用于处理用户提供的长文、笔记、会议记录等，压缩信息量供后续回答使用。

    参数:
        text: 待总结的文本内容（与 file_path 二选一）。
        file_path: 本地文本文件路径（与 text 二选一）。

    返回:
        结构化中文摘要。
    """
    logger.info("document_summarizer 开始: text=%s, file_path=%s",
                (len(text) if text else None), file_path)

    # ---------- 参数校验：text 与 file_path 必须二选一 ----------
    if text and file_path:
        raise ToolExecutionError(
            "text 与 file_path 不能同时提供，请二选一",
            tool_name="document_summarizer",
        )
    if not text and not file_path:
        raise ToolExecutionError(
            "必须提供 text 或 file_path 之一", tool_name="document_summarizer"
        )

    # ---------- 获取原始内容 ----------
    if file_path:
        content = _read_text_from_file(file_path)
    else:
        content = text or ""
        content = content.strip()
        if not content:
            raise ToolExecutionError(
                "text 内容为空", tool_name="document_summarizer"
            )

    # ---------- 超长截断，保护上下文窗口 ----------
    settings = get_settings()
    if len(content) > settings.summary_max_chars:
        logger.warning("文本超长（%d 字符），截断到 %d 字符",
                       len(content), settings.summary_max_chars)
        content = content[: settings.summary_max_chars] + "……[已截断]"

    # ---------- 独立调用 LLM 生成摘要 ----------
    prompt = (
        "请对以下文本生成结构化中文摘要，要求：\n"
        "1. 用 3-6 条要点概括核心内容；\n"
        "2. 最后给出 1 句总体结论；\n"
        f"3. 摘要总字数控制在 {settings.summary_max_tokens} token 以内。\n\n"
        "--- 待总结文本开始 ---\n"
        f"{content}\n"
        "--- 待总结文本结束 ---"
    )
    try:
        model = get_chat_model()
        response = model.invoke(prompt)
        summary = response.content if isinstance(response.content, str) else str(response.content)
    except LLMError as exc:
        logger.exception("document_summarizer 调用 LLM 失败")
        raise ToolExecutionError(
            f"文档总结失败（LLM 调用异常）: {exc}", tool_name="document_summarizer", cause=exc
        ) from exc
    except Exception as exc:  # 防御性兜底：OpenAI SDK 内部异常统一包装
        logger.exception("document_summarizer 未知异常")
        raise ToolExecutionError(
            f"文档总结失败: {exc}", tool_name="document_summarizer", cause=exc
        ) from exc

    logger.info("document_summarizer 完成: 摘要长度=%d 字符", len(summary))
    return summary
