"""
tools/base.py
=============
工具注册表。

设计原理：
    - 每个工具独立成模块（单一职责），通过统一的 get_tools() 入口汇总，
      让 Agent 图构建只依赖这一个接口，新增工具时无需改动图代码（开闭原则）。
    - 工具使用 langchain_core 的 @tool 装饰器定义：装饰器会自动从函数签名 +
      Docstring 生成 JSON Schema 并注入模型，模型据此决定何时调用、传什么参数。
      ⚠️ 因此 Docstring 必须写清「用途、参数含义、边界条件」，它直接影响工具调用准确率。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.calculator import calculator
from tools.document_summarizer import document_summarizer
from tools.web_search import web_search


def get_tools() -> list[BaseTool]:
    """返回全部已注册工具的列表（供 LangGraph 的 ToolNode 与模型绑定使用）。"""
    return [
        calculator,
        document_summarizer,
        web_search,
    ]
