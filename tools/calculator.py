"""
tools/calculator.py
===================
工具①：安全计算器。

设计原理（安全评估的关键）：
    - 绝不使用内置 eval()：它可执行任意 Python 代码，是远程代码执行（RCE）漏洞温床。
    - 采用「AST 白名单」方案：
        1. 先用 ast.parse 将表达式解析为抽象语法树；
        2. 递归校验语法树中每个节点的类型是否在允许白名单内
           （仅允许数字、四则运算、幂、括号、以及少量安全函数）；
        3. 只对通过校验的树调用 eval(..., {"__builtins__": {}}) 求值——
           由于语法树已确认不含属性访问 / 下标 / 调用任意对象，因此安全。
    - 该方案同时保留真实计算的精度与能力，是生产环境安全表达式的标准做法。

支持的运算：
    四则 + - * /、幂 **、括号、一元正负、abs / min / max / round / pow。
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

from core.exceptions import ToolExecutionError
from core.logging import get_logger

logger = get_logger(__name__)

# 二元运算符 → 对应 operator 实现（白名单映射，不直接信任 AST 的 op）
_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 一元运算符 → 对应 operator 实现
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 白名单：允许的 AST 节点类型
# 注意：ast.iter_child_nodes 会遍历到「运算符节点」（如 ast.Mult），
# 因此运算符节点也必须纳入白名单，否则会被误判为非法语法。
_ALLOWED_NODES = {
    ast.Expression,
    ast.Constant,        # 数字、字符串字面量
    ast.BinOp,           # 二元运算（+ - * / ** 等）
    ast.UnaryOp,         # 一元运算（-x、+x）
    ast.Call,            # 函数调用（仅限下方安全函数白名单）
    ast.Name,            # 标识符（仅限下方安全函数名）
    ast.Load,            # 名称加载上下文
    # 注：Python 3.12+ 会为括号生成 ast.Parenthesized 节点；3.10/3.11 中
    # 括号在解析阶段即被消化、不产生额外节点，故无需（也不能）在此引用。
} | set(_BINARY_OPS) | set(_UNARY_OPS)

# 允许被调用的「安全函数」白名单（值即实现）
_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "pow": pow,
}


def _safe_eval(expression: str) -> Any:
    """对表达式进行 AST 白名单校验后安全求值。

    Args:
        expression: 数学表达式字符串，例如 "(1234*56+789)/3"。

    Returns:
        计算结果。

    Raises:
        ToolExecutionError: 表达式包含非法语法 / 非法节点 / 除零等异常时抛出。
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError(
            f"表达式语法错误: {exc.msg}", tool_name="calculator", cause=exc
        ) from exc

    # 校验函数：递归检查所有节点是否在白名单内
    def _check(node: ast.AST) -> None:
        if type(node) not in _ALLOWED_NODES:
            raise ToolExecutionError(
                f"表达式包含不允许的语法元素: {type(node).__name__}",
                tool_name="calculator",
            )
        if isinstance(node, ast.Call):
            # 函数调用：只允许白名单函数名
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise ToolExecutionError(
                    f"不允许调用函数: {getattr(node.func, 'id', '?')}",
                    tool_name="calculator",
                )
            # 函数名（Name 节点）不参与「变量」校验；仅递归校验参数
            for arg in node.args:
                _check(arg)
            for kw in node.keywords:
                _check(kw.value)
            return
        if isinstance(node, ast.BinOp):
            # 二元运算：运算符必须在白名单映射内
            if type(node.op) not in _BINARY_OPS:
                raise ToolExecutionError(
                    f"不支持的运算符: {type(node.op).__name__}",
                    tool_name="calculator",
                )
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ToolExecutionError(
                    f"不支持的一元运算符: {type(node.op).__name__}",
                    tool_name="calculator",
                )
        if isinstance(node, ast.Name):
            # 裸标识符（非调用场景）一律禁止，防止读取任何变量
            raise ToolExecutionError(
                f"不允许使用变量: {node.id}", tool_name="calculator"
            )
        # 递归遍历子节点
        for child in ast.iter_child_nodes(node):
            _check(child)

    try:
        _check(tree)
    except ToolExecutionError:
        raise  # 校验失败直接上抛
    except Exception as exc:  # 防御性兜底
        raise ToolExecutionError(
            f"表达式校验异常: {exc}", tool_name="calculator", cause=exc
        ) from exc

    # 通过校验后求值：清空 builtins 防隐式能力泄露，
    # 同时注入白名单安全函数（abs/min/max/round/pow），使它们可被调用。
    eval_globals = {"__builtins__": {}, **_SAFE_FUNCS}
    try:
        return eval(compile(tree, "<safe_expr>", "eval"), eval_globals, {})
    except ZeroDivisionError as exc:
        raise ToolExecutionError("除数不能为零", tool_name="calculator", cause=exc) from exc
    except Exception as exc:
        raise ToolExecutionError(
            f"表达式求值失败: {exc}", tool_name="calculator", cause=exc
        ) from exc


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式并返回数值结果。

    支持四则运算（+ - * /）、取余（%）、整除（//）、幂（**）、括号，
    以及安全函数 abs/min/max/round/pow。

    参数:
        expression: 待计算的数学表达式字符串，例如 "(1234*56+789)/3" 或 "2**10"。

    返回:
        计算结果（字符串形式）。
    """
    logger.info("calculator 开始计算: expression=%r", expression)
    # 基本参数校验：空输入 / 超长输入直接拒绝
    expression = (expression or "").strip()
    if not expression:
        raise ToolExecutionError("表达式不能为空", tool_name="calculator")
    if len(expression) > 200:
        raise ToolExecutionError("表达式过长（超过 200 字符）", tool_name="calculator")

    try:
        result = _safe_eval(expression)
    except ToolExecutionError:
        logger.exception("calculator 计算失败: %r", expression)
        raise

    logger.info("calculator 完成: %r = %r", expression, result)
    # 返回字符串（LangChain 工具约定返回 str 或可序列化对象）
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
