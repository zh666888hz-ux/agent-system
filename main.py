"""
main.py
=======
ReAct Agent 的 CLI 入口。

用法：
    python main.py --question "计算 (1234*56+789)/3"
    python main.py                          # 进入交互式对话（输入 exit 退出）
    python main.py --question "..." --no-chain   # 不打印思考链

说明：本项目使用 from __future__ import annotations 与显式包结构，
运行时需在项目根目录执行（保证 config/core/tools/agent 可被导入）。
"""

from __future__ import annotations

import argparse
import sys

from agent.graph import run_agent
from config.settings import get_settings
from core.exceptions import AgentError
from core.logging import get_logger, setup_logging
from memory.repository import MemoryRepository

logger = get_logger("main")

# 交互式对话的退出关键词
_EXIT_KEYWORDS = {"exit", "quit", "退出", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="react-agent",
        description="LangGraph ReAct 模式智能体：内置 计算器 / 文档总结 / 网络搜索 三工具",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default="",
        help="单次提问内容；不提供则进入交互式对话",
    )
    parser.add_argument(
        "--no-chain",
        action="store_true",
        help="不打印思考链（默认打印每一步思考/工具调用过程）",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="会话 ID（短期记忆载体）。同一会话连续提问可保持对话上下文；不传则新建",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="列出历史会话并退出",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="覆盖日志级别，如 DEBUG / INFO / WARNING",
    )
    return parser.parse_args()


def print_result(result: dict) -> None:
    """格式化打印最终答案与工具调用统计。"""
    print("\n" + "=" * 60)
    print("🤖 最终答案")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print(f"本次共调用工具 {result['tool_calls']} 次，思考链 {len(result['chain'])} 步"
          + (f"，新增长期记忆 {result.get('memory_new', 0)} 条" if result.get("memory_new") else ""))
    if result.get("aborted"):
        print("⚠ 任务因工具连续失败而终止")
    print("=" * 60 + "\n")


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    settings = get_settings()

    # 记忆仓库：进程内复用一次（内部缓存数据库连接池与建表）
    repo: MemoryRepository | None = None
    if settings.memory_enabled:
        repo = MemoryRepository(settings.memory_db_path)

    try:
        # ---------- 列出历史会话 ----------
        if args.list_sessions:
            sessions = repo.list_sessions(20) if repo else []
            print(f"\n共 {len(sessions)} 个会话：")
            for s in sessions:
                print(f"  {s['id']}  |  {s['title'] or '(无标题)'}  |  {s['updated_at']}")
            return 0

        # ---------- 单次提问模式 ----------
        if args.question:
            result = run_agent(
                args.question,
                session_id=args.session,
                repository=repo,
                show_chain=not args.no_chain,
            )
            print_result(result)
            print(f"会话 ID: {result['session_id']}（下次提问加 --session 可续聊）")
            return 0

        # ---------- 交互式对话模式 ----------
        print("ReAct 智能体已启动（输入 exit / quit / 退出 结束对话）")
        print("试试：计算 (1234*56+789)/3  或  总结一段文本  或  搜索某话题\n")

        # 交互式：沿用同一会话，跨轮保持短期记忆
        session_id = args.session
        while True:
            try:
                question = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not question:
                continue
            if question.lower() in _EXIT_KEYWORDS:
                print("再见！")
                break
            try:
                result = run_agent(
                    question,
                    session_id=session_id,
                    repository=repo,
                    show_chain=not args.no_chain,
                )
                session_id = result["session_id"]  # 之后的轮次沿用此会话
                print_result(result)
            except AgentError as exc:
                logger.error("Agent 执行失败: %s", exc)
                print(f"\n[错误] {exc}\n")
    except AgentError as exc:
        logger.exception("启动或执行过程发生 Agent 异常")
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # 全局兜底：任何未预期异常不静默
        logger.exception("发生未预期异常")
        print(f"[未预期错误] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
