"""
core/logging.py
===============
统一日志配置模块。

设计原理：
    - 使用标准库 logging 的层级命名空间（logger 名即模块路径），天然区分不同模块的日志。
    - 同时输出到「控制台」与「滚动文件」两个 Handler：
        * 控制台：开发时实时观察；
        * RotatingFileHandler：生产环境日志持久化，按大小轮转，避免日志无限膨胀。
    - 提供 get_logger(name) 工厂函数，模块内统一通过它获取 logger，
      保证格式一致、避免重复初始化。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import get_settings

# 日志格式：时间 | 级别 | 模块 | 线程
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(threadName)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str | None = None, log_dir: Path | None = None) -> None:
    """初始化全局日志系统（幂等：重复调用不会叠加 Handler）。

    Args:
        level: 日志级别，默认读取配置（AGENT_LOG_LEVEL）。
        log_dir: 日志目录，默认读取配置（AGENT_LOG_DIR）。
    """
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = (level or settings.log_level).upper()
    log_dir = log_dir or settings.log_dir

    root = logging.getLogger()
    root.setLevel(level)

    # 清理历史 Handler，防止多次初始化导致日志重复打印
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # 1) 控制台 Handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # 2) 滚动文件 Handler（单文件 5MB，保留 5 个备份）
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "agent.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:  # 日志目录不可写时降级为仅控制台，不阻断主流程
        root.warning("日志文件不可用（%s），本次仅输出到控制台", exc)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """按模块路径获取 logger 实例。

    惯例：调用方传入 __name__，即得到与模块同名的 logger，
    例如 tools/calculator.py 的 logger 名为 "tools.calculator"。
    """
    return logging.getLogger(name)
