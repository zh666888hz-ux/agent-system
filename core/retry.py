"""
core/retry.py
=============
通用「失败自动重试 + 指数退避」工具。

设计原理：
    - 工具调用失败（网络抖动 / 远端服务临时不可用 / 超时）在生产环境中很常见。
      一次失败不应立即放弃，而应按「指数退避」自动重试：间隔随尝试次数翻倍，
      既给远端恢复时间，又不至于在故障时高频轰炸服务。
    - 连续失败终止：达到重试上限后抛出 RetryExhaustedError，由上层决定如何收敛
      （本项目由 Agent 图捕获后进入 abort 节点，给出友好提示而非裸堆栈）。
    - 可观测性：每次失败都记录详细日志（工具名、第几次、失败原因、下次等待），
      满足「完整运行日志」要求。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from core.exceptions import RetryExhaustedError

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    description: str,
    retries: int,
    backoff: float = 1.0,
    logger: logging.Logger,
) -> T:
    """执行 fn 并在失败时按指数退避自动重试。

    Args:
        fn: 需要执行的零参数可调用对象（工具调用请用闭包包装）。
        description: 操作描述（如 "工具 calculator 调用"），用于日志。
        retries: 失败后重试次数（总尝试 = retries + 1）。
        backoff: 首次退避基数（秒），后续每次翻倍：backoff, 2*backoff, 4*backoff ...
        logger: 日志器（需传入，避免模块级单例导致无法按模块记录）。

    Returns:
        fn 的返回值。

    Raises:
        RetryExhaustedError: 连续失败超过 retries 次后抛出（含最后一次异常原因）。
    """
    if retries < 0:
        raise ValueError(f"retries 不能为负: {retries}")

    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):  # 1..retries+1，共 retries+1 次尝试
        try:
            result = fn()
            if attempt > 1:
                logger.info("%s：第 %d 次重试成功", description, attempt)
            return result
        except Exception as exc:  # noqa: BLE001 - 需捕获所有工具异常以判定重试
            last_exc = exc
            if attempt <= retries:
                delay = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "%s 失败（第 %d/%d 次尝试）：%s；%.1fs 后自动重试",
                    description,
                    attempt,
                    retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s 连续失败 %d 次，终止重试，最后一次错误：%s",
                    description,
                    retries + 1,
                    exc,
                )

    # 走到这里说明最后一次尝试仍失败 → 连续失败，终止任务
    raise RetryExhaustedError(
        f"{description} 连续失败 {retries + 1} 次后终止", cause=last_exc
    ) from last_exc
