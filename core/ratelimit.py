"""
core/ratelimit.py
=================
令牌桶（Token Bucket）限流器。

设计原理：
    - 令牌桶算法：桶里按固定速率持续补充令牌（refill_rate 个/秒），容量上限 capacity。
      请求到来时若桶中有足够令牌则放行并扣减，否则阻塞等待（可设超时）或直接拒绝。
      相比固定窗口/滑动窗口，令牌桶允许「突发」（桶满时可一次消耗 capacity 个），
      又保证长期平均速率受限，非常适合 LLM API / HTTP 接口这种「偶发突发 + 总量控制」场景。
    - 线程安全：使用 threading.Lock 保护桶状态，支持多线程并发调用（FastAPI 多请求 /
      LangGraph 工具并行）下的安全限流。
    - 两个使用场景：
        * LLM 调用限流：RPM（次数桶）+ TPM（token 桶），控制对 OpenAI 兼容 API 的消耗；
        * HTTP 接口限流：按客户端 IP 各一个令牌桶，返回 429。
"""

from __future__ import annotations

import threading
import time

from core.logging import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """线程安全的令牌桶。

    Attributes:
        capacity: 桶容量（最大可突发令牌数）。
        refill_rate: 每秒补充的令牌数（长期平均速率）。
        tokens: 当前可用令牌数。
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        if capacity <= 0 or refill_rate <= 0:
            raise ValueError(f"capacity/refill_rate 必须为正: {capacity}/{refill_rate}")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = float(capacity)          # 初始满桶，允许首次突发
        self._last_refill = time.monotonic()    # 上次补充时间
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """按流逝时间补充令牌（补充量 = 间隔秒数 × 速率），不超过容量。"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

    def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        """尝试获取 tokens 个令牌。

        Args:
            tokens: 需要获取的令牌数（LLM 场景可传 token 数）。
            timeout: 最长等待秒数；None 表示无限等待，0 表示不等待（立即失败）。

        Returns:
            True 获取成功；False 在超时内仍未获取到（调用方决定拒绝/报错）。
        """
        if tokens <= 0:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                # 计算下一次能凑够令牌的最短等待时间
                shortage = tokens - self._tokens
                wait = shortage / self.refill_rate if self.refill_rate > 0 else float("inf")
            if deadline is not None:
                if time.monotonic() + wait > deadline:
                    return False  # 等不及了，拒绝
            # 释放锁后睡眠，避免忙等
            time.sleep(min(wait, 0.05))
            if deadline is not None and time.monotonic() >= deadline:
                return False

    @property
    def available(self) -> float:
        """当前可用令牌数（只读快照，供统计/展示）。"""
        with self._lock:
            self._refill()
            return self._tokens


class RateLimiter:
    """组合限流器：同时限「调用次数」与「token 消耗量」。

    - requests_per_minute: RPM 上限（次数令牌桶，rate = rpm/60 每秒）。
    - tokens_per_minute:   TPM 上限（token 令牌桶，rate = tpm/60 每秒）。
    - 调用前先请求「次数令牌 + 估算 token」；调用后把实际 token 消耗回填到统计
      （估算与实际有偏差，以实际为准持续校准桶状态）。

    实际 token 校准策略：请求前按估算扣减，避免超发；返回后若实际消耗大于估算，
    多出的部分从令牌桶补扣（下次请求会受影响），实现长期精确。
    """

    def __init__(
        self,
        requests_per_minute: float,
        tokens_per_minute: float,
        burst_seconds: float = 30.0,
    ) -> None:
        # 突发容量 = 每秒平均速率 × 突发窗口秒数。
        # 例：RPM=60 → 每秒 1 个请求，突发窗口 30s → 桶容量 30，
        # 允许一次性放行 30 个调用（应对并发突发），长期仍受 1 请求/秒限制。
        self._req_bucket = TokenBucket(
            capacity=max(requests_per_minute * burst_seconds / 60, 1),
            refill_rate=requests_per_minute / 60,
        )
        self._token_bucket = TokenBucket(
            capacity=max(tokens_per_minute * burst_seconds / 60, 1),
            refill_rate=tokens_per_minute / 60,
        )
        self._total_tokens_consumed = 0.0
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: float = 1.0, timeout: float | None = None) -> bool:
        """请求放行：先取次数令牌，再取 token 令牌。

        先次数后 token：若 token 不足等待超时，次数令牌已扣——为简单起见接受该偏差，
        实际场景中 token 桶通常是瓶颈，次数桶较少成为限制因素。
        """
        if not self._req_bucket.acquire(1.0, timeout):
            logger.warning("触发调用次数限流（RPM 超限）")
            return False
        if not self._token_bucket.acquire(max(estimated_tokens, 1.0), timeout):
            logger.warning("触发 token 限流（TPM 超限，估算 %d）", estimated_tokens)
            return False
        return True

    def settle(self, estimated_tokens: float, actual_tokens: float) -> None:
        """调用结束后回填实际 token 消耗，校准令牌桶。

        请求前按估算扣减；若实际消耗大于估算，差额（actual - estimated）从 token
        桶补扣，使长期统计精确；若实际小于估算，多扣的部分返还给桶（防过度限制）。
        """
        diff = actual_tokens - estimated_tokens
        with self._lock:
            self._total_tokens_consumed += actual_tokens
        if diff > 0:
            self._token_bucket.acquire(diff, timeout=0)  # 补扣超出部分（不阻塞）
        elif diff < 0:
            # 返还：桶里加回多扣的令牌（不超过容量，由 _refill 保证）
            self._token_bucket._tokens = min(
                self._token_bucket.capacity,
                self._token_bucket._tokens - diff,
            )
        logger.debug(
            "LLM 用量结算: 估算 %.0f / 实际 %.0f", estimated_tokens, actual_tokens
        )

    def stats(self) -> dict:
        """返回限流器状态（供健康检查/指标展示）。"""
        return {
            "rpm_bucket_available": round(self._req_bucket.available, 2),
            "tpm_bucket_available": round(self._token_bucket.available, 2),
            "total_tokens_consumed": round(self._total_tokens_consumed, 1),
        }
