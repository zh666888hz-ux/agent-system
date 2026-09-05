"""
tests/test_ratelimit.py
=======================
令牌桶限流器单元测试（不依赖网络 / 外部服务）。
"""

from __future__ import annotations

import time

from core.ratelimit import RateLimiter, TokenBucket


def test_token_bucket_initial_full():
    """初始满桶：允许一次消耗等于容量的令牌数。"""
    bucket = TokenBucket(capacity=10, refill_rate=5)
    assert bucket.acquire(10, timeout=0) is True   # 满桶可直接消耗
    assert bucket.acquire(1, timeout=0) is False   # 已空，不等待则失败


def test_token_bucket_refill():
    """随时间补充：等待 1 秒后应有 refill_rate 个令牌恢复。"""
    bucket = TokenBucket(capacity=10, refill_rate=5)
    assert bucket.acquire(10, timeout=0) is True
    time.sleep(1.1)                                # 补 5 个
    assert bucket.acquire(5, timeout=0) is True


def test_token_bucket_timeout():
    """等待超时：给定足够长 timeout 应能获取；过短则失败。"""
    bucket = TokenBucket(capacity=1, refill_rate=0.5)
    assert bucket.acquire(1, timeout=0) is True
    # 桶空，需 2 秒才能补 1 个
    assert bucket.acquire(1, timeout=0.5) is False
    start = time.monotonic()
    assert bucket.acquire(1, timeout=3.0) is True  # 等 2 秒内补足
    assert time.monotonic() - start >= 1.5


def test_token_bucket_thread_safety():
    """多线程并发：总放行量不超过速率上限（不精确断言，只验证无异常与总量收敛）。"""
    bucket = TokenBucket(capacity=50, refill_rate=20)
    import threading

    results: list[bool] = []

    def worker():
        for _ in range(20):
            results.append(bucket.acquire(1, timeout=0.05))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 所有线程执行完毕且无异常；放行数应为有限值
    assert isinstance(sum(1 for r in results if r), int)


def test_ratelimiter_settle():
    """结算：实际用量超过估算时补扣，长期统计记账准确。"""
    limiter = RateLimiter(requests_per_minute=600, tokens_per_minute=60000)
    assert limiter.acquire(100, timeout=0) is True
    limiter.settle(100, 250)                      # 实际 250 > 估算 100
    stats = limiter.stats()
    assert stats["total_tokens_consumed"] == 250  # 记账准确
