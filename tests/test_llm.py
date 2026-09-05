"""
tests/test_llm.py
=================
LLM 计量模块单元测试：token 估算 / 用量统计（不依赖网络）。
"""

from __future__ import annotations

from core.llm import LLMMetrics, LLMUsage, estimate_messages_tokens, estimate_tokens


def test_estimate_tokens_english():
    """英文按 ~4 字符/token 估算：200 字符 ≈ 50 token。"""
    est = estimate_tokens("a" * 200)
    assert 40 <= est <= 60


def test_estimate_tokens_chinese():
    """中文按 ~1.7 字符/token 估算：100 汉字 ≈ 50~60 token。"""
    est = estimate_tokens("软" * 100)
    assert 45 <= est <= 65


def test_estimate_messages():
    """多消息估算不为 0，且随内容增长而增长。"""
    from langchain_core.messages import HumanMessage

    msgs1 = [HumanMessage(content="你好")]
    msgs2 = [HumanMessage(content="你好" * 50)]
    assert estimate_messages_tokens(msgs2) > estimate_messages_tokens(msgs1)


def test_metrics_record_and_snapshot():
    """计量：record/snapshot 汇总正确，错误单独计数。"""
    m = LLMMetrics()
    m.record(LLMUsage(caller="agent", duration=0.5, prompt_tokens=100, completion_tokens=50, total_tokens=150))
    m.record(LLMUsage(caller="agent", duration=0.7, prompt_tokens=200, completion_tokens=60, total_tokens=260))
    m.record_error("finalize", 0.3, "boom")

    s = m.snapshot()
    assert s["total_calls"] == 3
    assert s["total_errors"] == 1
    assert s["total_tokens"] == 410
    assert s["total_duration_sec"] == 1.5
    assert s["by_caller"]["agent"]["count"] == 2
    assert s["by_caller"]["finalize"]["errors"] == 1
