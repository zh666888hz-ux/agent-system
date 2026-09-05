"""
api/server.py
=============
FastAPI HTTP 接口层：把 ReAct Agent 服务化为 HTTP API。

设计原理：
    - 三个端点：
        * POST /api/ask        对话接口（核心）：传入 question + 可选 session_id，
                               返回答案、思考链、工具统计、token 用量、会话 ID；
        * GET  /api/health     健康检查：服务存活 + 版本 + 限流/计量状态；
        * GET  /api/metrics    LLM 调用统计（耗时 / token / 调用方分组）。
    - 接口级限流：按「客户端 IP」各维护一个令牌桶（每 IP 独立配额），
      超过 AGENT_API_RATE_LIMIT_RPM 返回 HTTP 429 + 友好提示。
      使用令牌桶而非固定窗口：允许短时突发、长期平均受限。
    - 线程模型：run_agent 为同步阻塞调用，handler 声明为 def（FastAPI 自动放入
      线程池执行），不阻塞事件循环；配合令牌桶的线程安全实现，多请求安全。
    - 记忆复用：进程内单例 MemoryRepository，跨请求共享数据库连接与建表。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.graph import run_agent
from config.settings import get_settings
from core.exceptions import LLMError, RateLimitError
from core.llm import get_limiter, get_metrics
from core.logging import get_logger
from core.ratelimit import TokenBucket
from memory.repository import MemoryRepository

logger = get_logger(__name__)
settings = get_settings()

# 前端静态资源目录：<项目根>/static（index.html 聊天页）
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# 进程内单例：记忆仓库（跨请求复用，内部缓存 SQLite 连接与建表）
_repository: MemoryRepository | None = None
_repo_lock = threading.Lock()


def _get_repository() -> MemoryRepository | None:
    """懒加载全局记忆仓库（线程安全）。"""
    global _repository
    if _repository is None:
        with _repo_lock:
            if _repository is None and settings.memory_enabled:
                _repository = MemoryRepository(settings.memory_db_path)
    return _repository


# 接口限流：按客户端 IP 维护令牌桶
_buckets: dict[str, TokenBucket] = {}
_buckets_lock = threading.Lock()


def _get_client_bucket(client_ip: str) -> TokenBucket:
    """获取（或创建）某客户端 IP 的限流令牌桶。

    令牌桶参数：
        capacity = burst（突发容量）
        refill_rate = rpm / 60（每秒补充速率，即长期平均速率）
    """
    with _buckets_lock:
        bucket = _buckets.get(client_ip)
        if bucket is None:
            # 长期不活跃的桶做简单清理，防止内存泄漏（超过 1000 个 IP 时淘汰最旧）
            if len(_buckets) > 1000:
                _buckets.clear()
            bucket = TokenBucket(
                capacity=settings.api_rate_limit_burst,
                refill_rate=settings.api_rate_limit_rpm / 60,
            )
            _buckets[client_ip] = bucket
        return bucket


app = FastAPI(
    title="ReAct Agent API",
    description="LangGraph ReAct 智能体 HTTP 接口：计算器 / 文档总结 / 网络搜索 / 多轮记忆",
    version="1.0.0",
)


# ---------- 前端静态页面 ----------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """根路径返回聊天页面（前端由本项目内置，无需额外部署）。"""
    return FileResponse(STATIC_DIR / "index.html")


# /static 目录静态资源（CSS/JS/图片等）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- 请求 / 响应模型 ----------
class AskRequest(BaseModel):
    """对话请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(
        default=None, max_length=64, description="会话 ID（续聊用，不传则新建会话）"
    )


class AskResponse(BaseModel):
    """对话响应体：答案 + 溯源 + 用量。"""

    answer: str = Field(description="Agent 最终回答")
    session_id: str = Field(description="会话 ID（下次续聊传入）")
    tool_calls: int = Field(description="本次工具调用次数")
    memory_new: int = Field(description="本次新增长期记忆条数")
    aborted: bool = Field(description="是否因工具连续失败而终止")
    chain: list = Field(default_factory=list, description="思考链")
    usage: dict = Field(default_factory=dict, description="本次 LLM 用量统计")


# ---------- 接口 ----------
@app.get("/api/health")
def health() -> dict:
    """健康检查：服务是否存活 + 核心依赖状态。"""
    return {
        "status": "ok",
        "service": "react-agent",
        "version": "1.0.0",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm_limiter": get_limiter().stats(),
    }


@app.get("/api/metrics")
def metrics() -> dict:
    """LLM 调用指标：耗时 / token 消耗 / 调用方分组。"""
    return {
        "llm_metrics": get_metrics().snapshot(),
        "llm_limiter": get_limiter().stats(),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(request: Request, body: AskRequest) -> AskResponse:
    """对话接口：核心入口，带接口级限流。"""
    client_ip = request.client.host if request.client else "unknown"
    question = body.question.strip()

    # ---------- 接口级限流（每 IP 令牌桶，超限返回 429） ----------
    if not _get_client_bucket(client_ip).acquire(1.0, timeout=0):
        logger.warning("接口限流触发: ip=%s", client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请稍后重试（每 IP 限 {settings.api_rate_limit_rpm} 次/分钟）",
        )

    # ---------- 调用 Agent ----------
    logger.info("API 收到请求: ip=%s, session=%s, 问题=%s",
                client_ip, body.session_id or "(新会话)", question)
    try:
        result = run_agent(
            question,
            session_id=body.session_id,
            repository=_get_repository(),
        )
    except RateLimitError as exc:
        # LLM 限流等待超时：属于「上游繁忙」，返回 429（客户端可稍后重试），
        # 语义上区别于服务器自身故障（500）。
        logger.warning("LLM 限流超时: ip=%s, 原因=%s", client_ip, exc)
        raise HTTPException(status_code=429, detail=f"模型服务繁忙，请稍后重试: {exc}") from exc
    except LLMError as exc:
        # 上游 LLM 服务异常：返回 502（Bad Gateway），保留原因摘要供排查。
        logger.error("LLM 上游调用失败: ip=%s, 原因=%s", client_ip, exc)
        raise HTTPException(status_code=502, detail=f"模型服务异常: {exc}") from exc
    except Exception as exc:  # 其余未知错误统一 500，不向客户端暴露堆栈细节
        logger.exception("API 调用 Agent 失败")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {exc}") from exc

    # 汇总 LLM 用量（全局累计统计，含各调用方耗时/token 明细）
    usage = get_metrics().snapshot()

    logger.info("API 完成: session=%s, 工具调用 %d 次",
                result["session_id"], result["tool_calls"])
    return AskResponse(
        answer=result["answer"],
        session_id=result["session_id"],
        tool_calls=result["tool_calls"],
        memory_new=result.get("memory_new", 0),
        aborted=result.get("aborted", False),
        chain=result["chain"],
        usage=usage,
    )
