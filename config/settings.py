"""
config/settings.py
==================
集中式配置管理模块。

设计原理：
    - 基于 pydantic-settings 实现「环境变量 + .env 文件 + 默认值」三级配置解析。
    - 使用 env_prefix="AGENT_" 统一命名空间，避免与宿主机其他应用的环境变量冲突
      （工程化项目中，所有配置项带统一前缀是常见约定）。
    - 配置在进程启动时一次性加载并做「启动即校验」（fail-fast）：
      例如 API Key 缺失、模型名空白等致命错误在启动阶段直接抛异常，
      避免运行到一半才因配置缺失而崩溃——这正是生产环境要求的健壮性。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件位于 <root>/config/ 下，向上两级即为项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """全局配置模型。

    所有字段均可通过环境变量覆盖，例如：
        AGENT_OPENAI_BASE_URL=https://api.deepseek.com/v1
    也可写入项目根目录的 .env 文件（复制 .env.example 后填写）。
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",          # 环境变量统一前缀
        env_file=PROJECT_ROOT / ".env",  # 读取项目根目录 .env
        env_file_encoding="utf-8",
        extra="ignore",               # 忽略未声明字段，避免误报
    )

    # ---------- LLM（OpenAI 兼容网关） ----------
    # 任意 OpenAI 兼容服务均可：DeepSeek / OpenAI / 通义 / vLLM / OneAPI 等
    openai_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="OpenAI 兼容 API 的 Base URL",
    )
    openai_api_key: str = Field(
        default="",
        description="API Key（生产环境必须通过环境变量注入，禁止硬编码）",
    )
    chat_model: str = Field(
        default="deepseek-chat",
        description="对话模型名",
    )
    llm_timeout: float = Field(
        default=60.0,
        ge=1,
        description="LLM 单次调用超时（秒），防止外部服务无响应导致进程挂死",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="LLM 调用失败自动重试次数",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="采样温度：0 表示接近确定性输出，适合工具调用场景",
    )

    # ---------- LLM 调用限流与计量 ----------
    llm_rate_limit_rpm: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="LLM 每分钟最大调用次数（RPM），令牌桶限流防突发打爆上游",
    )
    llm_rate_limit_tpm: int = Field(
        default=100000,
        ge=1000,
        le=10000000,
        description="LLM 每分钟最大 token 消耗（TPM），按估算预扣+实际结算校准",
    )
    llm_rate_limit_timeout: float = Field(
        default=10.0,
        ge=0,
        le=120,
        description="限流等待超时（秒）：超过则放弃本次调用并抛限流异常",
    )
    llm_rate_limit_burst_seconds: float = Field(
        default=30.0,
        ge=1,
        le=3600,
        description="LLM 令牌桶突发窗口（秒）：桶容量=每秒速率×窗口，允许突发的同时限制长期均值",
    )

    # ---------- HTTP API（FastAPI 服务） ----------
    api_host: str = Field(
        default="0.0.0.0",
        description="API 服务监听地址（0.0.0.0 允许容器外部访问）",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API 服务监听端口",
    )
    api_rate_limit_rpm: int = Field(
        default=30,
        ge=1,
        le=10000,
        description="HTTP 接口每个客户端 IP 每分钟最大请求数（接口级限流）",
    )
    api_rate_limit_burst: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="HTTP 接口限流令牌桶突发容量（允许短时突发）",
    )

    # ---------- Agent 运行参数 ----------
    max_iterations: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Agent 最大「思考-调用工具」轮数，防止工具循环导致的死循环",
    )

    # ---------- 工具：失败自动重试 ----------
    tool_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="工具调用失败后的自动重试次数（连续失败超过该值则终止任务并友好提示）",
    )
    tool_retry_backoff: float = Field(
        default=1.0,
        ge=0.1,
        le=30,
        description="工具重试的指数退避基数（秒）：首次等待 1s，之后 2s、4s ... 翻倍",
    )

    # ---------- 记忆系统 ----------
    memory_enabled: bool = Field(
        default=True,
        description="是否启用记忆系统（短期会话记忆 + 长期跨会话记忆）",
    )
    memory_db_path: Path = Field(
        default=PROJECT_ROOT / "memory.db",
        description="记忆数据库路径（SQLite 单文件）",
    )
    memory_extract: bool = Field(
        default=True,
        description="是否在每轮对话后用 LLM 提炼长期记忆（关闭可省 token）",
    )
    memory_inject_limit: int = Field(
        default=20,
        ge=0,
        le=100,
        description="每次注入 Agent 上下文的长期记忆条数上限",
    )

    # ---------- 工具：网络搜索 ----------
    search_engine: str = Field(
        default="bing",
        description="搜索引擎后端：bing（默认，国内可达）/ wikipedia / duckduckgo",
    )
    search_top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="网络搜索返回结果条数上限",
    )
    search_timeout: float = Field(
        default=15.0,
        ge=1,
        description="网络搜索 HTTP 超时（秒）",
    )

    # ---------- 工具：文档总结 ----------
    summary_max_tokens: int = Field(
        default=500,
        ge=50,
        le=4000,
        description="文档总结输出最大 token 数",
    )
    summary_max_chars: int = Field(
        default=8000,
        ge=100,
        description="文档总结工具单次接受的最大字符数（超长截断，保护上下文窗口）",
    )

    # ---------- 日志 ----------
    log_level: str = Field(
        default="INFO",
        description="日志级别：DEBUG / INFO / WARNING / ERROR",
    )
    log_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="日志输出目录",
    )

    # ---------- 参数校验：启动即失败 ----------
    @field_validator("openai_api_key")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        """API Key 为必填项：缺失时启动阶段直接报错（fail-fast）。"""
        v = v.strip()
        if not v:
            raise ValueError(
                "缺少 AGENT_OPENAI_API_KEY：请复制 .env.example 为 .env 并填写 API Key"
            )
        return v

    @field_validator("search_engine")
    @classmethod
    def _validate_search_engine(cls, v: str) -> str:
        """搜索引擎后端白名单校验，非法值启动即拒绝。"""
        if v not in {"bing", "wikipedia", "duckduckgo"}:
            raise ValueError(
                f"不支持的 search_engine={v!r}，可选：bing / wikipedia / duckduckgo"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"非法的 log_level={v!r}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（进程内缓存，避免重复解析 .env）。"""
    return Settings()
