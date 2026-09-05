"""
serve.py
========
HTTP API 服务启动入口。

用法：
    python serve.py                        # 使用默认配置（0.0.0.0:8000）
    python serve.py --port 9000            # 指定端口
    python serve.py --reload               # 开发模式热重载

部署：
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse

import uvicorn

from config.settings import get_settings
from core.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="react-agent-api", description="启动 ReAct Agent HTTP API")
    parser.add_argument("--host", type=str, default=None, help="监听地址（默认取配置）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（默认取配置）")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    logger.info("启动 HTTP API 服务: http://%s:%s （接口限流 RPM=%d/客户端）",
                host, port, settings.api_rate_limit_rpm)

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=args.reload,
        log_config=None,  # 复用项目统一日志配置（core/logging）
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
