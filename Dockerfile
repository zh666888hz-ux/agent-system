# ============================================================
# LangGraph ReAct Agent 生产镜像
#
# 设计要点：
#   1. 基于 slim 基础镜像，减小体积；
#   2. 先复制 requirements 再装依赖，充分利用 Docker 层缓存（改代码不重装依赖）；
#   3. 非 root 用户运行（appuser），遵循最小权限原则，降低容器逃逸风险；
#   4. 敏感配置（API Key）通过 --env-file 运行时注入，绝不打进镜像；
#   5. PYTHONUNBUFFERED 保证日志实时输出到 stdout（便于 docker logs 查看）。
#
# 构建：docker build -t react-agent .
# 运行：
#   docker run --rm -it --env-file .env react-agent                     # 交互式对话
#   docker run --rm --env-file .env react-agent python main.py --question "计算 2**10"
# ============================================================

FROM python:3.11-slim

# 系统环境：UTF-8 保证中文输出正常；关闭字节码缓存
# PIP_INDEX_URL：使用国内 PyPI 镜像，规避海外源网络不稳定导致的下载失败/哈希校验错误
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

WORKDIR /app

# 1) 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) 复制应用代码
COPY main.py .
COPY config/ config/
COPY core/ core/
COPY tools/ tools/
COPY agent/ agent/
COPY memory/ memory/
COPY docs/ docs/

# 3) 创建非 root 运行用户，并赋予日志目录 / 数据目录写权限
#    memory_data 用于承载记忆数据库（运行时通过卷挂载持久化）
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs /app/memory_data \
    && chown -R appuser:appuser /app
USER appuser

# 默认命令：展示帮助信息
CMD ["python", "main.py", "--help"]
