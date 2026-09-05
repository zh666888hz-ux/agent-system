# 🤖 LangGraph ReAct Agent

> 基于 **LangGraph + OpenAI 兼容 API** 的 ReAct 模式智能体：自动拆解复杂问题、自主选择工具、记录每一步思考过程。内置 **计算器 / 文档总结 / 简单网络搜索** 三个工具。

---

## 一、项目简介

本项目实现了一个「思考 → 行动 → 观察」循环（ReAct: Reasoning + Acting）的智能体：

- 用户提出复杂问题，Agent 先拆解为子步骤；
- 每一步由大模型自主判断**该调用哪个工具**；
- 工具返回结果后，Agent 继续推理，直到信息充足才给出带依据的最终答案；
- 整个过程（模型思考、选择的工具、工具结果、最终回答）**逐条记录日志**，可完整审计。

**核心价值**：把「会推理的大模型」与「可执行的外部能力（计算 / 文档 / 网络）」组合起来，
解决单个模型无法独立完成的问题（如精确计算、读取本地文件、获取实时信息）。

---

## 二、架构设计

### 技术栈

| 类别 | 技术 |
|---|---|
| Agent 框架 | LangGraph（StateGraph + 条件回边，实现 ReAct 循环） |
| LLM | OpenAI 兼容 API（DeepSeek 等，base_url 可切换） |
| 工具协议 | LangChain `@tool`（自动生成 JSON Schema 注入模型） |
| 配置 | pydantic-settings（`AGENT_` 前缀环境变量 + .env） |
| 网络 | requests（超时 + 指数退避重试） |

### 分层结构

```
langgraph-react-agent/
├── main.py                    # CLI 入口（单次提问 / 交互式对话）
├── config/settings.py         # 集中配置 + 启动即校验（fail-fast）
├── core/
│   ├── exceptions.py          # 统一异常体系（AgentError / ToolExecutionError 等）
│   ├── logging.py             # 日志（控制台 + 滚动文件双输出）
│   └── llm.py                 # OpenAI 兼容 LLM 客户端工厂
├── tools/
│   ├── base.py                # 工具注册表（新增工具无需改图代码）
│   ├── calculator.py          # 安全计算器（AST 白名单，防代码注入）
│   ├── document_summarizer.py # 文档总结（LLM 摘要，编码探测 + 大小限制）
│   └── web_search.py          # 网络搜索（Bing / Wikipedia / DuckDuckGo 多后端）
├── agent/
│   ├── state.py               # 图状态（messages + add_messages 累积轨迹）
│   ├── prompts.py             # ReAct 系统提示词
│   └── graph.py               # ReAct 图构建 + 思考链日志 + 运行入口
├── requirements.txt
└── .env.example
```

### ReAct 图（LangGraph 状态机）

```
        ┌──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      │
    ┌───────┐   有工具调用且未达上限   ┌───────┐                │
    │ agent │────────────────────────▶│ tools │───────────────┘
    │ (LLM) │                         │ (执行) │
    └───────┘                         └───────┘
        │  ▲
        │  │ 有工具调用但已达次数上限（防无限循环）
        │  ▼
        │  ┌───────────┐
        └─▶│  finalize  │──▶ END（基于已有结果强制收敛作答）
           └───────────┘
        │ 无工具调用（信息充足）
        ▼
      END（输出最终答案）
```

- **agent 节点**：把「系统提示词 + 历史消息」交给绑定工具的 LLM，输出回答或工具调用请求；
- **tools 节点**：执行模型请求的工具调用，结果以 ToolMessage 回填，并自增工具调用计数器；
- **条件边**：有 `tool_calls` 且计数未达上限 → tools；有 `tool_calls` 但已达上限 → finalize；否则 → END。

### 防无限循环（关键安全设计）

大模型在某些场景下可能陷入「反复调用工具」的循环（如搜索不到满意结果时不断换词重搜）。
本项目采用**双重保障**：

1. **图内计数器（核心）**：状态中维护 `tool_calls` 计数器，每执行一轮工具调用就自增；
   条件边在模型再次请求调用工具时先检查是否已达 `AGENT_MAX_ITERATIONS` 上限，
   达到则进入 **finalize 节点**——用「未绑定工具」的模型强制生成最终答案，从根本上终止循环；
2. **框架层兜底**：`recursion_limit` 按上限动态放大（`max_iterations*3 + 5`），
   即使计数器逻辑失效，LangGraph 也会在超级步数耗尽时抛错而非无限挂起。

finalize 节点会清理「已请求但未执行」的孤儿 tool_calls 消息，保证发给 LLM 的消息序列
符合 OpenAI 工具调用协议（避免 400 错误），并如实告知用户已达上限、基于已有结果作答。

### 记忆系统（短期会话记忆 + 长期跨会话记忆）

对话记忆用 SQLite（零配置单文件，标准库实现）持久化，三张表：

| 表 | 类型 | 内容 |
|---|---|---|
| `conversations` | 会话 | 会话 ID、标题（自动取首问前 30 字）、时间 |
| `messages` | 短期记忆 | 每轮「用户提问 + Agent 回答」Q/A 对 |
| `long_term_memories` | 长期记忆 | 跨会话的持久事实（用户偏好 / 关注领域等） |

- **短期记忆**：同一 `session_id` 连续提问时，自动加载历史 Q/A 对拼入上下文，实现多轮对话记忆；
- **长期记忆**：每轮对话结束后，用 LLM 提炼「值得长期记住的信息」（记忆巩固），去重后入库；
  新会话开始时自动注入系统提示词，让 Agent 冷启动也能"记得"用户是谁；
- **CLI 支持**：`--session <id>` 指定会话续聊、`--list-sessions` 查看历史会话。

### 工具失败自动重试

生产环境中工具调用常因网络抖动 / 远端临时故障而失败。本项目内置重试机制：

- 工具调用失败后按 `AGENT_TOOL_MAX_RETRIES`（默认 2 次）自动重试；
- 重试间隔**指数退避**（`AGENT_TOOL_RETRY_BACKOFF` 基数：1s → 2s → 4s ...），给远端恢复时间；
- **连续失败终止**：超过重试上限后进入 abort 节点，生成**友好提示**（含失败工具名、
  已尝试次数、可能原因、建议），而不是抛裸堆栈或让 Agent 无限重试；
- 每次失败 / 重试 / 终止都记录详细日志，便于排查。

### 完整运行日志

全流程可审计，覆盖：会话生命周期（创建/复用/完成）、短期记忆加载、长期记忆注入与提炼、
模型思考、工具选择、工具执行与耗时、失败重试、上限收敛、任务终止、最终答案。

---

## 三、内置工具

| 工具 | 能力 | 关键实现 |
|---|---|---|
| `calculator` | 数学表达式计算 | AST 白名单安全求值，拦截 `__import__`/`open`/属性访问等注入攻击 |
| `document_summarizer` | 总结文本/本地文件 | 独立 LLM 调用；自动编码探测；大小限制；超长截断 |
| `web_search` | 互联网关键词搜索 | 默认 **Bing**（国内直连）；可选 Wikipedia/DuckDuckGo；超时+重试 |

---

## 四、依赖列表

| 依赖 | 版本要求 | 用途 |
|---|---|---|
| `langgraph` | >=1.0 | ReAct 图构建与执行 |
| `langchain-core` | >=1.0 | 工具定义（@tool）与消息模型 |
| `langchain-openai` | >=1.0 | OpenAI 兼容 LLM 客户端 |
| `pydantic` | >=2.0 | 数据模型 |
| `pydantic-settings` | >=2.0 | 配置加载（.env / 环境变量） |
| `requests` | >=2.31 | 网络搜索 HTTP 客户端 |

---

## 五、环境说明

- **Python**：3.10+
- **LLM 网关**：任意 OpenAI 兼容服务（DeepSeek / OpenAI / 通义 / vLLM / OneAPI 等）
- **网络**：默认搜索后端 Bing（cn.bing.com）需国内可直连；若使用海外后端需能访问对应站点
- **密钥**：API Key 通过 `.env` 注入（已被 .gitignore 排除，不入库）

---

## 六、启动步骤

```bash
# 1. 克隆/进入项目，创建虚拟环境
cd langgraph-react-agent
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（必填 API Key）
cp .env.example .env
# 编辑 .env，至少填写：
#   AGENT_OPENAI_API_KEY=sk-xxx

# 4a. 单次提问
python main.py --question "计算 (1234*56+789)/3"

# 4b. 交互式对话（输入 exit 退出）
python main.py

# 5. 其他参数
python main.py --question "..." --no-chain   # 不打印思考链
python main.py --question "..." --log-level DEBUG

# 6. 多轮会话记忆
python main.py --question "我的专业是软件工程" --session my-session   # 指定会话
python main.py --question "还记得我专业吗？" --session my-session      # 续聊：自动加载历史+长期记忆
python main.py --list-sessions                                         # 查看历史会话
```

### 关键配置（.env）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `AGENT_OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容网关地址 |
| `AGENT_OPENAI_API_KEY` | 必填 | API Key（生产用环境变量注入） |
| `AGENT_CHAT_MODEL` | `deepseek-chat` | 对话模型名 |
| `AGENT_LLM_TIMEOUT` / `AGENT_LLM_MAX_RETRIES` | 60 / 3 | LLM 超时与重试 |
| `AGENT_MAX_ITERATIONS` | 8 | Agent 最大思考-调用轮数（防死循环） |
| `AGENT_TOOL_MAX_RETRIES` / `AGENT_TOOL_RETRY_BACKOFF` | 2 / 1.0 | 工具失败重试次数与退避基数 |
| `AGENT_MEMORY_ENABLED` | `true` | 是否启用记忆系统 |
| `AGENT_MEMORY_DB_PATH` | `memory.db` | 记忆数据库路径 |
| `AGENT_MEMORY_EXTRACT` / `AGENT_MEMORY_INJECT_LIMIT` | `true` / 20 | 长期记忆提炼开关 / 注入条数上限 |
| `AGENT_SEARCH_ENGINE` | `bing` | bing / wikipedia / duckduckgo |
| `AGENT_LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## 七、Docker 部署

### 1. 构建镜像

```bash
docker build -t react-agent .
```

> 构建已内置阿里云 PyPI 镜像（`PIP_INDEX_URL`），规避海外源网络不稳定导致的下载失败/哈希校验错误。

### 2. 运行

```bash
# 单次提问（注入 .env 密钥）
docker run --rm --env-file .env react-agent python main.py --question "计算 2**10"

# 交互式对话（-it 保持终端）
docker run --rm -it --env-file .env react-agent

# 多轮会话记忆（挂载数据目录，持久化记忆数据库）
docker run --rm --env-file .env \
  -v "$(pwd)/memory_data:/app/memory_data" \
  -e AGENT_MEMORY_DB_PATH=/app/memory_data/memory.db \
  react-agent python main.py --question "我的专业是软件工程" --session my-session
docker run --rm --env-file .env \
  -v "$(pwd)/memory_data:/app/memory_data" \
  -e AGENT_MEMORY_DB_PATH=/app/memory_data/memory.db \
  react-agent python main.py --question "还记得我专业吗？" --session my-session
```

> **记忆持久化**：容器内 `memory.db` 默认写在容器文件系统，容器删除即丢失。
> 挂载数据目录（`-v 宿主机目录:/app/memory_data`）并设置
> `AGENT_MEMORY_DB_PATH=/app/memory_data/memory.db` 即可持久化对话记忆。

### 3. 使用 Docker Compose（推荐，统一管理）

```bash
docker compose build
docker compose run --rm agent python main.py --question "计算 2**10"   # 单次
docker compose run --rm agent                                           # 交互式
```

> Compose 已内置：`AGENT_MEMORY_DB_PATH=/app/memory_data/memory.db` +
> 卷挂载 `./memory_data:/app/memory_data`，开箱即用记忆持久化。

### 4. 网络代理说明（重要）

Docker 容器的 NAT 出口 IP 可能被搜索引擎（Bing）识别为数据中心/共享 IP 而降级返回结果。
若容器内 `web_search` 返回不相关内容，把宿主代理透传给容器即可（requests 自动读取）：

```bash
# Windows Docker Desktop：host.docker.internal 指向宿主机
docker run --rm --env-file .env \
  -e HTTP_PROXY=http://host.docker.internal:7890 \
  -e HTTPS_PROXY=http://host.docker.internal:7890 \
  react-agent python main.py --question "搜索一下什么是深度学习"
```

即使不配置代理，Agent 的 ReAct 设计也会优雅降级：搜索异常时**如实告知用户、不编造结果**，
并基于自身知识作答（明确标注信息来源）。

---

## 八、示例效果

```
你: 帮我计算 (1234*56+789)/3 的结果是多少

[思考] 这是一个纯数学计算问题，我直接用计算器工具来求值。
[选择工具] calculator, 参数={'expression': '(1234*56+789)/3'}
[工具] calculator 返回: 23297.666666666668
[给出答案] (1234×56 + 789) ÷ 3 = 23297.666…（约 23297.67）
```

复合问题（Agent 自动拆解并**并行调用**工具、失败后**自主重搜**、计算**二次验证**）：

```
你: 请搜索一下什么是强化学习，然后计算 15 的平方与 3 的立方的和

[思考] 分两步处理：先搜索定义，同时计算 15² 与 3³。
[选择工具] web_search + calculator（并行）
[工具] calculator 返回: 252
[思考] 第一次搜索结果不相关，换更聚焦的关键词重新搜索…
[工具] web_search 返回: 命中 5 条
[选择工具] calculator, 参数={'expression': '225 + 27'}   # 二次验证
[给出答案] 强化学习是…；15² + 3³ = 252（经 calculator 两次验证）
```
