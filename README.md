# 🤖 Agent 任务助手（LangGraph ReAct）

> 一个**工程级**的 ReAct 模式智能体：自动拆解复杂问题、自主调度工具、记录每一步思考过程。
> 内置 **计算器 / 文档总结 / 网络搜索** 三个工具，支持**多轮记忆**、**限流计量**、**HTTP API 与 Web 聊天界面**，Docker 一键部署。

---

## ✨ 项目亮点

- 🧠 **ReAct 思考循环**：`思考 → 行动 → 观察`，大模型自主决定何时调用哪个工具、何时收敛作答
- 🔧 **工具即插即用**：基于 LangChain `@tool` 注册表，新增工具无需改动图代码（开闭原则）
- 🛡️ **生产级健壮性**：防无限循环、工具失败自动重试、安全计算器（AST 白名单）、限流计量
- 🧵 **多轮记忆**：短期会话记忆 + 长期跨会话记忆（SQLite 持久化）
- 🌐 **Web 界面 + REST API**：内置聊天页面，浏览器零部署即用；`/api/ask` 可集成到任意系统
- 🐳 **Docker 一键部署**：API 常驻服务 + 记忆持久化 + 非 root 运行

---

## 📐 Agent 工作流程

### 核心思想：ReAct（Reasoning + Acting）

传统大模型只能「一次生成答案」；ReAct 让模型在**推理与行动之间循环迭代**，
直到信息足够才收敛——这是它能完成多步复杂任务的根本原因。

### 工作流程（StateGraph 状态机）

```
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              │
        ▼                                                              │
    ┌────────┐   有工具调用且未达上限      ┌─────────┐                  │
    │  agent │──────────────────────────▶│  tools  │──────────────────┘
    │  (LLM) │                           │ (执行)  │
    └────────┘                           └─────────┘
        │  ▲                                  │
        │  │ 有工具调用但已达次数上限           │ 工具结果回填(ToolMessage)
        │  ▼                                  │
        │  ┌───────────┐                      │
        └─▶│ finalize  │◀─────────────────────┘
           └───────────┘   → END（强制收敛作答）
        │
        │ 无工具调用（信息充足）
        ▼
      END（输出最终答案）
```

**分步说明：**

| 步骤 | 节点 | 做什么 |
|---|---|---|
| ① 理解与拆解 | `agent` | 模型读取「系统提示词 + 历史消息 + 长期记忆」，把复杂问题拆成子步骤 |
| ② 决策调用 | `agent` | 模型输出「思考 + 工具调用请求」（哪个工具、什么参数） |
| ③ 执行工具 | `tools` | LangGraph 的 ToolNode 执行模型请求的工具，返回结果作为「观察」 |
| ④ 继续推理 | `agent` | 模型读取工具结果，决定下一步（再调用工具 / 直接作答） |
| ⑤ 收敛作答 | `END` | 信息充足时模型停止调用工具，输出带依据的最终答案 |

**防无限循环双保险**：状态中维护 `tool_calls` 计数器，达到上限强制进入
`finalize` 节点用「未绑定工具」的模型收敛作答；框架层 `recursion_limit`
动态放大作为兜底——即使计数器失效，LangGraph 也会在超级步数耗尽时报错而非挂死。

---

## 🔧 工具调度原理

### 1. 模型如何"认识"工具？—— 函数调用协议（Function Calling）

LangChain `@tool` 装饰器会自动从**函数签名 + Docstring** 生成 JSON Schema，
随请求一起注入大模型。模型据此「知道」存在哪些工具、每个工具做什么、参数是什么：

```python
@tool
def calculator(expression: str) -> str:
    """安全计算数学表达式（四则运算、幂、取余、abs/min/max/round/pow）。

    参数:
        expression: 待计算的数学表达式，如 "2**10"、"15*3+7"
    """
    ...
```

> ⚠️ 因此 **Docstring 质量直接决定工具调用准确率**——它是模型选择工具的「说明书」。

### 2. 谁决定调用？—— 大模型决策

模型在每次 `agent` 节点运行时，根据当前问题自主输出二选一：

- **输出回答** → 无工具调用 → 收敛作答；
- **输出 tool_calls**（如 `calculator(expression="7**3")`）→ 进入工具执行。

这赋予了 Agent 与传统流水线本质不同的能力：**决策不是写死的，而是每步动态产生**。

### 3. 谁执行？—— LangGraph 的 ToolNode

`tools` 节点遍历模型请求的工具调用列表，逐一执行真实函数，结果以
`ToolMessage` 回填到图状态，供下一轮 `agent` 读取。

### 4. 工具注册表 —— 开闭原则

```python
# tools/base.py —— 新增工具只需在 get_tools() 里追加一行
def get_tools() -> list[BaseTool]:
    return [calculator, document_summarizer, web_search]
```

Agent 图只依赖这一个接口，**加工具不改图、不加路由逻辑**。

### 5. 工具执行保障

| 机制 | 说明 |
|---|---|
| 失败自动重试 | 指数退避（1s→2s→4s...），默认重试 2 次 |
| 连续失败终止 | 超过上限进入 `abort` 节点，生成**友好提示**而非裸堆栈 |
| 安全计算器 | AST 白名单求值，拦截 `__import__`/`open`/属性访问等注入攻击 |
| 耗时日志 | 每次工具调用记录执行耗时，可审计 |

---

## 🆚 Agent 能做什么传统 RAG 做不到的事？

RAG（检索增强生成）的范式是：**检索 → 拼接上下文 → 单轮生成**。
它擅长「从给定文档库中找事实」，但本质是**一次性的问答**；
而本项目的 Agent 是**多步执行器**，两者解决的是不同维度的问题。

| 维度 | 传统 RAG | 本项目 Agent |
|---|---|---|
| 处理流程 | 单轮：检索 → 拼上下文 → 生成 | 多轮：思考 → 调用工具 → 观察 → 再思考 |
| 多步组合任务 | ❌ 一次生成，无法中途调整 | ✅ 自动拆解，分步执行（如"先搜定义再算结果"） |
| 精确计算 | ❌ 只能找现成答案 | ✅ 调用计算器精确求值 |
| 实时信息 | ❌ 依赖静态文档库 | ✅ 联网搜索，获取最新事实 |
| 交互式总结 | ❌ 检索文档片段 | ✅ 读取文件/长文本，LLM 生成结构化摘要 |
| 结果验证迭代 | ❌ 错了就错 | ✅ 根据观察结果换策略重试 |
| 跨工具组合 | ❌ 单一检索通道 | ✅ 计算 + 搜索 + 总结任意编排 |
| 失败处理 | ❌ 返回空/幻觉 | ✅ 重试、降级、如实告知、友好终止 |
| 可审计性 | ❌ 黑盒一次调用 | ✅ 思考链逐步留痕，可回放 |

**典型 Agent 专属任务：**

```
你: 请搜索一下什么是强化学习，然后计算 15 的平方与 3 的立方的和

[思考] 分两步处理：先搜索定义，同时计算 15² 与 3³。
[选择工具] web_search + calculator
[工具] calculator 返回: 252
[思考] 第一次搜索结果不相关，换更聚焦的关键词重新搜索…
[工具] web_search 返回: 命中 5 条
[选择工具] calculator, 参数={'expression': '225 + 27'}   # 二次验证
[给出答案] 强化学习是…；15² + 3³ = 252（经 calculator 两次验证）
```

这类「检索 + 计算 + 验证」的组合，RAG 单轮范式无法完成。

---

## 🧵 记忆系统（短期 + 长期）

对话记忆用 SQLite（零配置单文件，标准库实现）持久化，三张表：

| 表 | 类型 | 内容 |
|---|---|---|
| `conversations` | 会话 | 会话 ID、标题（自动取首问前 30 字）、时间 |
| `messages` | 短期记忆 | 每轮「用户提问 + Agent 回答」Q/A 对 |
| `long_term_memories` | 长期记忆 | 跨会话的持久事实（用户偏好 / 关注领域等） |

- **短期记忆**：同一 `session_id` 连续提问，自动加载历史 Q/A 拼入上下文；
- **长期记忆**：每轮对话后 LLM 提炼「值得长期记住的信息」，去重入库；
  新会话自动注入系统提示词，让 Agent 冷启动也"记得"用户是谁。

---

## 🌐 界面与接口

### Web 聊天界面（内置）

启动 API 后浏览器访问 **`http://localhost:8000/`** 即可使用聊天页面：
左侧会话管理（新建/切换，本地持久化），右侧对话区可展开查看
**思考链、工具调用次数、token 用量**。

### REST API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 内置 Web 聊天页面 |
| `/api/ask` | POST | 对话接口：`question`（必填）+ `session_id`（可选，续聊） |
| `/api/health` | GET | 健康检查：服务状态 + 限流器状态 |
| `/api/metrics` | GET | LLM 用量统计：耗时 / token / 调用方分组 |

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"计算 12 的平方","session_id":"demo"}'
```

响应含答案、思考链 `chain`、工具调用次数、token 用量 `usage`。
**异常映射**：接口/LLM 限流 → `429`；LLM 上游异常 → `502`；未知错误 → `500`。

---

## ⚙️ 工程结构

```
langgraph-react-agent/
├── main.py                    # CLI 入口（单次提问 / 交互式对话）
├── serve.py                   # HTTP API 启动入口（uvicorn）
├── static/index.html          # 内置 Web 聊天页（零依赖单文件）
├── config/settings.py         # 集中配置 + 启动即校验（fail-fast）
├── core/
│   ├── exceptions.py          # 统一异常体系
│   ├── logging.py             # 日志（控制台 + 滚动文件双输出）
│   ├── llm.py                 # LLM 统一入口：限流预扣 + 耗时/token 计量 + 结算校准
│   ├── ratelimit.py           # 令牌桶限流器（RPM/TPM 双桶，线程安全）
│   └── retry.py               # 指数退避重试
├── api/server.py              # FastAPI：/api/ask /api/health /api/metrics + 每IP限流 + 静态托管
├── tools/
│   ├── base.py                # 工具注册表（新增工具无需改图）
│   ├── calculator.py          # 安全计算器（AST 白名单防注入）
│   ├── document_summarizer.py # 文档总结
│   └── web_search.py          # 网络搜索（Bing/Wikipedia/DuckDuckGo）
├── agent/
│   ├── state.py               # 图状态
│   ├── prompts.py             # ReAct 系统提示词
│   └── graph.py               # 图构建 + 思考链日志 + 运行入口
├── memory/                    # SQLite 记忆系统（短期/长期/仓库）
├── tests/                     # 单元测试（无需网络）
├── Dockerfile / docker-compose.yml
└── .env.example
```

---

## 🚀 快速开始

### 方式一：本地运行

```bash
# 1. 创建虚拟环境并安装依赖
cd langgraph-react-agent
python -m venv .venv
.venv\Scripts\activate        # Windows；Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量（必填 API Key）
cp .env.example .env          # 填写 AGENT_OPENAI_API_KEY
# 任意 OpenAI 兼容网关均可：DeepSeek / OpenAI / 通义 / vLLM / OneAPI

# 3. CLI 单次提问 / 交互对话
python main.py --question "计算 (1234*56+789)/3"
python main.py

# 4. 启动 Web 界面 + API（浏览器打开 http://localhost:8000/）
python serve.py

# 5. 运行单元测试
python -m pytest tests/ -v    # 需先 pip install pytest
```

### 方式二：Docker 一键部署（推荐）

```bash
# 一键构建 + 启动常驻 API（含 Web 界面、端口映射、记忆持久化、自动重启）
docker compose up -d --build

# 浏览器访问 Web 聊天页
open http://localhost:8000/

# 或直接调用 API
curl http://localhost:8000/api/health
```

> Docker 镜像默认启动 API 服务（监听 8000）；CLI 方式：
> `docker run --rm --env-file .env react-agent python main.py --question "计算 2**10"`

---

## ⚙️ 关键配置（.env）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `AGENT_OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容网关地址 |
| `AGENT_OPENAI_API_KEY` | 必填 | API Key（生产用环境变量注入，不入库） |
| `AGENT_CHAT_MODEL` | `deepseek-chat` | 对话模型名 |
| `AGENT_MAX_ITERATIONS` | 8 | Agent 最大思考-调用轮数（防死循环） |
| `AGENT_TOOL_MAX_RETRIES` / `AGENT_TOOL_RETRY_BACKOFF` | 2 / 1.0 | 工具失败重试与退避基数 |
| `AGENT_MEMORY_ENABLED` | `true` | 是否启用记忆系统 |
| `AGENT_MEMORY_DB_PATH` | `memory.db` | 记忆数据库路径 |
| `AGENT_LLM_RATE_LIMIT_RPM` / `TPM` | 60 / 100000 | LLM 每分钟调用次数 / token 上限 |
| `AGENT_LLM_RATE_LIMIT_TIMEOUT` | 10 | 限流等待超时（秒） |
| `AGENT_LLM_RATE_LIMIT_BURST_SECONDS` | 30 | 令牌桶突发窗口（秒） |
| `AGENT_API_HOST` / `AGENT_API_PORT` | 0.0.0.0 / 8000 | API 监听地址 / 端口 |
| `AGENT_API_RATE_LIMIT_RPM` / `BURST` | 30 / 10 | 每 IP 每分钟请求数 / 突发容量 |
| `AGENT_SEARCH_ENGINE` | `bing` | bing / wikipedia / duckduckgo |
| `AGENT_LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## 🏗️ 优化点（已实现的生产级设计）

1. **防无限循环**：图内工具计数器 + 框架 `recursion_limit` 双保险，从根本上杜绝模型陷入死循环
2. **工具失败自动重试**：指数退避；连续失败进入 `abort` 节点给用户**友好提示**
3. **安全计算器**：AST 白名单求值，拦截代码注入；搜索/总结均有超时与大小限制
4. **LLM 成本控制**：令牌桶限流（RPM 次数 + TPM token 双桶），估算预扣 + 实际结算校准
5. **接口防滥用**：按客户端 IP 独立令牌桶，超限返回 429
6. **可观测**：每次 LLM 调用打印耗时/token，`/api/metrics` 暴露进程内用量统计
7. **多轮记忆**：短期会话记忆 + 长期跨会话记忆，SQLite 持久化
8. **健壮分层**：统一异常体系、集中配置 fail-fast 校验、双输出日志
9. **安全默认**：API Key 不入库、Docker 非 root 运行、`.env`/`sync.ps1` 被 git 忽略

---

## ⚠️ 已知不足与后续改进方向

> 诚实说明当前边界，避免「宣称完成但实际不可用」的坑。

1. **无流式输出（SSE）**：当前 `/api/ask` 一次性返回，长回答等待感明显。
   改进：FastAPI `StreamingResponse` + 前端 `fetch` 流式渲染（已在路线图）。
2. **工具串行执行**：模型一次请求多个工具时当前逐个执行。
   改进：LangGraph `Send` API 支持并行工具调用，可显著加速复合任务。
3. **长会话 token 成本**：历史消息 + 长期记忆全量注入上下文，会话变长后成本线性上升。
   改进：滑动窗口截断、历史摘要压缩、记忆按相关性筛选注入。
4. **接口无鉴权**：目前只有每 IP 限流，无 API Key / JWT 认证。
   生产暴露公网前必须补充身份认证。
5. **搜索稳定性依赖外部站点**：Bing 等可能限流/降级。
   改进：多搜索引擎故障切换 + 结果缓存。
6. **记忆提炼依赖 LLM**：长期记忆抽取准确性受模型影响，可能误提炼。
   改进：加置信度阈值 / 人工确认机制。
7. **单机 SQLite 扩展性有限**：多实例水平扩展需替换为 PostgreSQL / Redis 记忆存储。
8. **单模型无路由**：复杂/简单任务共用同一模型，成本非最优。
   改进：按问题复杂度路由到不同规格模型。
9. **无知识库能力**：本 Agent 定位是「任务执行」而非「知识问答」。
   若需基于私有文档作答，可新增 `retriever` 工具（向量检索）挂入注册表即可。

---

## 📖 参考示例

```
你: 帮我计算 (1234*56+789)/3 的结果是多少

[思考] 这是一个纯数学计算问题，我直接用计算器工具来求值。
[选择工具] calculator, 参数={'expression': '(1234*56+789)/3'}
[工具] calculator 返回: 23297.666666666668
[给出答案] (1234×56 + 789) ÷ 3 = 23297.666…（约 23297.67）
```

---

## 📦 环境要求

- **Python** 3.10+
- **LLM 网关**：任意 OpenAI 兼容服务（DeepSeek / OpenAI / 通义 / vLLM / OneAPI 等）
- **Docker**（可选）：Docker Desktop 或任意 Docker Engine
- **网络**：默认搜索后端 Bing 需国内可直连；海外后端需能访问对应站点

---

MIT License © 2026 [zh666888hz-ux](https://github.com/zh666888hz-ux)
