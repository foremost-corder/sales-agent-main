# Sales Agent 项目说明书

> 代码目录与目标文件速查请先阅读 [ARCHITECTURE.md](ARCHITECTURE.md)。

> 文档状态：基于 2026-09-09 工作区代码整理。本文同时描述当前已经实现的系统、正在工作区开发但尚未提交的知识链路，以及下一阶段“销售通话评分”方案。凡标注为“规划”的内容均尚未在当前代码中实现。

## 1. 项目概述

Sales Agent 是一个面向销售通话场景的多轮对话智能体。系统将销售通话原文导入 PostgreSQL，通过可审计的知识分析流程完成说话人识别、原子事实提取、证据校验、知识切片与向量化，再由受控 ReAct Agent 根据用户问题选择查询工具，并基于工具证据生成回答。

项目的核心不是让大模型自由判断一切，而是将职责拆开：

- **确定性工作流**负责通话解析、证据定位、事实校验、版本管理和持久化；
- **Agent**负责理解自然语言、多轮上下文、选择工具和组织答案；
- **工具层**负责限定查询能力、参数范围、用户数据边界和错误格式；
- **证据账本**负责约束最终回答，避免模型在没有业务证据时给出确定性业务结论；
- **后续评分引擎**应基于已验证事实执行可版本化、可复算的规则，而不是直接让模型凭印象打分。

当前项目已经具备从通话导入、知识分析、知识检索到对话问答的主干能力。周评分 `score_week` 尚未落地，但现有 `phase`、`fact_type`、`score_tags`、原文证据、置信度和知识命名空间已经为评分提供了数据基础。

## 2. 建设目标与边界

### 2.1 当前目标

1. 持久化多轮会话和消息，服务重启后仍可读取上下文；
2. 允许 Agent 自主选择最少数量的业务工具；
3. 对销售通话做受控统计、原文读取、事实分析和语义检索；
4. 所有事实都尽可能绑定具体轮次和原文引用；
5. 保存 Agent 运行与工具调用审计记录；
6. 支持标准知识工作流与低调用成本的精简实验工作流；
7. 在不同知识命名空间中隔离基线结果与实验结果；
8. 为按通话、销售员和自然周生成可解释评分报告打好基础。

### 2.2 当前不应越过的边界

- 模型不能执行任意 SQL、Shell 或任意网络请求；
- 工具只接受 Pydantic 定义的白名单字段和有限操作符；
- 默认不把低置信度事实作为正式业务结论；
- 不能用“没有抽取到事实”直接证明“通话中没有发生”；
- 未经明确确认，不执行 CRM、任务、通知等外部写操作；
- 评分不能脱离原文证据，也不能因客户未提供某种机会而机械扣分；
- 当前 Web 端固定使用演示用户，尚不能视为生产级身份认证方案。

## 3. 当前完成度

| 模块 | 状态 | 当前说明 |
| --- | --- | --- |
| PostgreSQL + pgvector | 已实现 | Docker Compose 使用 `pgvector/pgvector:pg17`，迁移启用 `pgcrypto` 与 `vector` |
| 会话与消息持久化 | 已实现 | 支持创建、列表、详情、历史消息和发送消息 |
| Web 聊天工作台 | 已实现（基础版） | 原生 HTML/CSS/JS，支持会话切换、新建和发送消息 |
| ReAct Agent | 已实现 | LangGraph 有界工具循环，默认最多 6 次业务工具调用 |
| Agent/工具审计 | 已实现 | 保存 `agent_runs` 与 `tool_calls` |
| 最终答案证据约束 | 已实现 | `submit_final_answer` 是唯一正常出口，业务结论必须引用有效证据编号 |
| 通话导入 | 已实现 | 支持 TXT 文件夹预览和提交导入，按用户与外部通话 ID 幂等跳过 |
| 通话统计与原文读取 | 已实现 | 白名单语义查询、聚合、分页和长文本分段读取 |
| 标准知识分析 | 已实现 | 说话人解析、多阶段事实抽取、覆盖检查、降级发布、向量入库 |
| 精简知识分析 | 工作区开发中 | 一次主抽取 + 局部问题复核，不做通话级说话人解析 |
| 知识检索 | 已实现 | 精确事实读取与 pgvector 语义检索，支持用户和命名空间隔离 |
| MCP | 框架已实现，未接入运行时 | 有允许列表和只读策略，但 `get_external_tool_providers()` 当前返回空元组 |
| 会话滚动摘要 | 数据字段已预留 | `summary` 可进入上下文，但当前没有自动生成/更新摘要的任务 |
| 长期记忆 | 未实现 | 目录尚未形成实际业务实现 |
| 周评分 `score_week` | 未实现 | 当前只有评分所需的事实标签和证据基础 |
| 认证、租户与权限体系 | 未实现 | 当前主要依赖传入的 `user_id` 做数据过滤，适用于本地演示而非生产 |

## 4. 技术栈

| 层级 | 技术 | 用途 |
| --- | --- | --- |
| 语言与运行时 | Python `>=3.14,<3.15` | 后端、脚本、工作流与测试 |
| API | FastAPI + Uvicorn | REST API、健康检查和静态页面托管 |
| Agent 编排 | LangGraph | 构建有界的 Agent → Tool → Agent 状态图 |
| 模型 SDK | OpenAI Python SDK | 当前聊天与知识抽取使用 Chat Completions 兼容接口 |
| 数据访问 | SQLAlchemy 2 | ORM、事务和查询构建 |
| 数据迁移 | Alembic | 版本化数据库结构 |
| 数据库 | PostgreSQL 17 | 会话、审计、通话、事实和知识文档 |
| 向量检索 | pgvector + HNSW | 512 维向量的余弦距离检索 |
| 本地向量模型 | FastEmbed / `BAAI/bge-small-zh-v1.5` | 默认中文 512 维 embedding |
| 可选向量服务 | OpenAI Embeddings | 通过统一 `EmbeddingProvider` 接口切换 |
| 数据校验 | Pydantic / pydantic-settings | API、工具、模型输出与配置校验 |
| 测试 | pytest + pytest-asyncio + httpx | 单元测试、API/数据库/工作流集成测试 |
| 包管理 | uv | 依赖锁定、虚拟环境和命令运行 |

注意：早期蓝图曾建议 `text-embedding-3-small/1536`，当前实现已经切换为本地 BGE 小模型和固定的 **512 维**数据库列。当前 `AnalyzeCallKnowledgeWorkflow` 会主动拒绝非 512 维 provider，切换模型维度必须先设计新迁移和向量重建方案。

## 5. 总体架构

```text
浏览器（本地聊天页）
        │ REST
        ▼
FastAPI 会话接口 ───────────────► conversations / messages
        │
        ▼
LangGraph ReAct Agent ──────────► agent_runs / tool_calls
        │                            │
        │ 受控 ToolRegistry          └─ 审计每次工具入参、结果、状态
        ▼
┌──────────────────────────────────────────────────────────┐
│ 通话统计工具 │ 原文工具 │ 分析工具 │ 事实读取 │ 向量检索 │
└──────────────────────────────────────────────────────────┘
        │
        ├────────────► calls（原始通话）
        │
        └────────────► 知识分析工作流
                         │
                         ├─ 逐轮解析与说话人处理
                         ├─ 原子事实提取与证据校验
                         ├─ structured_fact / transcript_chunk / full_transcript
                         └─ FastEmbed/OpenAI 向量化
                                      │
                                      ▼
                 call_analysis_runs / call_turns / call_facts
                 documents / document_embeddings（pgvector）
                                      │
                                      ▼
                       规划：规则化评分与周报聚合
```

### 5.1 分层职责

- `api/`：HTTP 路由和请求/响应模型；
- `agent/`：模型适配、上下文、ReAct 图、协议和证据终结；
- `bootstrap/`：把内置工具和未来外部 provider 组合到运行时；
- `features/`：按 `calls`、`knowledge`、`scoring` 聚合各业务能力的契约、查询、仓储、工作流与工具适配；
- `tools/`：只保留跨业务工具契约、参数校验和统一注册表；
- `integrations/`：FastEmbed/OpenAI embedding 与 MCP 等外部技术适配；
- `core/`：全局配置和数据库 Session；
- `repositories/`：跨业务的会话和 Agent 审计数据访问；
- `domain/`：SQLAlchemy 数据模型；
- `mcp/`：外部 MCP 工具的命名空间、允许列表和只读控制；
- `web/`：本地演示页面；
- `scripts/`：通话批量导入、单通分析和并发批分析；
- `alembic/versions/`：数据库结构演进；
- `tests/`：单元与集成测试。

## 6. 目录结构

```text
sales-agent/
├─ src/sales_agent/
│  ├─ agent/                 # ReAct、模型协议、证据账本、最终答案校验
│  ├─ api/                   # FastAPI 应用、会话路由和 Schema
│  ├─ bootstrap/             # 内置/外部工具装配
│  ├─ core/                  # 配置和数据库 Session
│  ├─ domain/                # SQLAlchemy 模型
│  ├─ features/
│  │  ├─ calls/              # 通话导入、查询、仓储和工具
│  │  ├─ knowledge/          # 事实提取、知识入库、检索和工具
│  │  └─ scoring/            # 规则、召回、分段评分、审核返工
│  ├─ integrations/
│  │  ├─ embeddings/         # FastEmbed/OpenAI provider
│  │  └─ mcp/                # MCP provider 与策略
│  ├─ repositories/          # 会话和 Agent 审计数据访问
│  ├─ tools/                 # 通用工具契约与注册表
│  ├─ web/                   # 本地聊天 UI
│  └─ main.py                # Uvicorn 启动入口
├─ scripts/
│  ├─ calls/import_calls.py  # TXT 文件夹预览/导入
│  └─ knowledge/             # 单通与批量知识分析
├─ alembic/versions/         # 0001 至 0009 数据库迁移
├─ tests/unit/
├─ tests/integration/
├─ compose.yaml
├─ alembic.ini
├─ pyproject.toml
├─ uv.lock
└─ .env.example
```

## 7. 核心业务流程

### 7.1 通话导入

1. 递归扫描指定目录中的 `.txt` 文件；
2. 尝试 UTF-8/兼容编码解码并提取有效对话；
3. 优先从父目录推断日期，否则循环分配当前自然周日期；
4. 以文件名 stem 作为 `external_call_id`；
5. 生成来源哈希并保存原始文本和规整后的 transcript；
6. 默认只预览，不写库；显式传入 `--commit` 才提交；
7. `(user_id, external_call_id)` 唯一，已存在记录会跳过。

导入阶段的 `sales_id`、`sales_stage` 和日期可能是人工参数或脚本回退值，`metadata_is_synthetic=true` 用于提醒后续评分不要把推断元数据误当成可靠标签。

### 7.2 标准知识分析流程

标准 `analyze_call` 工作流顺序如下：

1. 依据 `call_id + user_id` 读取通话，阻止跨用户访问；
2. 计算输入指纹。指纹包含来源、数据库文本、解析器、说话人解析器、提取器、文档构建器、向量 provider、模型与维度；
3. 同一命名空间已有相同成功指纹时直接复用，`force=true` 才创建新版本；
4. 将 transcript 解析成有序 `TranscriptTurn`；
5. 调用说话人解析器，把源标签映射为 `sales/customer`；
6. 执行“高召回事实发现 → 事实标注 → 覆盖检查”的多阶段抽取，并进行有限返工；
7. 将事实证据锚定到指定轮次的连续原文；无法精确锚定时标为 `aligned` 或 `turn_only`；
8. 构建三类知识文档并生成 512 维向量；
9. 在同一 `call_id + knowledge_namespace` 内停用旧文档，再原子发布新分析版本；
10. 更新通话和分析运行状态。

标准流程允许**降级成功**：说话人识别或事实抽取失败时，仍尽量发布轮次片段和完整原文，`degraded`、`enrichment_status` 与 `degradation_json` 会记录质量损失。这样检索服务不会因一个模型阶段失败而完全失去原始资料。

### 7.3 精简实验流程

`analyze_call_lean` 使用独立命名空间 `experiment-lean-v1`，目标是降低模型调用次数：

- 跳过通话级说话人映射，由提取器逐事实判断说话人；
- 一次完成事实、阶段、类型、说话人和证据提取；
- 先在本地逐事实验证证据；
- 只把有问题的小范围上下文交给模型复核；
- 与基线命名空间隔离，可针对同一通话并行保留和比较结果。

它目前属于工作区内的实验功能。用于正式评分前，应先对比事实召回率、说话人准确率、证据精确率、耗时和调用成本，而不是仅按“事实数量更多”判断优劣。

### 7.4 知识文档

| 文档类型 | 内容 | 主要用途 |
| --- | --- | --- |
| `structured_fact` | 事实、评分语义标签、置信度、校验状态和原文证据 | 事实检索与后续评分 |
| `transcript_chunk` | 最大约 600 字符的连续对话轮次，默认重叠 1 轮 | 上下文检索和证据回看 |
| `full_transcript` | 完整原始源文本 | 审计、复核和整通语义表示 |

完整文本超过 12,000 字符时，不直接对整篇做 embedding，而是对分块向量求归一化均值，避免 provider 单次输入过长。

### 7.5 对话 Agent

每次发送消息时：

1. 用户消息先写入 `messages`；
2. 读取最近 `CONVERSATION_CONTEXT_MESSAGES` 条 user/assistant 消息，默认 12 条；
3. 如果会话已有摘要，则作为 developer 上下文加入；
4. Agent 读取当前工具定义并选择工具或准备答案；
5. 每次业务工具结果被登记到证据账本，可引用结果获得 `E1`、`E2` 等编号；
6. Agent 必须单独调用 `submit_final_answer`；
7. 业务数据结论要求 `grounding=tool` 且引用真实有效的证据编号；
8. 协议错误最多返工 3 次，业务工具默认最多调用 6 次；
9. 成功后保存 assistant 消息，整个运行与工具轨迹同时写入审计表。

该约束解决的是“回答是否有当前工具证据”，但不等同于事实百分之百正确。重要评分、报价和客户承诺仍应允许人工查看对应原文。

## 8. 当前工具目录

| 工具 | 读写 | 作用 | 关键限制 |
| --- | --- | --- | --- |
| `describe_calls_schema` | 只读 | 返回可查询指标、维度、操作符和限制 | 只描述能力，不返回业务事实 |
| `query_calls` | 只读 | 对当前用户的通话做计数、分组、筛选、排序和分页 | 当前唯一指标是 `call_count` |
| `get_call_content` | 只读 | 按 `call_id` 分段读取 transcript | 单次 500～12,000 字符，跨用户不可见 |
| `analyze_call` | 本系统写入 | 执行标准知识分析并发布知识 | 默认 `baseline-react-v4`，支持指纹复用 |
| `analyze_call_lean` | 本系统写入 | 执行精简实验分析 | 默认 `experiment-lean-v1` |
| `get_call_facts` | 只读 | 读取某通最新/指定分析运行的结构化事实 | 默认只返回 `verified`，精确过滤值不可猜测 |
| `search_call_knowledge` | 只读 | 用自然语言语义召回结构化事实 | 单次只查一个命名空间，默认相似度阈值 0.25 |
| `score_single_call_whiteboard` | 只读 | 先判定通话层级，再由适用模块并行完成14维白板评分和证据复核 | 不读取知识库或其他通话；TXT无法判断语速语调 |

`query_calls` 支持的维度为 `call_date`、`sales_id`、`sales_stage`、`analysis_status`、`call_id`；可筛选前四项，但禁止按原始文本、哈希和 `user_id` 查询。该语义层是安全边界，不能绕过它让模型拼 SQL。

## 9. API 说明

### 9.1 页面与健康检查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 返回本地聊天页面 |
| `GET` | `/health` | 进程存活检查 |
| `GET` | `/health/database` | 数据库连接检查，不可用时返回 503 |
| `GET` | `/health/model` | 返回模型是否配置和模型名，不暴露密钥 |
| `GET` | `/docs` | FastAPI 自动生成的 Swagger 文档 |

### 9.2 会话接口

| 方法 | 路径 | 输入/查询 | 返回 |
| --- | --- | --- | --- |
| `POST` | `/api/conversations` | `user_id`、可选 `title` | 新会话 |
| `GET` | `/api/conversations` | `user_id`、`limit=1..100` | 该用户的会话列表 |
| `GET` | `/api/conversations/{id}` | 会话 UUID | 会话详情 |
| `GET` | `/api/conversations/{id}/messages` | `limit=1..100` | 按 sequence 正序的历史消息 |
| `POST` | `/api/conversations/{id}/messages` | `content`，1～20,000 字符 | 本轮 user 与 assistant 两条消息 |

创建会话示例：

```powershell
$body = @{ user_id = "local-demo-user"; title = "本周通话复盘" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/conversations `
  -ContentType "application/json" -Body $body
```

发送消息示例：

```powershell
$conversationId = "替换为会话 UUID"
$body = @{ content = "销售员 001 本周一共有多少通电话？" } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/conversations/$conversationId/messages" `
  -ContentType "application/json" -Body $body
```

当前接口没有独立的鉴权中间件：创建和列出会话依赖客户端传 `user_id`，会话详情也尚未额外验证调用者身份。生产化前必须由认证令牌解析用户/租户，并让所有会话、消息、工具和评分查询使用服务端身份作用域。

## 10. 数据模型与关系

### 10.1 会话与审计

| 表 | 关键字段 | 说明 |
| --- | --- | --- |
| `conversations` | `id, user_id, title, summary, summary_through_message_id` | 会话元数据和摘要覆盖位置 |
| `messages` | `conversation_id, role, content, sequence_no` | 有序且追加式的消息历史 |
| `agent_runs` | `conversation_id, status, graph_state, started_at, finished_at` | 每轮 Agent 执行摘要 |
| `tool_calls` | `run_id, tool_name, arguments_json, result_json, status` | 工具审计；已预留 `approval_status` |

### 10.2 通话与知识

| 表 | 关键字段 | 说明 |
| --- | --- | --- |
| `calls` | `user_id, external_call_id, sales_id, call_date, transcript_text, source_hash, analysis_status` | 原始业务通话；用户内 external ID 唯一 |
| `call_analysis_runs` | 版本指纹、各组件版本、namespace、质量状态、错误 | 每次分析运行，可复用、失败或降级 |
| `call_turns` | `analysis_run_id, turn_no, source_speaker_label, speaker_role, text` | 可定位的逐轮原文 |
| `call_facts` | `phase, fact_type, speaker, fact_text, confidence, validation_status, score_tags, evidence_json` | 评分和检索最重要的结构化输入 |
| `documents` | `doc_type, source_key, content, metadata_json, is_active, knowledge_namespace` | 可检索文档与版本激活状态 |
| `document_embeddings` | 512 维向量、provider/model/version/strategy | 向量索引数据 |

主要关系：一条 `Call` 可以有多个 `CallAnalysisRun`；每个 run 拥有自己的 turns、facts 和 documents；每个 document 对应一个 embedding。新 run 发布时只停用同一通话、同一知识命名空间的旧 documents，因此基线和实验知识可同时存在。

### 10.3 迁移历史

| 迁移 | 作用 |
| --- | --- |
| `0001` | 会话、消息、Agent 运行、工具调用及 `pgcrypto` |
| `0002` | 通话表 |
| `0003` | 用户 + 日期索引 |
| `0004` | 分析运行、轮次、事实、文档、向量与 `vector` 扩展 |
| `0005` | 说话人解析版本与映射 |
| `0006` | 向量列切换为 BGE 512 维并重建 HNSW |
| `0007` | 降级状态、事实置信度和质量问题 |
| `0008` | 事实校验状态 + 置信度索引 |
| `0009` | 知识命名空间及相关索引 |

当前 Alembic head 为 `0009_knowledge_namespaces`。

## 11. 事实体系与评分数据基础

事实阶段固定为：

- `opening`：开场、业务介绍和承接上下文；
- `discovery`：提问、需求、现状、画像、痛点、预算、意向和障碍；
- `proposal_negotiation`：价值表达、产品建议、价格优惠、异议和谈判；
- `fulfillment_support`：合同、交付、履约和售后；
- `relationship_closing`：同理心、关系推进、微信、下一步和礼貌收尾；
- `unknown`：证据明确但阶段无法可靠判断。

推荐的稳定事实类型包括：`business_introduction`、`prior_context_bridge`、`sales_discovery_question`、`customer_need`、`customer_profile`、`customer_current_state`、`customer_pain_point`、`customer_barrier`、`customer_budget`、`customer_intention`、`customer_objection`、`value_proposition`、`product_recommendation`、`price_or_promotion`、`objection_response`、`unresolved_objection`、`contract_or_delivery`、`after_sales_solution`、`empathy_or_relationship`、`wechat_agreement`、`next_step`、`polite_closing` 和 `other_business_fact`。

其中：

- `phase` 与 `fact_type` 是相对稳定、适合规则判断的字段；
- `score_tags` 是模型生成的中文软标签，适合检索、候选规则和分析，不应直接作为唯一计分依据；
- `evidence_json` 是评分解释的证据来源；
- `confidence`、`validation_status` 和 `quality_issues` 决定事实是否可用于正式评分；
- `knowledge_namespace` 决定评分读取哪一套抽取结果，禁止一份报告混用不同命名空间。

## 12. 评分模块设计（下一阶段规划）

### 12.1 评分目标

评分模块应回答四类问题：

1. **单通质量**：一通电话中销售完成了哪些关键动作、遗漏了哪些可执行动作；
2. **周度表现**：某销售员在一个自然周内的综合得分、稳定性和趋势；
3. **证据解释**：每个得分/扣分能定位到通话、轮次和原文；
4. **改进建议**：从低分维度和重复问题生成优先级明确的训练建议。

评分用于教练、质检和复盘，不应直接等价于绩效工资。正式用于绩效前需要业务负责人确认评分项、权重、适用条件、申诉流程和抽样审计制度。

### 12.2 推荐评分维度（v1 草案）

| 维度 | 权重 | 主要正向证据 | 主要风险/扣分证据 |
| --- | ---: | --- | --- |
| 开场与上下文承接 | 10 | `business_introduction`、`prior_context_bridge` | 未说明来意、机械开场、上下文断裂 |
| 需求发现 | 25 | `sales_discovery_question`，以及由销售提问引出的需求/现状/痛点/预算/意向 | 只讲产品、不提问；关键需求信息长期缺失 |
| 方案与价值匹配 | 20 | `value_proposition`、`product_recommendation`，且与已发现需求可建立证据关系 | 泛泛介绍、推荐与客户需求无关 |
| 异议处理 | 15 | `customer_objection` 后出现有针对性的 `objection_response` | `unresolved_objection`；回避、强压或答非所问 |
| 价格、合同与履约说明 | 10 | `price_or_promotion`、`contract_or_delivery`、`after_sales_solution` 且表达清楚 | 承诺不清、关键条件遗漏或互相冲突 |
| 沟通与关系推进 | 10 | `empathy_or_relationship`、有效确认和客户反馈 | 打断、忽视客户表达；仅礼貌用语不能拿满分 |
| 下一步闭环 | 10 | `wechat_agreement`、明确的 `next_step`，包含对象/动作/时间中的至少两项 | 没有明确行动，或只说“以后再联系” |

总权重为 100。此表是基于当前事实类型给出的可实现草案，业务方确认后应保存为版本化 policy，不能硬编码在提示词中。

### 12.3 机会感知：避免机械扣分

并非每通电话都具备所有评分机会。例如客户没有提出异议时，不能因没有 `objection_response` 扣分；短暂的预约确认电话也不应按完整销售演示标准评分。因此每个维度需要先判断 `applicable`：

- 客户出现 `customer_objection` 时，异议处理维度适用；
- 通话进入报价/方案阶段时，价格或履约维度才可能适用；
- 新客户首次触达通常适用开场维度，已有关系的跟进应更关注上下文承接；
- 纯售后通话应使用售后型 rubric，不能直接套售前型 rubric；
- 信息量过低、转语音信箱或有效轮次不足的电话应标为 `not_scorable`，不应用零分拉低周均值。

建议每通先选择 rubric（如 `sales_first_contact_v1`、`sales_follow_up_v1`、`after_sales_v1`），再计算适用维度。若 v1 暂时只有一个 rubric，也必须保留 `applicable` 和 `not_applicable_reason` 字段。

### 12.4 确定性计分方式

推荐由规则引擎读取事实，而不是让模型直接给 0～100：

```text
维度得分率 = 获得的规则点数 / 该通话适用规则的最大点数
单通得分   = Σ(维度得分率 × 维度权重) / Σ(适用维度权重) × 100
周度得分   = Σ(单通得分 × 通话权重) / Σ(通话权重)
```

建议第一版让 `通话权重=1`，避免长通话天然支配结果。后续如要按有效时长或业务阶段加权，应单独发布新评分 policy 版本。

规则示例：

| 规则 | 判断方式 | 示例点数 |
| --- | --- | ---: |
| 明确业务介绍 | 有 verified `business_introduction` | +3 |
| 承接历史上下文 | 有 verified `prior_context_bridge` | +2 |
| 主动提问 | 至少 1 条销售 `sales_discovery_question` | +3 |
| 多层需求发现 | 需求/现状/痛点/预算中覆盖至少 3 类 | +5 |
| 需求驱动的产品推荐 | 推荐事实与客户需求证据同时存在且时序合理 | +5 |
| 完成一次异议回应 | 每个明确异议后存在对应回应 | +3/次，上限受控 |
| 异议仍未解决 | 有 verified `unresolved_objection` | -2/次，下限受控 |
| 明确下一步 | 有 `next_step`，且包含动作与时间/对象 | +5 |

精确分值、阈值和上限需要用人工标注样本校准。规则结果必须记录命中的 `fact_id` 和证据，不能只保存最终总分。

### 12.5 质量与置信度策略

正式评分建议默认：

- 只读取 `validation_status=verified` 的事实；
- `low_confidence` 事实不直接加减分，只进入 `review_flags`；
- `analysis_run.degraded=true` 时仍可生成报告，但降低报告可信度并提示人工复核；
- `enrichment_status=skipped` 或事实为空时不应输出确定性低分，应返回 `insufficient_evidence`；
- 没有声学分析输入时，音频规则返回 `not_applicable`，不参与评判覆盖率；
- `metadata_is_synthetic=true` 时，销售员、日期、阶段等维度要显示数据来源警告；
- 评分使用的 embedding/检索仅用于找证据，不应通过相似度直接换算分数；
- 一条事实即便有多个 `score_tags`，也要通过规则去重，防止重复加分。

报告需要拆分展示三种不同质量概念：

```text
reliability             = 知识分析运行质量（完整 / 降级 / 缺失）
assessment_coverage     = 已完成判断的适用规则数 / 适用规则总数
evidence_reliability    = 实际命中证据的置信度与原文对齐质量
```

不得因为某条规则没有进入周期级 Top-K 召回，就把整通电话标成低可靠；
`insufficient_evidence` 表示评判覆盖不足，不代表已有 verified 事实不可信。
模型审核只复核“证据是否满足规则必要条件”，不得重新质疑 verified 事实本身；
审核返工必须限定到被点名的 `(call_id, rule_id)`，未被质疑的得分继续保留并单独汇总为无争议分。

### 12.6 周范围与聚合口径

`score_week` 建议输入：

```json
{
  "sales_id": "001",
  "week_start": "2026-09-07",
  "timezone": "Asia/Shanghai",
  "knowledge_namespace": "baseline-react-v4",
  "score_policy_version": "sales-general-v1",
  "include_low_confidence": false
}
```

约束：

- `week_start` 必须是业务时区的周一；范围采用 `[week_start, week_start + 7天)`；
- 当前 `calls.call_date` 是 date，没有通话发生时刻，因此时区主要用于定义业务口径和未来兼容；
- 必须用服务端认证得到的 `user_id` 过滤，不能由模型自行指定其他用户；
- 一次报告固定一个 namespace 和 policy version；
- 同一 call 在该 namespace 中只取最新成功、有效的 analysis run；
- 未分析、失败、降级、不可评分的通话分别计数并展示，不能静默丢弃；
- 当可评分样本过少时返回结果但标记低可靠性，不做强结论或排名。

建议输出：

```json
{
  "sales_id": "001",
  "week_start": "2026-09-07",
  "week_end_exclusive": "2026-09-14",
  "score": 82.4,
  "reliability": "medium",
  "policy_version": "sales-general-v1",
  "knowledge_namespace": "baseline-react-v4",
  "call_counts": {
    "total": 18,
    "scored": 14,
    "not_analyzed": 2,
    "not_scorable": 1,
    "degraded": 1
  },
  "dimensions": [],
  "strengths": [],
  "improvements": [],
  "evidence": [],
  "generated_at": "..."
}
```

### 12.7 评分数据表建议

不要把评分结果塞入 `call_facts`。建议新增：

| 表 | 关键字段 | 用途 |
| --- | --- | --- |
| `score_policies` | `id, name, version, rubric_json, status, effective_from` | 版本化权重、规则、适用条件 |
| `score_runs` | `id, user_id, sales_id, period_start, period_end, policy_version, namespace, input_fingerprint, status` | 一次周评分运行与幂等键 |
| `call_scores` | `score_run_id, call_id, analysis_run_id, rubric, total_score, scorable, reliability` | 单通得分和输入版本 |
| `score_items` | `call_score_id, dimension, rule_id, points, max_points, applicable, fact_refs, evidence_json` | 最细粒度的规则命中和解释 |
| `weekly_scorecards` | `score_run_id, total_score, dimension_json, summary_json, generated_at` | 周报聚合快照 |

`input_fingerprint` 至少包含通话集合、各 `analysis_run_id`、policy version、namespace 和评分引擎版本。输入未变化时复用旧结果；任何事实或策略变化都创建新 score run，保留历史报告以便审计。

### 12.8 评分服务与工具边界

当前评分工作流位于 `features/scoring/`；后续接入持久化和 Agent 工具时建议新增：

- `features/scoring/repository.py`：评分结果持久化；
- `features/scoring/tool.py`：向 Agent 暴露 `score_week` 与可选 `get_score_details`；
- `tests/integration/scoring/`：用户隔离、版本、幂等和证据链。

`score_week` 第一版应是只读语义上的业务工具：它可以创建本系统内部的评分快照，但不得触发 CRM 等外部写入。若范围中存在未分析通话，推荐返回缺口并由用户决定是否运行批分析，不要在一次 Agent 工具调用中隐式分析大量通话导致超时和不可控成本。

### 12.9 评分展示建议

周报页面至少包含：

- 总分、可靠性、评分通话数和数据缺口；
- 七个维度的分数、周环比和样本量；
- 3 条优势与 3 条优先改进项；
- 每条结论对应的通话 ID、日期、轮次和短原文；
- 降级分析、低置信度事实和合成元数据警告；
- 可下钻到单通评分项和完整原文；
- policy version、knowledge namespace、score run ID 与生成时间。

模型可以把结构化评分结果改写成自然语言复盘，但总分、维度分、证据引用和数据缺口必须来自评分工具，不能由模型自行修改。

## 13. 评分模块实施顺序与验收标准

### 阶段 S0：规则确认与标注集

- [ ] 业务方确认通话类型、七个维度、权重和适用条件；
- [ ] 选取覆盖高/中/低质量及异常情况的真实脱敏通话；
- [ ] 两名业务人员独立标注，解决分歧并形成金标；
- [ ] 定义不可评分、低可靠性和人工复核条件；
- [ ] 固化 `sales-general-v1` policy 文件。

验收：同一份 policy 可被机器校验；金标包含总分、维度分、规则命中与原文证据，而不只是一个总分。

### 阶段 S1：单通确定性评分

- [ ] 新增评分契约与纯规则引擎；
- [ ] 只消费指定 analysis run 的结构化事实；
- [ ] 实现适用性判断、去重、上下限和证据绑定；
- [ ] 覆盖空事实、低置信度、降级运行和多异议场景；
- [ ] 输出稳定、可序列化的 `CallScore`。

验收：同一输入与 policy 必须得到完全一致的结果；每个非零分项都有 fact/evidence 引用；无评分机会的维度不会被错误扣分。

### 阶段 S2：周评分工作流与存储

- [ ] 新增评分迁移和 repository；
- [ ] 按用户、销售员、自然周筛选通话；
- [ ] 选择同一 namespace 中最新成功分析；
- [ ] 实现幂等指纹、单通评分、周聚合和可靠性；
- [ ] 显式返回未分析、失败、不可评分和降级数量。

验收：跨用户数据绝不进入报告；重复执行可复用；变更 policy 或任一 analysis run 后生成新版本。

### 阶段 S3：Agent 工具与 API

- [ ] 注册 `score_week` 工具；
- [ ] 新增周报读取 API 和单通下钻 API；
- [ ] 提示词要求评分结论必须引用工具证据；
- [ ] 在 Web 端增加周评分入口、维度展示和证据下钻；
- [ ] 大批量处理改为后台任务或显式脚本，不占用同步聊天请求。

验收：用户可以问“001 本周表现如何”，Agent 自动选择评分工具；回答中的数字与持久化报告一致；点击证据可定位原文。

### 阶段 S4：校准、评测与上线

- [ ] 对照人工金标计算维度一致率、平均绝对误差和证据准确率；
- [ ] 对不同通话类型、销售员、时长和数据质量做切片评估；
- [ ] 评估基线/精简知识流程对评分的影响；
- [ ] 加入 policy 发布、回滚和历史报告对比；
- [ ] 建立申诉、人工复核、审计抽样和脱敏策略。

建议门槛：规则证据引用准确率优先于总分相关性；若证据不可靠，即使总分看似接近人工评分也不应上线。

## 14. 本地运行

### 14.1 前置条件

- Windows PowerShell（当前开发环境）；
- Python 3.14；
- uv；
- Docker Desktop / Docker Compose；
- 可用的 OpenAI Chat Completions 兼容 API Key；
- 首次使用 FastEmbed 时需要能够下载本地模型。

### 14.2 初始化配置

```powershell
Set-Location C:\projectOneV2\sales-agent
Copy-Item .env.example .env
```

至少修改 `.env` 中的 `POSTGRES_PASSWORD`、`DATABASE_URL` 和 `OPENAI_API_KEY`。不要提交 `.env`；健康接口只显示是否已配置，不会回传密钥。

### 14.3 安装依赖、启动数据库和迁移

```powershell
uv sync --group dev
docker compose up -d postgres
uv run alembic upgrade head
```

检查：

```powershell
docker compose ps
uv run alembic current
```

### 14.4 启动应用

```powershell
uv run sales-agent-api
```

浏览器访问 `http://127.0.0.1:8000/`；API 文档访问 `http://127.0.0.1:8000/docs`。

### 14.5 导入通话

先 dry-run：

```powershell
uv run python scripts/calls/import_calls.py C:\path\to\calls `
  --user-id local-demo-user --sales-id 001 --sales-stage 销售线索
```

确认统计和跳过列表后再写入：

```powershell
uv run python scripts/calls/import_calls.py C:\path\to\calls --commit `
  --user-id local-demo-user --sales-id 001 --sales-stage 销售线索
```

### 14.6 分析通话

单通标准分析：

```powershell
uv run python scripts/knowledge/analyze_call.py <call-uuid> --user-id local-demo-user
```

批量标准分析：

```powershell
uv run python scripts/knowledge/analyze_calls_batch.py --sales-id 001 `
  --workflow standard --workers 5
```

批量精简实验分析：

```powershell
uv run python scripts/knowledge/analyze_calls_batch.py --sales-id 001 `
  --workflow lean --namespace experiment-lean-v1 --workers 5
```

可先增加 `--dry-run` 核对选择范围。`--all` 不能和 `--sales-id/--call-date` 同时使用；`--force` 会创建新分析版本，应谨慎用于重算。

### 14.7 大模型严格 JSON 协议

所有会生成业务数据的大模型调用统一使用 Chat Completions Structured Outputs：

- `response_format.type=json_schema` 且 `strict=true`，不再使用仅保证 JSON 可解析的 `json_object`。
- 每种响应都由 Pydantic 模型生成唯一 Schema，并要求顶层固定 `schema_version`。
- 所有对象字段均为必填项、`additionalProperties=false`，带本地默认值的字段也必须由模型明确返回。
- Agent 函数调用统一设置 `strict=true`，工具参数在发送前递归闭合并进行本地 Schema 校验。
- 提示词不再重复传输 `output_schema`；结构正确性由 API 协议保证。本地 Pydantic 与业务证据校验仍作为信任边界，返工主要处理证据、语义和跨字段约束。
- 若兼容网关不支持严格 Structured Outputs，请求会明确失败，不会静默降级到宽松 JSON，以免把结构错误带入知识库或评分流程。

Embedding 接口返回的是 SDK 定义的向量数组，不生成模型文本，因此不适用 Structured Outputs。

### 14.8 Agent 解耦边界

- `ReActAgent` 是无数据库核心：仅依赖 `ChatModel`、`ToolRuntime` 和可选的 `RunAuditSink`。
- 默认 `NullRunAuditSink` 支持无数据库运行；生产环境在组合根注入 `SqlAlchemyRunAuditSink`。
- 工具注册表向核心提供传输无关的 `ToolSpec`；Chat Completions 格式转换仅存在于 OpenAI 适配器。
- `ConversationMemory` 统一负责消息追加、上下文加载和摘要游标保存；SQLAlchemy 实现位于仓储层。
- 加载记忆时只返回 `summary_through_message_id` 之后的消息，避免摘要内容与原消息重复进入上下文。

## 15. 配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | 必填 | SQLAlchemy PostgreSQL 连接串 |
| `OPENAI_API_KEY` | 空 | 聊天、说话人和事实提取所需 |
| `OPENAI_BASE_URL` | 空 | OpenAI 兼容服务地址 |
| `CHAT_MODEL` | `gpt-5-mini` | 聊天与默认知识模型 |
| `SCORING_MODEL` | 空 | 评分工具专用模型；空时回退到 `CHAT_MODEL` |
| `SCORING_MAX_OUTPUT_TOKENS` | `8192` | 周期级评分与审核的最大输出 token |
| `SCORING_TIMEOUT_SECONDS` | `180` | 单次评分或审核模型请求超时秒数 |
| `SCORING_MAX_RETRIES` | `3` | 评分请求遇到连接错误、超时、429 或 5xx 时的最大重试次数 |
| `SCORING_RETRY_BASE_SECONDS` | `1` | 评分请求指数退避的基础秒数（自动增加抖动） |
| `SCORING_RETRY_MAX_SECONDS` | `10` | 单次评分重试等待的最大秒数 |
| `CHAT_MAX_OUTPUT_TOKENS` | `1200` | 对话模型最大输出 |
| `CONVERSATION_CONTEXT_MESSAGES` | `12` | 每轮读取的最近消息数 |
| `AGENT_MAX_STEPS` | `6` | 每轮最大业务工具次数 |
| `AGENT_FINAL_ANSWER_MAX_REPAIRS` | `3` | 最终答案协议最大返工次数 |
| `FACT_EXTRACTION_MAX_OUTPUT_TOKENS` | `4096` | 事实抽取输出上限 |
| `KNOWLEDGE_MAX_REPAIRS` | `3` | 知识模型结构/业务返工上限 |
| `KNOWLEDGE_EVALUATOR_MODEL` | 空 | 空时复用聊天模型 |
| `KNOWLEDGE_EVALUATION_MAX_OUTPUT_TOKENS` | `2400` | 知识评估输出上限 |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` 或 `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 当前默认向量模型 |
| `EMBEDDING_DIMENSIONS` | `512` | 必须与数据库 Vector(512) 一致 |
| `EMBEDDING_CACHE_DIR` | `.models/fastembed` | 本地模型缓存目录 |
| `EMBEDDING_THREADS` | `0` | 0 表示使用 backend 默认值 |
| `EMBEDDING_API_KEY` | 空 | 未提供时可回退使用 OpenAI key |
| `EMBEDDING_BASE_URL` | 空 | 独立 embedding 服务地址 |

## 16. 测试与质量状态

常用命令：

```powershell
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run python -m compileall src
```

截至本文整理时：

- pytest 可收集 **76** 个测试；
- 源码 `compileall` 通过；
- Alembic 迁移链只有一个 head：`0009_knowledge_namespaces`；
- 本次受限执行中有 51 个单元测试通过，3 个使用 `tmp_path` 的导入测试因运行环境拒绝创建临时目录而报 setup error，并非断言失败；
- 集成测试需要 `.env` 中的 PostgreSQL 测试连接及完成迁移的数据库，不能把“成功收集”误写成“全部集成测试已通过”。

测试覆盖重点包括：会话 API、上下文窗口、Agent 工具循环、证据引用修复、最大返工、用户隔离、安全语义查询、原文分页、知识工作流的成功与降级、事实证据锚定、说话人识别、基线/精简命名空间隔离、MCP 只读策略和 embedding 维度。

## 17. 已知限制与风险

1. **评分尚未实现**：`score_tags` 不是最终评分结果，也没有 `score_week` 工具或评分表；
2. **身份认证不足**：Web 固定 `local-demo-user`，API 未接正式认证；
3. **会话详情作用域不足**：仅用 conversation UUID 查询，生产前需绑定认证用户；
4. **摘要未自动维护**：长会话超过最近消息窗口后，早期内容可能丢失；
5. **同步请求耗时**：聊天接口内同步运行模型和工具，长分析可能导致 HTTP 超时；
6. **工具超时/重试未统一**：蓝图中的单工具 20 秒超时尚未成为统一 runtime 机制；
7. **MCP 未装配**：provider 代码存在，但运行时没有实际外部 provider；
8. **向量维度固定**：更换到非 512 维模型不能只改环境变量；
9. **模型输出仍可能降级**：系统能记录质量问题，但不能消除所有抽取误差；
10. **软标签漂移**：中文 `score_tags` 未使用固定枚举，跨模型/版本可能出现同义标签；
11. **元数据可能是合成值**：导入脚本的日期、销售员、阶段必须在评分前验证来源；
12. **前端仅为本地原型**：没有登录、评分视图、流式响应、取消运行和细粒度错误详情；
13. **工作树含未提交功能**：精简流程、降级事实和命名空间相关改动应完成评审与提交后再视为稳定基线。

## 18. 安全、隐私与审计要求

- `.env`、API Key、数据库密码不得提交；
- 日志记录模型、耗时、状态码和 request ID，但不得记录密钥；
- 工具错误对 Agent 和用户只返回稳定的公开错误，不暴露异常栈；
- 所有业务读取必须强制绑定服务端 `user_id/tenant_id`；
- 通话原文属于敏感业务数据，生产日志和评测集需要脱敏；
- `messages`、`tool_calls`、analysis run、score run 应保留可追踪版本；
- 外部系统创建、更新、删除和发送操作必须进入审批节点；
- 评分报告需记录 policy、namespace、analysis run 和证据，支持人工申诉与重算；
- 对低可靠性或数据不足的评分必须显式标注，禁止伪造精确结论。

## 19. 推荐近期开发优先级

1. 整理并提交当前精简提取、降级发布和知识命名空间改动；
2. 在真实 PostgreSQL 环境跑通全部 76 个测试，并修复临时目录执行环境问题；
3. 为 API 接入认证身份，补齐会话/消息的用户作用域；
4. 建立评分金标集并确认 `sales-general-v1` 规则；
5. 先实现纯函数式单通评分，不接 Agent；
6. 实现版本化评分表、周聚合和幂等重算；
7. 再注册 `score_week` 工具并做证据化自然语言周报；
8. 最后增加评分 UI、趋势、人工复核和 policy 管理。

## 20. 关键设计原则总结

- **原文优先**：事实、检索和评分都能追溯到通话轮次；
- **规则算分，模型解释**：模型负责抽取和表述，确定性引擎负责分数；
- **不把缺失当负面证据**：先判断是否存在评分机会；
- **默认信任 verified**：低置信度事实进入复核，不直接影响正式分数；
- **全链路版本化**：模型、抽取器、文档、向量、namespace 和评分 policy 都进入指纹；
- **用户边界前置**：查询条件中自动加入用户作用域，而不是依赖模型记住；
- **降级可见**：可以部分成功，但质量损失必须进入结果和报告；
- **实验隔离**：标准和精简流程通过 namespace 对比，不能互相覆盖；
- **审计优先于“看起来聪明”**：任何无法解释来源的高分、低分或业务结论都不应成为正式输出。
