# Sales Agent 目录导航

本项目采用“通用 Agent 内核 + 业务能力垂直切片 + 外部适配器”的目录结构。定位代码时先判断它属于通用编排、某个业务能力，还是外部基础设施。

```text
sales-agent/
├─ src/sales_agent/
│  ├─ agent/                  # 通用 Agent 循环、提示词、证据账本、模型协议
│  ├─ api/                    # HTTP 路由、请求响应 Schema、Web 应用入口
│  ├─ bootstrap/              # 组合根：把 Agent、业务工具和外部 Provider 装配起来
│  ├─ core/                   # 全局配置和数据库连接
│  ├─ domain/                 # 跨业务共用的 ORM 实体
│  ├─ features/               # 可独立定位的业务能力
│  │  ├─ calls/               # 通话导入、查询语义、仓储和 Agent 工具
│  │  ├─ knowledge/           # 说话人、事实提取、知识入库、检索和工具
│  │  └─ scoring/             # 评分规则、召回、评分 Agent、审核返工和工作流
│  ├─ integrations/           # 外部技术适配
│  │  ├─ embeddings/          # FastEmbed/OpenAI 向量 Provider
│  │  └─ mcp/                 # MCP Provider 与权限策略
│  ├─ repositories/           # 仅保留跨业务会话和 Agent 审计仓储
│  ├─ tools/                  # 仅保留通用工具契约和注册表
│  ├─ web/                    # 静态聊天页面
│  └─ main.py                 # 服务启动入口
├─ scripts/
│  ├─ calls/                  # 通话导入运维脚本
│  └─ knowledge/              # 单通/批量知识分析脚本
├─ tests/
│  ├─ unit/                   # 按 agent/api/calls/knowledge/scoring/integrations 分区
│  └─ integration/            # 按 agent/api/calls/knowledge/core 分区
└─ alembic/                   # 数据库迁移
```

## 目标文件速查

| 想修改的能力 | 入口文件 |
| --- | --- |
| Agent 工具循环 | `src/sales_agent/agent/react.py` |
| Agent 审计端口 | `src/sales_agent/agent/audit.py` |
| 会话记忆端口 | `src/sales_agent/agent/memory.py` |
| Agent 系统提示词 | `src/sales_agent/agent/prompts.py` |
| 大模型严格 JSON 协议 | `src/sales_agent/agent/structured_json.py` |
| 工具通用契约与注册 | `src/sales_agent/tools/contracts.py`、`registry.py` |
| 工具运行时装配 | `src/sales_agent/bootstrap/tools.py` |
| 通话导入解析 | `src/sales_agent/features/calls/import_workflow.py` |
| 通话统计查询 | `src/sales_agent/features/calls/query.py`、`repository.py` |
| 通话查询工具 | `src/sales_agent/features/calls/tool.py` |
| 标准知识提取 | `src/sales_agent/features/knowledge/fact_extractor.py` |
| 精简知识提取 | `src/sales_agent/features/knowledge/lean_fact_extractor.py` |
| 知识入库工作流 | `src/sales_agent/features/knowledge/standard_workflow.py`、`lean_workflow.py` |
| 知识查询与向量检索 | `src/sales_agent/features/knowledge/query.py`、`repository.py` |
| 知识 Agent 工具 | `src/sales_agent/features/knowledge/standard_tool.py`、`lean_tool.py`、`read_tools.py` |
| 评分规则 | `src/sales_agent/features/scoring/policy.py` |
| 评分工作流 | `src/sales_agent/features/scoring/workflow.py` |
| 单通分层并行白板评分与证据复核 | `src/sales_agent/features/single_call_scoring/` |
| 评分模型代理 | `src/sales_agent/features/scoring/model_agents.py` |
| 评分证据召回 | `src/sales_agent/features/scoring/retrieval.py` |
| Embedding 实现 | `src/sales_agent/integrations/embeddings/` |
| MCP 接入 | `src/sales_agent/integrations/mcp/` |
| ORM 数据表 | `src/sales_agent/domain/models.py` |
| SQLAlchemy 审计/记忆适配器 | `src/sales_agent/repositories/agent_audit.py`、`conversation_memory.py` |
| API | `src/sales_agent/api/` |

## 依赖方向

```text
api / scripts
    -> bootstrap / feature factories
        -> agent + feature workflows
            -> feature contracts/repositories
                -> domain + core

integrations -> feature contracts
feature tool adapters -> common tools contracts
```

约束：

- `agent/` 不依赖具体销售业务工具；工具通过运行时协议注入。
- `ReActAgent` 不依赖 SQLAlchemy、ORM 或仓储；运行审计通过 `RunAuditSink` 注入。
- 工具目录在内核中使用 `ToolSpec`，只有 OpenAI 模型适配器负责转换 Chat Completions JSON。
- 对话历史与摘要通过 `ConversationMemory` 管理；摘要游标之前的消息不会重复进入模型上下文。
- `tools/` 不放具体业务工具，只放跨业务的契约和注册机制。
- 一个业务功能的规则、查询、仓储、工作流和工具适配优先放在同一个 `features/<name>/` 下。
- `integrations/` 只处理外部技术细节，不承载销售业务规则。
- `scripts/` 是人工运维入口，不承载核心逻辑；核心逻辑必须调用 feature workflow。
- 测试目录按源码业务边界镜像，新增测试放到对应子目录。
