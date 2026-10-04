# 运维脚本

- `calls/import_calls.py`：预览或提交 TXT 通话导入。
- `knowledge/analyze_call.py`：运行单通标准知识分析。
- `knowledge/analyze_calls_batch.py`：按条件批量运行标准或精简知识分析。
- `scoring/run_score.py`：按销售和日期范围分别使用标准/精简知识命名空间评分并输出对比；按行为表的十四个关键行为分别对整个周期统一召回，每条语义行为最多一次无历史模型调用。可用 `--sample-limit` 做小样本冒烟测试。
- `scoring/score_call_text.py`：直接读取一份单通电话 TXT，先判定业务层级，再让适用评分模块并行完成14维判定，经独立证据复核后进行本地确定性计分；不读取数据库、知识库、周期评分或聊天记忆。
- `scoring/preview_retrieval.py`：只检查整个周期的向量召回数量、来源和通话覆盖，不调用大模型、不输出原文。

脚本只负责参数解析和调用工作流，不应包含核心业务规则。

单通 TXT 白板评分示例：

```powershell
uv run python scripts/scoring/score_call_text.py C:\path\to\call.txt `
  --sales-stage 销售线索 `
  --output artifacts/scoring/single-call-score.json
```

TXT 每个非空行使用 `用户0：话术内容`、`用户1：话术内容` 格式；也支持行尾通话时间。输出先给出 `call_level` 和实际执行的 `executed_modules`，随后为全部14维生成 `hit`、`miss`、`not_applicable`、`unassessable` 或 `review_rejected` 白板状态。纯文本无法可靠判断语速、音量、语调及普通话标准度，因此该维度计0分。

双知识库评分示例：

```powershell
uv run python scripts/scoring/run_score.py `
  --user-id local-demo-user `
  --sales-id 001 `
  --date-from 2026-03-02 `
  --date-to 2026-03-06 `
  --summary
```

默认依次运行 `baseline-react-v4` 和 `experiment-lean-v1`，两套证据不会混用。小样本检查可增加 `--sample-limit 2`。仅运行标准知识库时增加：

评分段、审核段和返工段都不读取聊天记录或 ReAct 轨迹。十二条文本语义行为各自评分，互动质量在本地确定性计算，语速语调没有声学证据时不调用模型；之后一次整体审核。完整周正常为 13 次模型调用，只有审核失败时才按问题行为补召回和返工；默认由 `--max-model-calls 18` 限制总调用量。

评分模型与聊天 Agent 解耦：可在 `.env` 设置 `SCORING_MODEL`，也可在单次运行时传入 `--model <服务商已启用的模型名>`。命令行参数只影响本次评分。

正式批量评分建议使用 `--output artifacts/scoring/<文件名>.json` 保存完整结果；不要设置 `--sample-limit` 即会处理日期范围内全部通话。

```powershell
--namespace baseline-react-v4
```
