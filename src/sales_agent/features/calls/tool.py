"""Tool boundary for governed, semantic sales-call querying."""

from typing import Any
from uuid import UUID

from pydantic import Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sales_agent.features.calls.repository import get_call_content, query_calls
from sales_agent.features.calls.query import CALLS_CATALOG, QueryCallsInput
from sales_agent.tools.contracts import ToolDefinition, ToolExecutionResult, ToolInput


class DescribeCallsSchemaInput(ToolInput):
    pass


class GetCallContentInput(ToolInput):
    """Read one transcript in bounded chunks after a call has been identified."""

    call_id: UUID = Field(description="来自 query_calls 结果的通话记录 ID。")
    start_char: int = Field(
        default=0,
        ge=0,
        description="要读取的起始字符位置；首次读取传 0。",
    )
    max_chars: int = Field(
        default=8000,
        ge=500,
        le=12000,
        description="本次最多读取的字符数，最大 12000。",
    )


class CallSemanticToolExecutor:
    """Allowlisted analytical tools for a single authenticated user."""

    _definitions = {
        "describe_calls_schema": ToolDefinition(
            name="describe_calls_schema",
            description="返回销售通话可查询的指标、维度、筛选和数据限制。先调用它再进行复杂查询。",
            parameters=DescribeCallsSchemaInput.model_json_schema(),
            evidence_kind="metadata",
        ),
        "query_calls": ToolDefinition(
            name="query_calls",
            description="依据销售通话语义模型执行只读聚合查询。只能使用 describe_calls_schema 返回的字段和操作符。",
            parameters=QueryCallsInput.model_json_schema(),
            evidence_kind="substantive",
        ),
        "get_call_content": ToolDefinition(
            name="get_call_content",
            description="读取当前用户某一通通话的文本片段。必须先有明确 call_id；长文本使用 next_cursor 继续读取。",
            parameters=GetCallContentInput.model_json_schema(),
            evidence_kind="substantive",
        ),
    }

    def __init__(self, session: Session, *, user_id: str) -> None:
        self._session = session
        self._user_id = user_id

    @classmethod
    def tool_definitions(cls) -> list[ToolDefinition]:
        return list(cls._definitions.values())

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        if tool_name == "describe_calls_schema":
            try:
                DescribeCallsSchemaInput.model_validate(arguments)
            except ValidationError:
                return self._invalid_arguments(tool_name)
            return ToolExecutionResult(tool_name=tool_name, ok=True, result=CALLS_CATALOG)
        if tool_name != "query_calls":
            if tool_name == "get_call_content":
                return self._get_call_content(arguments)
            return ToolExecutionResult(tool_name=tool_name, ok=False, error={"code": "unknown_tool", "message": "请求的工具不在允许列表中。"})
        try:
            query = QueryCallsInput.model_validate(arguments)
        except ValidationError:
            return self._invalid_arguments(tool_name)
        try:
            page = query_calls(self._session, user_id=self._user_id, query=query)
        except SQLAlchemyError:
            return ToolExecutionResult(tool_name=tool_name, ok=False, error={"code": "query_failed", "message": "通话数据暂时无法查询，请稍后重试。"})
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=True,
            result={
                "rows": page.rows,
                "total_matches": page.total_matches,
                "total_result_rows": page.total_result_rows,
                "returned_count": len(page.rows),
                "has_more": query.offset + len(page.rows) < page.total_result_rows,
                "next_offset": (
                    query.offset + len(page.rows)
                    if query.offset + len(page.rows) < page.total_result_rows
                    else None
                ),
                "query": query.model_dump(mode="json"),
                "metric_definitions": CALLS_CATALOG["metrics"],
            },
        )

    def _get_call_content(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        try:
            payload = GetCallContentInput.model_validate(arguments)
        except ValidationError:
            return self._invalid_arguments("get_call_content")
        try:
            call = get_call_content(self._session, user_id=self._user_id, call_id=payload.call_id)
        except SQLAlchemyError:
            return ToolExecutionResult(tool_name="get_call_content", ok=False, error={"code": "query_failed", "message": "通话内容暂时无法读取，请稍后重试。"})
        if call is None:
            return ToolExecutionResult(tool_name="get_call_content", ok=False, error={"code": "not_found", "message": "未找到该通话记录。"})

        end_char = min(payload.start_char + payload.max_chars, len(call.transcript_text))
        return ToolExecutionResult(
            tool_name="get_call_content",
            ok=True,
            result={
                "call_id": str(call.call_id),
                "content": call.transcript_text[payload.start_char:end_char],
                "start_char": payload.start_char,
                "end_char": end_char,
                "next_cursor": end_char if end_char < len(call.transcript_text) else None,
                "total_chars": len(call.transcript_text),
                "analysis_status": call.analysis_status,
            },
        )

    @staticmethod
    def _invalid_arguments(tool_name: str) -> ToolExecutionResult:
        return ToolExecutionResult(tool_name=tool_name, ok=False, error={"code": "invalid_arguments", "message": "查询字段、筛选条件或排序不符合语义模型。"})
