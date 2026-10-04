"""Independent read-only tools for browsing and searching extracted facts."""

import logging
from typing import Any

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sales_agent.integrations.embeddings.factory import build_embedding_provider
from sales_agent.core.config import get_settings
from sales_agent.features.knowledge.contracts import EmbeddingProvider
from sales_agent.features.knowledge.repository import get_call_facts, search_call_knowledge
from sales_agent.features.knowledge.query import GetCallFactsInput, SearchCallKnowledgeInput
from sales_agent.tools.contracts import ToolDefinition, ToolExecutionResult


logger = logging.getLogger(__name__)


class KnowledgeReadToolExecutor:
    """Thin tool adapter; independent from the knowledge ingestion executor."""

    _definitions = {
        "get_call_facts": ToolDefinition(
            name="get_call_facts",
            description=(
                "读取当前用户某通电话已提取的结构化事实和原文证据。"
                "默认读取最新成功分析中的已验证事实；审计时可显式包含低置信度事实。"
                "默认查询 baseline-react-v4，也可指定实验命名空间进行隔离对比。"
                "可按阶段、事实类型、说话人和评分标签精确筛选。"
                "phase 和 fact_type 只能使用已有结果中的英文值，不得从中文问题猜测；"
                "用户转述事实并索要原文时应改用 search_call_knowledge。"
            ),
            parameters=GetCallFactsInput.model_json_schema(),
            evidence_kind="substantive",
        ),
        "search_call_knowledge": ToolDefinition(
            name="search_call_knowledge",
            description=(
                "使用自然语言在当前用户已入库的通话事实中进行语义检索，"
                "默认只检索已验证事实，审计时可显式包含低置信度事实；"
                "每次只检索指定的一个知识命名空间，默认 baseline-react-v4；"
                "返回置信度、质量问题、原文证据、来源通话和相似度。"
                "当用户引用或转述一条事实并询问原文、依据、证据或轮次时优先使用本工具。"
            ),
            parameters=SearchCallKnowledgeInput.model_json_schema(),
            evidence_kind="substantive",
        ),
    }

    def __init__(
        self,
        session: Session,
        *,
        user_id: str,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._session = session
        self._user_id = user_id
        self._embedder = embedder

    @classmethod
    def tool_definitions(cls) -> list[ToolDefinition]:
        return list(cls._definitions.values())

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        if tool_name == "get_call_facts":
            return self._get_call_facts(arguments)
        if tool_name == "search_call_knowledge":
            return self._search_call_knowledge(arguments)
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=False,
            error={"code": "unknown_tool", "message": "请求的工具不在允许列表中。"},
        )

    def _get_call_facts(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        try:
            query = GetCallFactsInput.model_validate(arguments)
        except ValidationError:
            return self._invalid_arguments("get_call_facts")
        try:
            page = get_call_facts(self._session, user_id=self._user_id, query=query)
        except SQLAlchemyError:
            logger.exception("get_call_facts_failed call_id=%s", query.call_id)
            return self._query_failed("get_call_facts")
        if not page.call_found:
            return ToolExecutionResult(
                tool_name="get_call_facts",
                ok=False,
                error={"code": "not_found", "message": "未找到该通话记录。"},
            )
        result: dict[str, Any] = {
            "call_id": str(query.call_id),
            "knowledge_namespace": query.knowledge_namespace,
            "analysis_run_id": str(page.analysis_run_id) if page.analysis_run_id else None,
            "status": "completed" if page.analysis_run_id else "not_analyzed",
            "facts": page.rows,
            "returned_count": len(page.rows),
            "next_offset": page.next_offset,
        }
        if page.analysis_run_id and not page.rows and _has_fact_filters(query):
            result["status"] = "no_matching_facts"
            result["retry_hint"] = (
                "精确过滤没有命中不代表事实不存在；若过滤值不是来自已有事实结果，"
                "请去掉猜测的过滤条件并使用 search_call_knowledge 按自然语言检索。"
            )
        return ToolExecutionResult(
            tool_name="get_call_facts",
            ok=True,
            result=result,
        )

    def _search_call_knowledge(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        try:
            query = SearchCallKnowledgeInput.model_validate(arguments)
        except ValidationError:
            return self._invalid_arguments("search_call_knowledge")
        try:
            embedder = self._embedder or build_embedding_provider(get_settings())
            query_vector = embedder.embed([query.query])[0]
            matches = search_call_knowledge(
                self._session,
                user_id=self._user_id,
                query=query,
                query_vector=query_vector,
                embedding_provider=embedder.provider_name,
                embedding_model=embedder.model,
                embedding_dimensions=embedder.dimensions,
            )
        except (SQLAlchemyError, ValueError, RuntimeError, IndexError):
            logger.exception("search_call_knowledge_failed")
            return self._query_failed("search_call_knowledge")
        return ToolExecutionResult(
            tool_name="search_call_knowledge",
            ok=True,
            result={
                "matches": matches,
                "returned_count": len(matches),
                "embedding_model": embedder.model,
                "knowledge_namespace": query.knowledge_namespace,
            },
        )

    @staticmethod
    def _invalid_arguments(tool_name: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=False,
            error={"code": "invalid_arguments", "message": "事实查询参数无效。"},
        )

    @staticmethod
    def _query_failed(tool_name: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=False,
            error={"code": "query_failed", "message": "事实知识暂时无法查询，请稍后重试。"},
        )


def _has_fact_filters(query: GetCallFactsInput) -> bool:
    return any(
        (
            query.phase,
            query.fact_type,
            query.speaker,
            query.score_tags,
            query.include_low_confidence,
        )
    )
