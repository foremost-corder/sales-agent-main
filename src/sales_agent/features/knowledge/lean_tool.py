"""Chat-tool adapter for the independent low-call knowledge workflow."""

import logging
from typing import Any
from uuid import UUID

from pydantic import Field, ValidationError
from sqlalchemy.orm import Session

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.integrations.embeddings.factory import build_embedding_provider
from sales_agent.core.config import get_settings
from sales_agent.features.knowledge.fact_extractor import FactExtractionError
from sales_agent.features.knowledge.lean_fact_extractor import OpenAILeanFactExtractor
from sales_agent.features.knowledge.namespaces import (
    KNOWLEDGE_NAMESPACE_PATTERN,
    LEAN_KNOWLEDGE_NAMESPACE,
)
from sales_agent.features.knowledge.parser import TranscriptParseError
from sales_agent.tools.contracts import ToolDefinition, ToolExecutionResult, ToolInput
from sales_agent.features.knowledge.standard_workflow import (
    CallKnowledgeConfigurationError,
    CallKnowledgeNotFoundError,
)
from sales_agent.features.knowledge.lean_workflow import AnalyzeCallKnowledgeLeanWorkflow


logger = logging.getLogger(__name__)


class AnalyzeLeanCallInput(ToolInput):
    call_id: UUID
    force: bool = False
    knowledge_namespace: str = Field(
        default=LEAN_KNOWLEDGE_NAMESPACE,
        min_length=1,
        max_length=64,
        pattern=KNOWLEDGE_NAMESPACE_PATTERN,
    )


def build_lean_call_knowledge_workflow(
    session: Session,
) -> AnalyzeCallKnowledgeLeanWorkflow:
    settings = get_settings()
    return AnalyzeCallKnowledgeLeanWorkflow(
        session,
        extractor=OpenAILeanFactExtractor(settings),
        embedder=build_embedding_provider(settings),
    )


class LeanCallKnowledgeToolExecutor:
    _definition = ToolDefinition(
        name="analyze_call_lean",
        description=(
            "使用低调用工作流处理当前用户的一通电话：一次提炼事实、分类、事实级说话人和证据，"
            "本地逐条校验证据，仅对异常事实做一次局部模型复核，然后写入知识库。"
            "它与多阶段 analyze_call 工具独立，默认按版本指纹幂等复用。"
        ),
        parameters=AnalyzeLeanCallInput.model_json_schema(),
        read_only=False,
        requires_approval=False,
        evidence_kind="substantive",
    )

    def __init__(
        self,
        session: Session,
        *,
        user_id: str,
        workflow: AnalyzeCallKnowledgeLeanWorkflow | None = None,
    ) -> None:
        self._session = session
        self._user_id = user_id
        self._workflow = workflow

    @classmethod
    def tool_definitions(cls) -> list[ToolDefinition]:
        return [cls._definition]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        if tool_name != "analyze_call_lean":
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "unknown_tool", "message": "请求的工具不在允许列表中。"},
            )
        try:
            payload = AnalyzeLeanCallInput.model_validate(arguments)
        except ValidationError:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "invalid_arguments", "message": "call_id 或 force 参数无效。"},
            )
        try:
            workflow = self._workflow or build_lean_call_knowledge_workflow(self._session)
            result = workflow.execute(
                call_id=payload.call_id,
                user_id=self._user_id,
                force=payload.force,
                knowledge_namespace=payload.knowledge_namespace,
            )
        except CallKnowledgeNotFoundError:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "not_found", "message": "未找到当前用户可分析的通话。"},
            )
        except ChatModelNotConfiguredError:
            logger.exception("analyze_call_lean_model_not_configured call_id=%s", payload.call_id)
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "model_not_configured", "message": "事实提取或向量模型尚未配置。"},
            )
        except (
            ChatModelRequestError,
            FactExtractionError,
            TranscriptParseError,
            CallKnowledgeConfigurationError,
            ValueError,
        ) as exc:
            logger.exception(
                "analyze_call_lean_failed call_id=%s error_type=%s",
                payload.call_id,
                type(exc).__name__,
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "analysis_failed", "message": "低调用知识入库失败，请检查分析运行记录。"},
            )
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=True,
            result=result.model_dump(mode="json"),
        )
