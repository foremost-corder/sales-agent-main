"""Thin chat-tool adapter for the independently testable call knowledge workflow."""

import logging
from typing import Any
from uuid import UUID

from pydantic import Field, ValidationError
from sqlalchemy.orm import Session

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.integrations.embeddings.factory import build_embedding_provider
from sales_agent.core.config import get_settings
from sales_agent.features.knowledge.fact_extractor import FactExtractionError, OpenAIFactExtractor
from sales_agent.features.knowledge.namespaces import (
    BASELINE_KNOWLEDGE_NAMESPACE,
    KNOWLEDGE_NAMESPACE_PATTERN,
)
from sales_agent.features.knowledge.parser import TranscriptParseError
from sales_agent.features.knowledge.speaker_resolver import (
    OpenAISpeakerRoleResolver,
    SpeakerResolutionError,
)
from sales_agent.tools.contracts import ToolDefinition, ToolExecutionResult, ToolInput
from sales_agent.features.knowledge.standard_workflow import (
    AnalyzeCallKnowledgeWorkflow,
    CallKnowledgeConfigurationError,
    CallKnowledgeNotFoundError,
)


logger = logging.getLogger(__name__)


class AnalyzeCallInput(ToolInput):
    call_id: UUID
    force: bool = False
    knowledge_namespace: str = Field(
        default=BASELINE_KNOWLEDGE_NAMESPACE,
        min_length=1,
        max_length=64,
        pattern=KNOWLEDGE_NAMESPACE_PATTERN,
    )


def build_call_knowledge_workflow(session: Session) -> AnalyzeCallKnowledgeWorkflow:
    settings = get_settings()
    return AnalyzeCallKnowledgeWorkflow(
        session,
        speaker_resolver=OpenAISpeakerRoleResolver(settings),
        extractor=OpenAIFactExtractor(settings),
        embedder=build_embedding_provider(settings),
    )


class CallKnowledgeToolExecutor:
    _definition = ToolDefinition(
        name="analyze_call",
        description="解析当前用户已有的一通电话，从数据库文本提取可审计事实，并把结构事实、轮次原文片段和完整原文连同元数据写入向量知识库。相同来源和版本默认幂等复用。",
        parameters=AnalyzeCallInput.model_json_schema(),
        read_only=False,
        requires_approval=False,
        evidence_kind="substantive",
    )

    def __init__(
        self,
        session: Session,
        *,
        user_id: str,
        workflow: AnalyzeCallKnowledgeWorkflow | None = None,
    ) -> None:
        self._session = session
        self._user_id = user_id
        self._workflow = workflow

    @classmethod
    def tool_definitions(cls) -> list[ToolDefinition]:
        return [cls._definition]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        if tool_name != "analyze_call":
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "unknown_tool", "message": "请求的工具不在允许列表中。"},
            )
        try:
            payload = AnalyzeCallInput.model_validate(arguments)
        except ValidationError:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "invalid_arguments", "message": "call_id 或 force 参数无效。"},
            )
        try:
            workflow = self._workflow or build_call_knowledge_workflow(self._session)
            logger.info(
                "analyze_call_tool_start call_id=%s user_id=%s force=%s",
                payload.call_id,
                self._user_id,
                payload.force,
            )
            result = workflow.execute(
                call_id=payload.call_id,
                user_id=self._user_id,
                force=payload.force,
                knowledge_namespace=payload.knowledge_namespace,
            )
        except CallKnowledgeNotFoundError:
            logger.warning(
                "analyze_call_tool_not_found call_id=%s user_id=%s",
                payload.call_id,
                self._user_id,
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "not_found", "message": "未找到当前用户可分析的通话。"},
            )
        except ChatModelNotConfiguredError:
            logger.exception(
                "analyze_call_tool_model_not_configured call_id=%s user_id=%s",
                payload.call_id,
                self._user_id,
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "model_not_configured", "message": "事实提取或向量模型尚未配置。"},
            )
        except (ChatModelRequestError, FactExtractionError, SpeakerResolutionError, TranscriptParseError, CallKnowledgeConfigurationError, ValueError) as exc:
            logger.exception(
                "analyze_call_tool_failed call_id=%s user_id=%s error_type=%s",
                payload.call_id,
                self._user_id,
                type(exc).__name__,
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "analysis_failed", "message": "通话解析或知识入库失败，请检查分析运行记录。"},
            )
        logger.info(
            "analyze_call_tool_completed call_id=%s analysis_run_id=%s reused=%s document_count=%d",
            payload.call_id,
            result.analysis_run_id,
            result.reused,
            result.document_count,
        )
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=True,
            result=result.model_dump(mode="json"),
        )
