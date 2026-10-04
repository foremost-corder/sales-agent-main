"""Agent-tool adapter for independent single-call whiteboard scoring."""

from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.core.config import get_settings
from sales_agent.features.knowledge.parser import TranscriptParseError
from sales_agent.features.single_call_scoring.model import (
    OpenAICallLevelClassifier,
    OpenAIEvidenceReviewer,
    OpenAIModuleJudge,
    WhiteboardModelOutputError,
)
from sales_agent.features.single_call_scoring.repository import (
    get_single_call_scoring_source,
)
from sales_agent.features.single_call_scoring.workflow import (
    SingleCallWhiteboardScoringWorkflow,
    WhiteboardScoringError,
)
from sales_agent.tools.contracts import ToolDefinition, ToolExecutionResult, ToolInput


class ScoreSingleCallWhiteboardInput(ToolInput):
    call_id: UUID


class SingleCallWhiteboardToolExecutor:
    _definition = ToolDefinition(
        name="score_single_call_whiteboard",
        description=(
            "直接读取当前用户指定的一通电话原文，先判定业务层级，再由适用的独立模块并行"
            "评估14个关键行为，经过证据复核后输出白板状态和本地确定性分数。不读取知识库、"
            "历史评分或其他通话。"
        ),
        parameters=ScoreSingleCallWhiteboardInput.model_json_schema(),
        evidence_kind="substantive",
    )

    def __init__(
        self,
        session: Session,
        *,
        user_id: str,
        workflow: SingleCallWhiteboardScoringWorkflow | None = None,
    ) -> None:
        self._session = session
        self._user_id = user_id
        self._workflow = workflow

    @classmethod
    def tool_definitions(cls) -> list[ToolDefinition]:
        return [cls._definition]

    def execute(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolExecutionResult:
        if tool_name != self._definition.name:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "unknown_tool", "message": "请求的工具不存在。"},
            )
        try:
            payload = ScoreSingleCallWhiteboardInput.model_validate(arguments)
        except ValidationError:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "invalid_arguments", "message": "call_id 无效。"},
            )
        try:
            source = get_single_call_scoring_source(
                self._session,
                user_id=self._user_id,
                call_id=payload.call_id,
            )
        except SQLAlchemyError:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "query_failed", "message": "通话读取失败，请稍后重试。"},
            )
        if source is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "not_found", "message": "未找到该通话记录。"},
            )
        if self._workflow is None:
            settings = get_settings()
            workflow = SingleCallWhiteboardScoringWorkflow(
                OpenAICallLevelClassifier(settings),
                OpenAIModuleJudge(settings),
                OpenAIEvidenceReviewer(settings),
            )
        else:
            workflow = self._workflow
        try:
            score = workflow.score(
                call_id=str(source.call_id),
                sales_stage=source.sales_stage,
                transcript_text=source.transcript_text,
            )
        except ChatModelNotConfiguredError:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "model_not_configured", "message": "评分模型尚未配置。"},
            )
        except ChatModelRequestError as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "model_request_failed", "message": exc.public_detail},
            )
        except (
            TranscriptParseError,
            WhiteboardModelOutputError,
            WhiteboardScoringError,
        ) as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "scoring_failed", "message": str(exc)[:1000]},
            )
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=True,
            result=score.model_dump(mode="json"),
        )
