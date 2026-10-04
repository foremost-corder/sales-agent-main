"""Types shared by the agent model adapter and ReAct loop."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sales_agent.agent.tool_runtime import ToolSpec


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentResult:
    content: str
    steps: int
    tool_calls: int


class FinalAnswer(BaseModel):
    """The model's final answer and the evidence it claims to use."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent-final-answer-v1"]

    answer: str = Field(
        min_length=1,
        max_length=12000,
        description="最终展示给用户的完整回答正文。",
    )
    grounding: Literal["none", "metadata", "tool"] = Field(
        description="通用回答用 none，能力目录说明用 metadata，业务数据结论用 tool。"
    )
    evidence_refs: list[str] = Field(
        max_length=12,
        description="支撑回答的短证据编号（如 E1）；grounding=none 时必须为空。",
    )


class ChatModel(Protocol):
    def create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> Any: ...
