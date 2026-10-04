"""Validated, bounded query language for extracted call knowledge."""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from sales_agent.tools.contracts import ToolInput
from sales_agent.features.knowledge.namespaces import (
    BASELINE_KNOWLEDGE_NAMESPACE,
    KNOWLEDGE_NAMESPACE_PATTERN,
)


SpeakerFilter = Literal["sales", "customer", "both", "unknown"]


class GetCallFactsInput(ToolInput):
    call_id: UUID = Field(description="要查阅事实的通话 ID。")
    knowledge_namespace: str = Field(
        default=BASELINE_KNOWLEDGE_NAMESPACE,
        min_length=1,
        max_length=64,
        pattern=KNOWLEDGE_NAMESPACE_PATTERN,
        description="知识库命名空间；默认查询现有多阶段基线库。",
    )
    analysis_run_id: UUID | None = Field(
        default=None,
        description="可选的历史分析运行 ID；默认读取最新一次成功分析。",
    )
    phase: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="仅使用已有结果中出现过的精确英文 snake_case 值，不得翻译或猜测。",
    )
    fact_type: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="仅使用已有结果中出现过的精确英文 snake_case 值，不得翻译或猜测。",
    )
    speaker: SpeakerFilter | None = None
    include_low_confidence: bool = Field(
        default=False,
        description="默认只返回已验证事实；仅在审计或回溯时开启低置信度事实。",
    )
    score_tags: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="事实必须同时包含这里给出的全部标签。",
    )
    offset: int = Field(default=0, ge=0, le=10_000)
    limit: int = Field(default=50, ge=1, le=100)


class SearchCallKnowledgeInput(ToolInput):
    query: str = Field(
        min_length=2,
        max_length=500,
        description="用于召回事实的自然语言问题或描述。",
    )
    knowledge_namespace: str = Field(
        default=BASELINE_KNOWLEDGE_NAMESPACE,
        min_length=1,
        max_length=64,
        pattern=KNOWLEDGE_NAMESPACE_PATTERN,
        description="知识库命名空间；默认查询现有多阶段基线库。",
    )
    call_id: UUID | None = None
    sales_id: str | None = Field(default=None, min_length=1, max_length=64)
    date_from: date | None = None
    date_to: date | None = None
    phase: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    fact_type: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"
    )
    speaker: SpeakerFilter | None = None
    include_low_confidence: bool = Field(
        default=False,
        description="默认只检索已验证事实；仅在审计或回溯时开启低置信度事实。",
    )
    validation_status: Literal["verified", "low_confidence"] | None = Field(
        default=None,
        description="可选的精确验证状态过滤；主要用于评分召回分路。",
    )
    score_tags: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="事实必须同时包含这里给出的全部标签。",
    )
    min_similarity: float = Field(default=0.25, ge=-1.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def dates_are_ordered(self) -> "SearchCallKnowledgeInput":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self
