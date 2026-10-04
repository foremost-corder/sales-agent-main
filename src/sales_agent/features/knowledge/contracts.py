from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpeakerRole(StrEnum):
    SALES = "sales"
    CUSTOMER = "customer"
    UNKNOWN = "unknown"


class TranscriptTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_no: int = Field(ge=1)
    source_speaker_label: str = Field(min_length=1, max_length=64)
    speaker: SpeakerRole
    timestamp_ms: int | None = Field(default=None, ge=0)
    text: str = Field(min_length=1)
    raw_line: str = Field(min_length=1)


class SpeakerAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_speaker_label: str = Field(min_length=1, max_length=64)
    role: Literal["sales", "customer"]
    confidence: float = Field(ge=0, le=1)
    evidence_turn_nos: list[int] = Field(min_length=1, max_length=5)


class SpeakerResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["speaker-roles-v1"]
    assignments: list[SpeakerAssignment] = Field(min_length=1)

    @model_validator(mode="after")
    def labels_are_unique(self) -> "SpeakerResolutionResult":
        labels = [item.source_speaker_label for item in self.assignments]
        if len(labels) != len(set(labels)):
            raise ValueError("source_speaker_label must be unique")
        return self


class FactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_no: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=500)
    grounding: Literal["exact", "aligned", "turn_only"] = "exact"


class ExtractedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(pattern=r"^fact-[1-9][0-9]*$")
    phase: Literal[
        "opening",
        "discovery",
        "proposal_negotiation",
        "fulfillment_support",
        "relationship_closing",
        "unknown",
    ]
    fact_type: str = Field(min_length=1, max_length=64)
    speaker: Literal["sales", "customer", "unknown"]
    fact: str = Field(min_length=1, max_length=200)
    explicit: bool
    confidence: float = Field(default=1.0, ge=0, le=1)
    validation_status: Literal["verified", "low_confidence"] = "verified"
    quality_issues: list[str] = Field(default_factory=list, max_length=10)
    score_tags: list[str] = Field(default_factory=list, max_length=8)
    evidence: list[FactEvidence] = Field(min_length=1, max_length=3)


class FactExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-facts-v1"]
    call_id: str = Field(min_length=1)
    facts: list[ExtractedFact] = Field(max_length=100)
    status: Literal["completed", "low_confidence", "skipped"] = "completed"
    quality_issues: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def fact_ids_are_sequential(self) -> "FactExtractionResult":
        expected = [f"fact-{index}" for index in range(1, len(self.facts) + 1)]
        if [fact.fact_id for fact in self.facts] != expected:
            raise ValueError("fact_id must start at fact-1 and remain sequential")
        return self


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["schema", "business", "semantic"]
    code: str = Field(min_length=1, max_length=64)
    path: list[str | int] = Field(default_factory=list)
    message: str = Field(min_length=1, max_length=1000)


class KnowledgeOutputEvaluator(Protocol):
    version: str

    def evaluate(
        self,
        *,
        task_name: str,
        source: dict[str, Any],
        candidate: dict[str, Any],
        criteria: list[str],
    ) -> list[ValidationIssue]: ...


class KnowledgeDocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_type: Literal["structured_fact", "transcript_chunk", "full_transcript"]
    source_key: str = Field(min_length=1, max_length=255)
    ordinal: int = Field(ge=1)
    content: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    metadata: dict[str, Any]


class FactExtractor(Protocol):
    version: str

    def extract(self, *, call_id: str, turns: list[TranscriptTurn]) -> FactExtractionResult: ...


class SpeakerRoleResolver(Protocol):
    version: str

    def resolve(self, *, call_id: str, turns: list[TranscriptTurn]) -> SpeakerResolutionResult: ...


class EmbeddingProvider(Protocol):
    provider_name: str
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class CallKnowledgeResult(BaseModel):
    analysis_run_id: str
    call_id: str
    status: Literal["completed"]
    reused: bool
    degraded: bool = False
    enrichment_status: Literal["completed", "low_confidence", "skipped"] = "completed"
    fact_count: int = Field(ge=0)
    low_confidence_fact_count: int = Field(default=0, ge=0)
    transcript_chunk_count: int = Field(ge=0)
    full_transcript_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    embedding_model: str
    knowledge_namespace: str = "baseline-react-v4"
