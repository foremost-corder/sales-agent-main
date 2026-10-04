from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sales_agent.features.knowledge.contracts import TranscriptTurn


CallLevel = Literal[
    "initial_contact",
    "needs_discovery",
    "solution_negotiation",
    "contract_delivery",
    "after_sales",
]
ScoringMode = Literal["binary", "count_capped", "audio_required"]
WhiteboardStatus = Literal[
    "hit", "miss", "not_applicable", "unassessable", "review_rejected"
]


class WhiteboardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WhiteboardRule(WhiteboardModel):
    dimension_id: str
    name: str
    explanation: str
    criteria: list[str]
    points_per_match: int
    maximum_points: int
    scoring_mode: ScoringMode


class WhiteboardScoringModule(WhiteboardModel):
    module_id: str
    name: str
    purpose: str
    rule_ids: list[str] = Field(min_length=1)
    applicable_levels: list[CallLevel] = Field(min_length=1)


class JudgmentEvidence(WhiteboardModel):
    turn_no: int = Field(ge=1)
    source_speaker_label: str = Field(min_length=1, max_length=64)
    speaker: Literal["sales", "customer", "unknown"]
    quote: str = Field(min_length=1, max_length=500)


class WhiteboardSpeakerAssignment(WhiteboardModel):
    source_speaker_label: str = Field(min_length=1, max_length=64)
    speaker: Literal["sales", "customer", "unknown"]


class CallLevelClassification(WhiteboardModel):
    schema_version: Literal["single-call-level-v1"]
    level: CallLevel
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=1000)
    evidence: list[JudgmentEvidence] = Field(min_length=1, max_length=5)
    speaker_assignments: list[WhiteboardSpeakerAssignment] = Field(min_length=1)


class DimensionJudgment(WhiteboardModel):
    dimension_id: str
    present: bool
    matched_count: int = Field(ge=0)
    rationale: str = Field(min_length=1, max_length=1000)
    evidence: list[JudgmentEvidence] = Field(max_length=10)


class ModuleJudgmentOutput(WhiteboardModel):
    schema_version: Literal["single-call-module-judgment-v1"]
    module_id: str
    dimensions: list[DimensionJudgment] = Field(min_length=1, max_length=14)


class EvidenceReview(WhiteboardModel):
    dimension_id: str
    accepted: bool
    rationale: str = Field(min_length=1, max_length=1000)


class EvidenceReviewOutput(WhiteboardModel):
    schema_version: Literal["single-call-evidence-review-v1"]
    reviews: list[EvidenceReview] = Field(max_length=14)


class DimensionScore(WhiteboardModel):
    dimension_id: str
    name: str
    applicable: bool
    assessable: bool
    present: bool
    whiteboard_status: WhiteboardStatus
    review_status: Literal["not_required", "accepted", "rejected"]
    matched_count: int = Field(ge=0)
    points_per_match: int
    maximum_points: int
    score: int
    rationale: str
    review_rationale: str | None
    evidence: list[JudgmentEvidence]


class SingleCallWhiteboardScore(WhiteboardModel):
    schema_version: Literal["single-call-whiteboard-score-v2"]
    call_id: str
    source_sales_stage: str
    call_level: CallLevel
    level_confidence: float = Field(ge=0, le=1)
    level_rationale: str
    level_evidence: list[JudgmentEvidence]
    executed_modules: list[str]
    total_score: int
    maximum_applicable_positive_score: int
    minimum_applicable_negative_score: int
    speaker_assignments: list[WhiteboardSpeakerAssignment] = Field(min_length=1)
    dimensions: list[DimensionScore] = Field(min_length=14, max_length=14)
    warnings: list[str]

    @model_validator(mode="after")
    def total_matches_dimensions(self) -> "SingleCallWhiteboardScore":
        if self.total_score != sum(item.score for item in self.dimensions):
            raise ValueError("total_score must equal the sum of dimension scores")
        return self


class CallLevelClassifier(Protocol):
    version: str

    def classify(
        self, *, call_id: str, turns: list[TranscriptTurn]
    ) -> CallLevelClassification: ...


class ModuleJudge(Protocol):
    version: str

    def judge_module(
        self,
        *,
        call_id: str,
        level: CallLevel,
        turns: list[TranscriptTurn],
        speaker_assignments: list[WhiteboardSpeakerAssignment],
        module: WhiteboardScoringModule,
        rules: tuple[WhiteboardRule, ...],
    ) -> ModuleJudgmentOutput: ...


class EvidenceReviewer(Protocol):
    version: str

    def review(
        self,
        *,
        call_id: str,
        level: CallLevel,
        turns: list[TranscriptTurn],
        rules: tuple[WhiteboardRule, ...],
        judgments: list[DimensionJudgment],
    ) -> EvidenceReviewOutput: ...
