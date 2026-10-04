"""Transport-neutral contracts for evidence-grounded scoring."""

from __future__ import annotations

from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sales_agent.features.knowledge.namespaces import (
    BASELINE_KNOWLEDGE_NAMESPACE,
    KNOWLEDGE_NAMESPACE_PATTERN,
)


class ScoringModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PeriodScoreRequest(ScoringModel):
    user_id: str = Field(min_length=1, max_length=255)
    sales_id: str = Field(min_length=1, max_length=64)
    date_from: date
    date_to: date
    knowledge_namespace: str = Field(
        default=BASELINE_KNOWLEDGE_NAMESPACE,
        min_length=1,
        max_length=64,
        pattern=KNOWLEDGE_NAMESPACE_PATTERN,
    )
    include_low_confidence: bool = False
    trace_low_confidence_to_transcript: bool = Field(
        default=True,
        description="低置信度事实仅作为线索，自动回溯同一分析运行中的原文片段。",
    )
    min_similarity: float = Field(default=0.25, ge=-1.0, le=1.0)
    evidence_limit_per_rule: int = Field(
        default=40,
        ge=1,
        le=100,
        description="每条规则在整个销售周期范围内保留的候选证据上限。",
    )
    max_review_repairs: int = Field(default=2, ge=0, le=3)
    max_model_calls: int = Field(default=18, ge=13, le=30)

    @model_validator(mode="after")
    def dates_are_ordered(self) -> "PeriodScoreRequest":
        if self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class ScoreRule(ScoringModel):
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1, max_length=100)
    explanation: str = Field(min_length=1, max_length=500)
    criteria: list[str] = Field(min_length=1, max_length=20)
    retrieval_queries: list[str] = Field(min_length=1, max_length=8)
    points_per_match: int
    maximum_points: int
    scoring_mode: Literal["binary", "count_capped", "interaction_metric", "audio_required"]
    retrieval_per_query_k: int = Field(default=8, ge=1, le=20)
    retrieval_final_k: int = Field(default=20, ge=1, le=100)
    retrieval_per_call_k: int = Field(default=2, ge=1, le=10)
    retrieval_low_confidence_k: int = Field(default=6, ge=0, le=50)
    retrieval_min_similarity: float = Field(default=0.35, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def points_match_mode(self) -> "ScoreRule":
        if self.retrieval_low_confidence_k > self.retrieval_final_k:
            raise ValueError("low-confidence retrieval budget cannot exceed final_k")
        if self.scoring_mode == "count_capped":
            if self.points_per_match == 0:
                raise ValueError("count-capped rules require non-zero points")
            if abs(self.maximum_points) < abs(self.points_per_match):
                raise ValueError("maximum_points cannot be smaller than points_per_match")
        elif self.maximum_points != self.points_per_match:
            raise ValueError("non-count rules must have equal points and maximum_points")
        return self


class ScoreSection(ScoringModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1, max_length=100)
    purpose: str = Field(min_length=1, max_length=500)
    rule_ids: list[str] = Field(min_length=1)


class ScoringPolicy(ScoringModel):
    policy_id: str
    version: str
    rules: list[ScoreRule] = Field(min_length=1)
    sections: list[ScoreSection] = Field(min_length=3, max_length=20)

    @model_validator(mode="after")
    def rule_ids_are_unique(self) -> "ScoringPolicy":
        ids = [item.rule_id for item in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule_id must be unique")
        section_ids = [item.section_id for item in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("section_id must be unique")
        assigned = [rule_id for section in self.sections for rule_id in section.rule_ids]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(ids):
            raise ValueError("sections must assign every rule exactly once")
        return self


class CallTarget(ScoringModel):
    call_id: str
    external_call_id: str
    call_date: date
    sales_stage: str
    analysis_run_id: str | None = None
    analysis_degraded: bool = False


class ScoreEvidence(ScoringModel):
    evidence_id: str
    retrieved_for_rule_id: str
    source: Literal["vector_fact", "transcript_context", "call_metric"] = "vector_fact"
    call_id: str
    analysis_run_id: str | None = None
    fact_id: str | None = None
    document_id: str | None = None
    trigger_validation_status: str | None = None
    fact_type: str | None = None
    speaker: str | None = None
    fact: str
    turn_no: int | None = None
    quote: str
    grounding: Literal["exact", "aligned", "turn_only", "computed"]
    confidence: float = Field(ge=0, le=1)
    similarity: float | None = None


class RuleJudgment(ScoringModel):
    call_id: str
    rule_id: str
    matched_evidence_ids: list[str]
    rationale: str = Field(min_length=1, max_length=1000)


class RuleScore(ScoringModel):
    rule_id: str
    rule_name: str
    status: Literal["scored", "not_met", "insufficient_evidence", "not_applicable"]
    matched_count: int = Field(ge=0)
    points: int
    evidence: list[ScoreEvidence]
    evidence_reliability: Literal["high", "medium", "low", "not_scored"] = (
        "not_scored"
    )
    rationale: str


class CallScore(ScoringModel):
    call_id: str
    external_call_id: str
    call_date: date
    sales_stage: str
    score: int
    rule_scores: list[RuleScore]
    reliability: Literal["high", "medium", "low"]
    assessment_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    assessed_rule_count: int = Field(default=0, ge=0)
    applicable_rule_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class DailyScore(ScoringModel):
    call_date: date
    call_count: int = Field(ge=0)
    scored_call_count: int = Field(ge=0)
    total_points: int
    average_score: float | None


class ReviewIssue(ScoringModel):
    issue_id: str
    call_id: str
    rule_id: str
    reason: str = Field(min_length=1, max_length=1000)
    supplemental_queries: list[str] = Field(max_length=5)


class ReviewReport(ScoringModel):
    schema_version: Literal["score-review-v1"] = "score-review-v1"
    status: Literal["completed", "failed"]
    reasonable: bool
    issues: list[ReviewIssue] = Field(max_length=50)
    summary: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def verdict_matches_issues(self) -> "ReviewReport":
        if self.status == "failed":
            if self.reasonable or self.issues:
                raise ValueError("failed review must be unreasonable with no issues")
            return self
        if self.reasonable == bool(self.issues):
            raise ValueError("reasonable must be true exactly when issues is empty")
        return self


class PeriodScoreResult(ScoringModel):
    sales_id: str
    date_from: date
    date_to: date
    policy_id: str
    policy_version: str
    knowledge_namespace: str
    call_count: int = Field(ge=0)
    scored_call_count: int = Field(ge=0)
    average_score: float | None
    total_points: int
    calls: list[CallScore]
    daily_scores: list[DailyScore]
    review: ReviewReport
    review_history: list[ReviewReport]
    review_repairs: int = Field(ge=0)
    model_call_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ScoreEvidenceRetriever(Protocol):
    def list_calls(self, request: PeriodScoreRequest) -> list[CallTarget]: ...

    def retrieve_period(
        self,
        *,
        request: PeriodScoreRequest,
        calls: list[CallTarget],
        rule: ScoreRule,
        queries: list[str] | None = None,
    ) -> list[ScoreEvidence]: ...


class SectionScoringAgent(Protocol):
    def score_section(
        self,
        *,
        section: ScoreSection,
        rules: list[ScoreRule],
        calls: list[CallTarget],
        evidence: list[ScoreEvidence],
        repair_issues: list[ReviewIssue] = (),
    ) -> list[RuleJudgment]: ...


class ScoreReviewAgent(Protocol):
    def review(
        self,
        *,
        policy: ScoringPolicy,
        calls: list[CallScore],
    ) -> ReviewReport: ...
