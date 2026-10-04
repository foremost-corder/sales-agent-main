"""Low-call fact extraction with local, per-fact evidence validation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.agent.structured_json import (
    completion_content,
    strict_json_instruction,
    strict_response_format,
)
from sales_agent.core.config import Settings
from sales_agent.features.knowledge.contracts import (
    ExtractedFact,
    FactEvidence,
    FactExtractionResult,
    TranscriptTurn,
)
from sales_agent.features.knowledge.fact_extractor import FactExtractionError, FactPhase, _anchor_quote_to_source


logger = logging.getLogger(__name__)


class _LeanEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_no: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=500)


class _LeanCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_no: int = Field(ge=1)
    phase: FactPhase
    fact_type: str = Field(min_length=1, max_length=64)
    speaker: Literal["sales", "customer", "unknown"]
    fact: str = Field(min_length=1, max_length=200)
    explicit: bool = True
    confidence: float = Field(ge=0, le=1)
    score_tags: list[str] = Field(default_factory=list, max_length=8)
    evidence: list[_LeanEvidence] = Field(min_length=1, max_length=3)


class _LeanExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-facts-lean-v1"]
    call_id: str = Field(min_length=1)
    candidates: list[_LeanCandidate] = Field(max_length=100)


class _LeanReviewItem(_LeanCandidate):
    supported: bool


class _LeanReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["conversation-facts-lean-review-v1"]
    candidates: list[_LeanReviewItem] = Field(max_length=100)


_EXTRACTION_INSTRUCTION = """你是销售通话事实提炼器。一次完成事实提取、阶段分类、事实类型、事实级说话人和证据定位。
只输出 JSON。每条事实必须原子化、可由给定 turn_no 的连续原文支持；不要把猜测写成明确事实。
speaker 表示说出该事实内容的人，而不是源标签。源标签可能错误或把双方混在同一标签中。
confidence 是你对该条事实及其说话人归属的置信度。candidate_no 从 1 连续编号。
不要提取寒暄、语音信箱固定播报或没有业务意义的话。"""


_REVIEW_INSTRUCTION = """你只复核少量未通过本地证据检查的事实，不要分析整通电话。
根据每条候选事实附近的原文修正事实、说话人和证据引用。证据 quote 必须是对应 turn_no 中的连续原文。
若原文不能支持该事实，supported=false；否则 supported=true。"""


class OpenAILeanFactExtractor:
    """Use one full-transcript call and at most one compact exception review."""

    version = "conversation-facts-lean-v1"

    def __init__(self, settings: Settings, *, review_low_confidence: bool = True) -> None:
        self._settings = settings
        self._review_low_confidence = review_low_confidence

    def extract(self, *, call_id: str, turns: list[TranscriptTurn]) -> FactExtractionResult:
        raw = self._complete(
            instruction=_EXTRACTION_INSTRUCTION,
            payload={
                "call_id": call_id,
                "transcript": [
                    {
                        "turn_no": turn.turn_no,
                        "source_speaker_label": turn.source_speaker_label,
                        "timestamp_ms": turn.timestamp_ms,
                        "text": turn.text,
                    }
                    for turn in turns
                ],
            },
            max_output_tokens=self._settings.fact_extraction_max_output_tokens,
            operation="lean fact extraction",
            response_model=_LeanExtractionOutput,
        )
        try:
            output = _LeanExtractionOutput.model_validate_json(raw)
        except ValidationError as exc:
            raise FactExtractionError(f"lean fact extraction returned invalid JSON: {exc}") from exc
        if output.call_id != call_id:
            raise FactExtractionError("lean fact extraction returned a mismatched call_id")

        candidates = _deduplicate_candidates(output.candidates)
        facts = _assemble_locally_validated_facts(candidates, turns)
        review_indexes = [
            index
            for index, fact in enumerate(facts)
            if fact.validation_status == "low_confidence"
        ]
        if self._review_low_confidence and review_indexes:
            reviewed = self._review(candidates, facts, review_indexes, turns)
            if reviewed is not None:
                candidates = reviewed
                facts = _assemble_locally_validated_facts(candidates, turns)

        facts = [
            fact.model_copy(update={"fact_id": f"fact-{index}"})
            for index, fact in enumerate(facts, start=1)
        ]
        low_confidence = [fact for fact in facts if fact.validation_status == "low_confidence"]
        return FactExtractionResult(
            schema_version="conversation-facts-v1",
            call_id=call_id,
            facts=facts,
            status="low_confidence" if low_confidence else "completed",
            quality_issues=sorted(
                {issue for fact in low_confidence for issue in fact.quality_issues}
            )[:20],
        )

    def _review(
        self,
        candidates: list[_LeanCandidate],
        facts: list[ExtractedFact],
        review_indexes: list[int],
        turns: list[TranscriptTurn],
    ) -> list[_LeanCandidate] | None:
        turn_map = {turn.turn_no: turn for turn in turns}
        review_candidates: list[dict[str, Any]] = []
        relevant_turn_nos: set[int] = set()
        for index in review_indexes:
            candidate = candidates[index]
            candidate_turn_nos = {
                number
                for evidence in candidate.evidence
                for number in range(evidence.turn_no - 1, evidence.turn_no + 2)
                if number in turn_map
            }
            if not candidate_turn_nos:
                continue
            review_candidates.append(
                {
                    **candidate.model_dump(mode="json"),
                    "quality_issues": facts[index].quality_issues,
                }
            )
            relevant_turn_nos.update(candidate_turn_nos)
        if not review_candidates:
            return None
        payload = {
            "candidates": review_candidates,
            "nearby_transcript": [
                {
                    "turn_no": number,
                    "source_speaker_label": turn_map[number].source_speaker_label,
                    "text": turn_map[number].text,
                }
                for number in sorted(relevant_turn_nos)
            ],
        }
        try:
            raw = self._complete(
                instruction=_REVIEW_INSTRUCTION,
                payload=payload,
                max_output_tokens=min(
                    self._settings.knowledge_evaluation_max_output_tokens,
                    2400,
                ),
                operation="lean fact review",
                response_model=_LeanReviewOutput,
            )
            output = _LeanReviewOutput.model_validate_json(raw)
        except (ChatModelRequestError, FactExtractionError, ValidationError) as exc:
            logger.warning("lean_fact_review_skipped error_type=%s", type(exc).__name__)
            return None

        by_number = {item.candidate_no: item for item in output.candidates}
        result = list(candidates)
        for index in review_indexes:
            original = candidates[index]
            item = by_number.get(original.candidate_no)
            if item is None or not item.supported:
                continue
            result[index] = _LeanCandidate.model_validate(
                item.model_dump(exclude={"supported"})
            )
        return result

    def _complete(
        self,
        *,
        instruction: str,
        payload: dict[str, Any],
        max_output_tokens: int,
        operation: str,
        response_model: type[BaseModel],
    ) -> str:
        if not self._settings.has_openai_api_key:
            raise ChatModelNotConfiguredError("OPENAI_API_KEY is not configured")
        assert self._settings.openai_api_key is not None
        options: dict[str, object] = {
            "api_key": self._settings.openai_api_key.get_secret_value(),
            "max_retries": 1,
            "timeout": 45.0,
        }
        if self._settings.openai_base_url:
            options["base_url"] = self._settings.openai_base_url
        instruction = strict_json_instruction(instruction)
        try:
            response = OpenAI(**options).chat.completions.create(
                model=self._settings.chat_model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=strict_response_format(response_model, operation),
                max_completion_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise ChatModelRequestError(
                f"{operation} request failed",
                status_code=getattr(exc, "status_code", None),
                request_id=getattr(exc, "request_id", None),
            ) from exc
        content = completion_content(response)
        if not isinstance(content, str) or not content.strip():
            raise FactExtractionError(f"{operation} returned empty content")
        return content


def _deduplicate_candidates(candidates: list[_LeanCandidate]) -> list[_LeanCandidate]:
    result: list[_LeanCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = re.sub(r"[\W_]+", "", candidate.fact, flags=re.UNICODE).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _assemble_locally_validated_facts(
    candidates: list[_LeanCandidate],
    turns: list[TranscriptTurn],
) -> list[ExtractedFact]:
    turn_map = {turn.turn_no: turn for turn in turns}
    fallback_turn = turns[0]
    facts: list[ExtractedFact] = []
    for index, candidate in enumerate(candidates, start=1):
        confidence = candidate.confidence
        issues: list[str] = []
        evidence_items: list[FactEvidence] = []
        for evidence in candidate.evidence:
            turn = turn_map.get(evidence.turn_no)
            if turn is None:
                if "evidence_turn_invalid" not in issues:
                    issues.append("evidence_turn_invalid")
                confidence = min(confidence, 0.3)
                continue
            if evidence.quote in turn.text:
                evidence_items.append(
                    FactEvidence(turn_no=turn.turn_no, quote=evidence.quote, grounding="exact")
                )
                continue
            anchored = _anchor_quote_to_source(evidence.quote, turn.text)
            if anchored is not None:
                evidence_items.append(
                    FactEvidence(turn_no=turn.turn_no, quote=anchored, grounding="aligned")
                )
                confidence = min(confidence, 0.7)
                if "evidence_quote_aligned" not in issues:
                    issues.append("evidence_quote_aligned")
            else:
                evidence_items.append(
                    FactEvidence(turn_no=turn.turn_no, quote=turn.text, grounding="turn_only")
                )
                confidence = min(confidence, 0.4)
                if "evidence_quote_unverified" not in issues:
                    issues.append("evidence_quote_unverified")
        if not evidence_items:
            evidence_items = [
                FactEvidence(
                    turn_no=fallback_turn.turn_no,
                    quote=fallback_turn.text,
                    grounding="turn_only",
                )
            ]
            confidence = min(confidence, 0.2)
            if "evidence_missing" not in issues:
                issues.append("evidence_missing")
        if candidate.speaker == "unknown":
            confidence = min(confidence, 0.5)
            issues.append("speaker_unresolved")
        if not candidate.explicit:
            confidence = min(confidence, 0.6)
            issues.append("fact_not_explicit")
        if candidate.confidence < 0.7:
            issues.append("model_low_confidence")
        facts.append(
            ExtractedFact(
                fact_id=f"fact-{index}",
                phase=candidate.phase,
                fact_type=candidate.fact_type,
                speaker=candidate.speaker,
                fact=candidate.fact,
                explicit=candidate.explicit,
                confidence=confidence,
                validation_status="low_confidence" if issues else "verified",
                quality_issues=list(dict.fromkeys(issues)),
                score_tags=candidate.score_tags,
                evidence=evidence_items,
            )
        )
    return facts
