"""Layered one-call workflow: classify, parallel score, review, aggregate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from sales_agent.features.knowledge.contracts import TranscriptTurn
from sales_agent.features.knowledge.parser import parse_transcript_turns
from sales_agent.features.single_call_scoring.contracts import (
    CallLevelClassification,
    CallLevelClassifier,
    DimensionJudgment,
    DimensionScore,
    EvidenceReview,
    EvidenceReviewer,
    JudgmentEvidence,
    ModuleJudge,
    ModuleJudgmentOutput,
    SingleCallWhiteboardScore,
    WhiteboardRule,
    WhiteboardScoringModule,
)
from sales_agent.features.single_call_scoring.policy import (
    WHITEBOARD_RULES,
    modules_for_level,
)


class WhiteboardScoringError(ValueError):
    pass


class SingleCallWhiteboardScoringWorkflow:
    def __init__(
        self,
        classifier: CallLevelClassifier,
        module_judge: ModuleJudge,
        reviewer: EvidenceReviewer,
        *,
        rules: tuple[WhiteboardRule, ...] = WHITEBOARD_RULES,
        max_workers: int = 4,
    ) -> None:
        self._classifier = classifier
        self._module_judge = module_judge
        self._reviewer = reviewer
        self._rules = rules
        self._rule_by_id = {rule.dimension_id: rule for rule in rules}
        self._max_workers = max(1, max_workers)

    def score(
        self, *, call_id: str, sales_stage: str, transcript_text: str
    ) -> SingleCallWhiteboardScore:
        turns = parse_transcript_turns(transcript_text)
        classification = self._classifier.classify(call_id=call_id, turns=turns)
        speaker_map = _validate_classification(classification, turns)
        modules = modules_for_level(classification.level)
        outputs = self._score_modules(
            call_id=call_id,
            classification=classification,
            turns=turns,
            modules=modules,
        )
        judgments: dict[str, DimensionJudgment] = {}
        for module, output in zip(modules, outputs, strict=True):
            validated = _validate_module_output(
                module, output, turns, speaker_map, self._rule_by_id
            )
            judgments.update(validated)

        candidates = [item for item in judgments.values() if item.present]
        reviews: dict[str, EvidenceReview] = {}
        if candidates:
            candidate_rules = tuple(
                self._rule_by_id[item.dimension_id] for item in candidates
            )
            review_output = self._reviewer.review(
                call_id=call_id,
                level=classification.level,
                turns=turns,
                rules=candidate_rules,
                judgments=candidates,
            )
            reviews = _validate_reviews(review_output.reviews, candidates)

        dimensions = [
            _aggregate_dimension(rule, judgments.get(rule.dimension_id), reviews)
            for rule in self._rules
        ]
        active_rule_ids = {
            rule_id for module in modules for rule_id in module.rule_ids
        }
        assessable_active_rules = [
            rule for rule in self._rules
            if rule.dimension_id in active_rule_ids
            and rule.scoring_mode != "audio_required"
        ]
        return SingleCallWhiteboardScore(
            schema_version="single-call-whiteboard-score-v2",
            call_id=call_id,
            source_sales_stage=sales_stage,
            call_level=classification.level,
            level_confidence=classification.confidence,
            level_rationale=classification.rationale,
            level_evidence=classification.evidence,
            executed_modules=[module.module_id for module in modules],
            total_score=sum(item.score for item in dimensions),
            maximum_applicable_positive_score=sum(
                max(0, rule.maximum_points) for rule in assessable_active_rules
            ),
            minimum_applicable_negative_score=sum(
                min(0, rule.maximum_points) for rule in assessable_active_rules
            ),
            speaker_assignments=classification.speaker_assignments,
            dimensions=dimensions,
            warnings=[
                "层级由当前通话原文判定；source_sales_stage 只作为来源元数据展示，不参与层级分类。",
                "输入只有TXT转写文本，语速、音量、语调和普通话标准度不可判定；该项不加扣分。",
                "命中项经独立证据复核；复核拒绝的项保留溯源信息但计0分。",
            ],
        )

    def _score_modules(
        self,
        *,
        call_id: str,
        classification: CallLevelClassification,
        turns: list[TranscriptTurn],
        modules: tuple[WhiteboardScoringModule, ...],
    ) -> list[ModuleJudgmentOutput]:
        results: dict[str, ModuleJudgmentOutput] = {}
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(modules)),
            thread_name_prefix="single-call-score",
        ) as pool:
            futures = {
                pool.submit(
                    self._module_judge.judge_module,
                    call_id=call_id,
                    level=classification.level,
                    turns=turns,
                    speaker_assignments=classification.speaker_assignments,
                    module=module,
                    rules=tuple(self._rule_by_id[item] for item in module.rule_ids),
                ): module.module_id
                for module in modules
            }
            for future in as_completed(futures):
                module_id = futures[future]
                results[module_id] = future.result()
        return [results[module.module_id] for module in modules]


def _validate_classification(
    classification: CallLevelClassification, turns: list[TranscriptTurn]
) -> dict[str, str]:
    expected = {turn.source_speaker_label for turn in turns}
    labels = [
        item.source_speaker_label for item in classification.speaker_assignments
    ]
    if len(labels) != len(set(labels)):
        raise WhiteboardScoringError("speaker assignment labels must be unique")
    if set(labels) != expected:
        raise WhiteboardScoringError(
            f"speaker assignments must cover exactly {sorted(expected)!r}"
        )
    speaker_map = {
        item.source_speaker_label: item.speaker
        for item in classification.speaker_assignments
    }
    _validate_evidence(classification.evidence, turns, speaker_map)
    return speaker_map


def _validate_module_output(
    module: WhiteboardScoringModule,
    output: ModuleJudgmentOutput,
    turns: list[TranscriptTurn],
    speaker_map: dict[str, str],
    rule_by_id: dict[str, WhiteboardRule],
) -> dict[str, DimensionJudgment]:
    if output.module_id != module.module_id:
        raise WhiteboardScoringError(
            f"module id must be {module.module_id}, received {output.module_id}"
        )
    actual = [item.dimension_id for item in output.dimensions]
    if actual != module.rule_ids:
        raise WhiteboardScoringError(
            f"dimension ids for {module.module_id} must be exactly {module.rule_ids!r}"
        )
    for judgment in output.dimensions:
        rule = rule_by_id[judgment.dimension_id]
        if judgment.present:
            if judgment.matched_count < 1 or not judgment.evidence:
                raise WhiteboardScoringError(
                    f"present dimension {rule.dimension_id} requires count and evidence"
                )
        elif judgment.matched_count != 0 or judgment.evidence:
            raise WhiteboardScoringError(
                f"absent dimension {rule.dimension_id} must have zero count and no evidence"
            )
        if rule.scoring_mode == "binary" and judgment.matched_count not in {0, 1}:
            raise WhiteboardScoringError(
                f"binary dimension {rule.dimension_id} has invalid matched_count"
            )
        if rule.scoring_mode == "count_capped":
            cap = rule.maximum_points // rule.points_per_match
            if judgment.matched_count > cap:
                raise WhiteboardScoringError(
                    f"dimension {rule.dimension_id} exceeds count cap {cap}"
                )
        _validate_evidence(judgment.evidence, turns, speaker_map)
    return {item.dimension_id: item for item in output.dimensions}


def _validate_evidence(
    evidence_items: list[JudgmentEvidence],
    turns: list[TranscriptTurn],
    speaker_map: dict[str, str],
) -> None:
    turn_map = {turn.turn_no: turn for turn in turns}
    for evidence in evidence_items:
        turn = turn_map.get(evidence.turn_no)
        if turn is None:
            raise WhiteboardScoringError(f"unknown evidence turn {evidence.turn_no}")
        if evidence.source_speaker_label != turn.source_speaker_label:
            raise WhiteboardScoringError(
                f"evidence label mismatch at turn {evidence.turn_no}"
            )
        if evidence.speaker != speaker_map[evidence.source_speaker_label]:
            raise WhiteboardScoringError(
                f"evidence speaker mismatch at turn {evidence.turn_no}"
            )
        # The model only selects the evidence turn. The canonical quotation is
        # always copied locally from the transcript, so verbatim reproduction is
        # not part of the model contract and cannot cause a false failure.
        evidence.quote = turn.text


def _validate_reviews(
    reviews: list[EvidenceReview], candidates: list[DimensionJudgment]
) -> dict[str, EvidenceReview]:
    expected = [item.dimension_id for item in candidates]
    actual = [item.dimension_id for item in reviews]
    if actual != expected:
        raise WhiteboardScoringError(
            f"review ids must be exactly {expected!r}, received {actual!r}"
        )
    return {item.dimension_id: item for item in reviews}


def _aggregate_dimension(
    rule: WhiteboardRule,
    judgment: DimensionJudgment | None,
    reviews: dict[str, EvidenceReview],
) -> DimensionScore:
    if rule.scoring_mode == "audio_required":
        return _empty_score(
            rule,
            applicable=True,
            assessable=False,
            status="unassessable",
            rationale="纯TXT无法判定真实语速、音量、语调或普通话标准度。",
        )
    if judgment is None:
        return _empty_score(
            rule,
            applicable=False,
            assessable=False,
            status="not_applicable",
            rationale="该维度不属于本通电话层级启用的评分模块。",
        )
    if not judgment.present:
        return DimensionScore(
            dimension_id=rule.dimension_id,
            name=rule.name,
            applicable=True,
            assessable=True,
            present=False,
            whiteboard_status="miss",
            review_status="not_required",
            matched_count=0,
            points_per_match=rule.points_per_match,
            maximum_points=rule.maximum_points,
            score=0,
            rationale=judgment.rationale,
            review_rationale=None,
            evidence=[],
        )
    review = reviews[rule.dimension_id]
    accepted = review.accepted
    if accepted and rule.scoring_mode == "count_capped":
        score = min(
            judgment.matched_count * rule.points_per_match,
            rule.maximum_points,
        )
    elif accepted:
        score = rule.points_per_match
    else:
        score = 0
    return DimensionScore(
        dimension_id=rule.dimension_id,
        name=rule.name,
        applicable=True,
        assessable=True,
        present=accepted,
        whiteboard_status="hit" if accepted else "review_rejected",
        review_status="accepted" if accepted else "rejected",
        matched_count=judgment.matched_count,
        points_per_match=rule.points_per_match,
        maximum_points=rule.maximum_points,
        score=score,
        rationale=judgment.rationale,
        review_rationale=review.rationale,
        evidence=judgment.evidence,
    )


def _empty_score(
    rule: WhiteboardRule,
    *,
    applicable: bool,
    assessable: bool,
    status: str,
    rationale: str,
) -> DimensionScore:
    return DimensionScore(
        dimension_id=rule.dimension_id,
        name=rule.name,
        applicable=applicable,
        assessable=assessable,
        present=False,
        whiteboard_status=status,  # type: ignore[arg-type]
        review_status="not_required",
        matched_count=0,
        points_per_match=rule.points_per_match,
        maximum_points=rule.maximum_points,
        score=0,
        rationale=rationale,
        review_rationale=None,
        evidence=[],
    )
