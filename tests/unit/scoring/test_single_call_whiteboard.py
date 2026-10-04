from threading import Barrier

import pytest

from sales_agent.agent.structured_json import strict_model_json_schema
from sales_agent.features.single_call_scoring.contracts import (
    CallLevelClassification,
    DimensionJudgment,
    EvidenceReview,
    EvidenceReviewOutput,
    JudgmentEvidence,
    ModuleJudgmentOutput,
    WhiteboardSpeakerAssignment,
)
from sales_agent.features.single_call_scoring.model import (
    _MODULE_INSTRUCTION,
    _REVIEW_INSTRUCTION,
)
from sales_agent.features.single_call_scoring.policy import (
    WHITEBOARD_MODULES,
    WHITEBOARD_RULES,
    modules_for_level,
)
from sales_agent.features.single_call_scoring.workflow import (
    SingleCallWhiteboardScoringWorkflow,
)


TRANSCRIPT = """用户0：我们是做企业电脑租赁的
用户1：我们下个月需要二十台办公电脑，预算十万元
用户1：价格还是太高了，不考虑了"""


ASSIGNMENTS = [
    WhiteboardSpeakerAssignment(source_speaker_label="用户0", speaker="sales"),
    WhiteboardSpeakerAssignment(source_speaker_label="用户1", speaker="customer"),
]


class FakeClassifier:
    version = "fake-level-v1"

    def classify(self, **_) -> CallLevelClassification:
        return CallLevelClassification(
            schema_version="single-call-level-v1",
            level="solution_negotiation",
            confidence=0.95,
            rationale="对话包含需求、预算和价格异议。",
            evidence=[_evidence(2, "用户1", "customer", "预算十万元")],
            speaker_assignments=ASSIGNMENTS,
        )


class FakeModuleJudge:
    version = "fake-module-v1"

    def __init__(self, barrier: Barrier | None = None) -> None:
        self.calls: list[str] = []
        self.barrier = barrier

    def judge_module(self, *, module, rules, **_) -> ModuleJudgmentOutput:
        self.calls.append(module.module_id)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        judgments = [_absent(rule.dimension_id) for rule in rules]
        by_id = {item.dimension_id: item for item in judgments}
        if "business_introduction" in by_id:
            by_id["business_introduction"] = _present(
                "business_introduction",
                1,
                [_evidence(1, "用户0", "sales", "企业电脑租赁")],
            )
            by_id["discovery_information"] = _present(
                "discovery_information",
                3,
                [
                    _evidence(2, "用户1", "customer", "下个月需要二十台办公电脑"),
                    _evidence(2, "用户1", "customer", "预算十万元"),
                ],
            )
        if "objection_unresolved" in by_id:
            by_id["objection_unresolved"] = _present(
                "objection_unresolved",
                1,
                [_evidence(3, "用户1", "customer", "价格还是太高了，不考虑了")],
            )
        return ModuleJudgmentOutput(
            schema_version="single-call-module-judgment-v1",
            module_id=module.module_id,
            dimensions=[by_id[rule.dimension_id] for rule in rules],
        )


class FakeReviewer:
    version = "fake-reviewer-v1"

    def __init__(self, reject: set[str] | None = None) -> None:
        self.reject = reject or set()
        self.reviewed_ids: list[str] = []

    def review(self, *, judgments, **_) -> EvidenceReviewOutput:
        self.reviewed_ids = [item.dimension_id for item in judgments]
        return EvidenceReviewOutput(
            schema_version="single-call-evidence-review-v1",
            reviews=[
                EvidenceReview(
                    dimension_id=item.dimension_id,
                    accepted=item.dimension_id not in self.reject,
                    rationale="证据充分。" if item.dimension_id not in self.reject else "证据不足。",
                )
                for item in judgments
            ],
        )


def _evidence(turn_no: int, label: str, speaker: str, quote: str) -> JudgmentEvidence:
    return JudgmentEvidence(
        turn_no=turn_no,
        source_speaker_label=label,
        speaker=speaker,
        quote=quote,
    )


def _absent(dimension_id: str) -> DimensionJudgment:
    return DimensionJudgment(
        dimension_id=dimension_id,
        present=False,
        matched_count=0,
        rationale="未命中。",
        evidence=[],
    )


def _present(
    dimension_id: str, count: int, evidence: list[JudgmentEvidence]
) -> DimensionJudgment:
    return DimensionJudgment(
        dimension_id=dimension_id,
        present=True,
        matched_count=count,
        rationale="命中。",
        evidence=evidence,
    )


def _workflow(
    *, judge: FakeModuleJudge | None = None, reviewer: FakeReviewer | None = None
) -> SingleCallWhiteboardScoringWorkflow:
    return SingleCallWhiteboardScoringWorkflow(
        FakeClassifier(), judge or FakeModuleJudge(), reviewer or FakeReviewer()
    )


def test_layered_whiteboard_routes_scores_reviews_and_aggregates() -> None:
    judge = FakeModuleJudge()
    reviewer = FakeReviewer()
    result = _workflow(judge=judge, reviewer=reviewer).score(
        call_id="call-1", sales_stage="销售线索", transcript_text=TRANSCRIPT
    )

    scores = {item.dimension_id: item for item in result.dimensions}
    assert result.schema_version == "single-call-whiteboard-score-v2"
    assert result.call_level == "solution_negotiation"
    assert result.executed_modules == [
        "opening_discovery", "solution_offer", "objection_handling",
        "relationship_interaction",
    ]
    assert set(judge.calls) == set(result.executed_modules)
    assert reviewer.reviewed_ids == [
        "business_introduction", "discovery_information", "objection_unresolved"
    ]
    assert scores["business_introduction"].score == 10
    assert scores["discovery_information"].score == 30
    assert scores["objection_unresolved"].score == -30
    assert scores["contract_delivery_process"].whiteboard_status == "not_applicable"
    assert scores["abnormal_speech"].whiteboard_status == "unassessable"
    assert result.total_score == 10
    assert result.maximum_applicable_positive_score == 180
    assert result.minimum_applicable_negative_score == -30


def test_module_scoring_runs_in_parallel() -> None:
    judge = FakeModuleJudge(Barrier(4))
    result = _workflow(judge=judge).score(
        call_id="call-1", sales_stage="销售线索", transcript_text=TRANSCRIPT
    )
    assert len(result.executed_modules) == 4


def test_reviewer_rejection_zeroes_score_but_keeps_trace() -> None:
    result = _workflow(
        reviewer=FakeReviewer({"business_introduction"})
    ).score(call_id="call-1", sales_stage="销售线索", transcript_text=TRANSCRIPT)
    item = next(
        score for score in result.dimensions
        if score.dimension_id == "business_introduction"
    )
    assert item.whiteboard_status == "review_rejected"
    assert item.review_status == "rejected"
    assert item.present is False
    assert item.score == 0
    assert item.evidence[0].quote == "我们是做企业电脑租赁的"


def test_whiteboard_replaces_model_quote_with_canonical_source_turn() -> None:
    judge = FakeModuleJudge()
    original = judge.judge_module

    def invalid(**kwargs):
        output = original(**kwargs)
        if output.module_id == "opening_discovery":
            output.dimensions[0].evidence[0].quote = "原文中不存在"
        return output

    judge.judge_module = invalid  # type: ignore[method-assign]
    result = _workflow(judge=judge).score(
        call_id="call-1", sales_stage="销售线索", transcript_text=TRANSCRIPT
    )
    item = next(
        score for score in result.dimensions
        if score.dimension_id == "business_introduction"
    )
    assert item.evidence[0].quote == "我们是做企业电脑租赁的"


def test_level_schema_is_strict_and_versioned() -> None:
    schema = strict_model_json_schema(CallLevelClassification)
    assert schema["properties"]["schema_version"]["const"] == "single-call-level-v1"
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("initial_contact", ["opening_discovery", "solution_offer", "objection_handling", "relationship_interaction"]),
        ("needs_discovery", ["opening_discovery", "solution_offer", "objection_handling", "relationship_interaction"]),
        ("solution_negotiation", ["opening_discovery", "solution_offer", "objection_handling", "relationship_interaction"]),
        ("contract_delivery", ["solution_offer", "objection_handling", "relationship_interaction", "fulfillment_support"]),
        ("after_sales", ["objection_handling", "relationship_interaction", "fulfillment_support"]),
    ],
)
def test_every_level_has_a_deterministic_module_route(level, expected) -> None:
    assert [item.module_id for item in modules_for_level(level)] == expected


def test_every_text_rule_belongs_to_a_module_exactly_once() -> None:
    routed = [rule_id for module in WHITEBOARD_MODULES for rule_id in module.rule_ids]
    text_rules = [
        rule.dimension_id for rule in WHITEBOARD_RULES
        if rule.scoring_mode != "audio_required"
    ]
    assert sorted(routed) == sorted(text_rules)
    assert len(routed) == len(set(routed))


def test_single_call_prompts_keep_four_recall_oriented_calibrations() -> None:
    for label in ("介绍业务", "承上启下", "挖到需求或信息", "介绍卖点/优势"):
        assert label in _MODULE_INSTRUCTION
        assert label in _REVIEW_INSTRUCTION
    for example in ("设备都有", "找老板", "随时退", "一台起租", "免费上门", "先用后付"):
        assert example in _MODULE_INSTRUCTION
