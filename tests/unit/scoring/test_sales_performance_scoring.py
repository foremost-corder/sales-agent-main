from datetime import date
from threading import Barrier

from sales_agent.features.scoring.contracts import (
    CallTarget,
    PeriodScoreRequest,
    ReviewIssue,
    ReviewReport,
    RuleJudgment,
    ScoreEvidence,
)
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1
from sales_agent.features.scoring.rendering import render_period_score_summary
from sales_agent.features.scoring.workflow import SalesPerformanceScoringWorkflow


CALL_ID = "00000000-0000-0000-0000-000000000001"
RUN_ID = "00000000-0000-0000-0000-000000000002"


class FakeRetriever:
    def __init__(self) -> None:
        self.supplemental_queries: list[str] = []

    def list_calls(self, request):
        assert request.sales_id == "001"
        return [
            CallTarget(
                call_id=CALL_ID,
                external_call_id="call-1",
                call_date=date(2026, 9, 8),
                sales_stage="销售线索",
                analysis_run_id=RUN_ID,
            )
        ]

    def retrieve(self, *, request, call, rule, queries=None):
        if queries:
            self.supplemental_queries.extend(queries)
            if rule.rule_id == "product_recommendation":
                return [_evidence(rule.rule_id, 1, "销售根据办公用途推荐了 i5、16G 配置", call.call_id)]
        if rule.rule_id == "business_introduction":
            return [_evidence(rule.rule_id, 1, "销售介绍公司从事电脑租赁", call.call_id)]
        if rule.rule_id == "discovery_information":
            return [
                _evidence(rule.rule_id, index, f"客户需求信息 {index}", call.call_id)
                for index in range(1, 7)
            ]
        if rule.rule_id == "objection_unresolved":
            return [_evidence(rule.rule_id, 1, "客户拒绝后销售没有给出解决方案", call.call_id)]
        if rule.rule_id == "conversation_interaction":
            return [_metric_evidence(rule.rule_id, call.call_id)]
        return []

    def retrieve_period(self, *, request, calls, rule, queries=None):
        return [
            evidence
            for call in calls
            for evidence in self.retrieve(
                request=request, call=call, rule=rule, queries=queries
            )
        ]


class FakeSectionAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.seen_rule_ids: list[set[str]] = []

    def score_section(self, *, section, rules, calls, evidence, repair_issues=()):
        self.calls += 1
        self.batch_sizes.append(len(calls))
        self.seen_rule_ids.append(
            {item.retrieved_for_rule_id for item in evidence}
        )
        by_pair = {}
        for item in evidence:
            by_pair.setdefault((item.call_id, item.retrieved_for_rule_id), []).append(
                item.evidence_id
            )
        result = []
        for call in calls:
            for rule in rules:
                ids = by_pair.get((call.call_id, rule.rule_id), [])
                should_match = rule.rule_id in {
                    "business_introduction",
                    "discovery_information",
                    "objection_unresolved",
                    "conversation_interaction",
                }
                if repair_issues and rule.rule_id == "product_recommendation":
                    should_match = True
                result.append(
                    RuleJudgment(
                        call_id=call.call_id,
                        rule_id=rule.rule_id,
                        matched_evidence_ids=ids if should_match else [],
                        rationale="有证据" if should_match else "没有满足规则的证据",
                    )
                )
        return result


class BarrierSectionAgent(FakeSectionAgent):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self._barrier = barrier

    def score_section(self, **kwargs):
        self._barrier.wait(timeout=2)
        return super().score_section(**kwargs)


class RepairingReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, *, policy, calls):
        self.calls += 1
        if self.calls == 1:
            return ReviewReport(
                status="completed",
                reasonable=False,
                issues=[
                    ReviewIssue(
                        issue_id="review-1",
                        call_id=CALL_ID,
                        rule_id="product_recommendation",
                        reason="需要补查推荐是否建立在客户用途之上",
                        supplemental_queries=["客户用途与销售配置推荐之间的对应关系"],
                    )
                ],
                summary="产品推荐证据不完整，需要定向补召回。",
            )
        return ReviewReport(
            status="completed",
            reasonable=True,
            issues=[],
            summary="返工后的评分合理。",
        )


class AlwaysReasonableReviewer:
    def review(self, *, policy, calls):
        return ReviewReport(
            status="completed", reasonable=True, issues=[], summary="评分合理。"
        )


class BatchRecordingReviewer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def review(self, *, policy, calls):
        self.batch_sizes.append(len(calls))
        return ReviewReport(
            status="completed", reasonable=True, issues=[], summary="本批评分合理。"
        )


class FiveCallRetriever(FakeRetriever):
    def list_calls(self, request):
        return [
            CallTarget(
                call_id=f"00000000-0000-0000-0000-{index:012d}",
                external_call_id=f"call-{index}",
                call_date=date(2026, 9, 8),
                sales_stage="销售线索",
                analysis_run_id=RUN_ID,
            )
            for index in range(1, 6)
        ]

def _evidence(
    rule_id: str, index: int, text: str, call_id: str = CALL_ID
) -> ScoreEvidence:
    return ScoreEvidence(
        evidence_id=f"{rule_id}:fact-{index}",
        retrieved_for_rule_id=rule_id,
        source="vector_fact",
        call_id=call_id,
        analysis_run_id=RUN_ID,
        fact_id=f"fact-{index}",
        fact_type=rule_id,
        speaker="sales",
        fact=text,
        turn_no=index,
        quote=text,
        grounding="exact",
        confidence=1.0,
        similarity=0.9,
    )


def _metric_evidence(rule_id: str, call_id: str = CALL_ID) -> ScoreEvidence:
    return ScoreEvidence(
        evidence_id=f"{rule_id}:metric:{call_id}",
        retrieved_for_rule_id=rule_id,
        source="call_metric",
        call_id=call_id,
        analysis_run_id=RUN_ID,
        fact_id=None,
        fact_type="computed_interaction_metric",
        speaker=None,
        fact="互动来回=3，规则命中=True",
        turn_no=None,
        quote="互动来回=3，规则命中=True",
        grounding="computed",
        confidence=1.0,
    )


def _workflow(retriever, reviewer):
    agents = {
        section.section_id: FakeSectionAgent()
        for section in SALES_KEY_BEHAVIOR_POLICY_V1.sections
    }
    return (
        SalesPerformanceScoringWorkflow(
            retriever=retriever,
            section_agents=agents,
            reviewer=reviewer,
        ),
        agents,
    )


def test_policy_assigns_each_key_behavior_to_an_independent_dimension() -> None:
    policy = SALES_KEY_BEHAVIOR_POLICY_V1
    assert len(policy.sections) == 14
    assert all(len(section.rule_ids) == 1 for section in policy.sections)
    assert {rule_id for section in policy.sections for rule_id in section.rule_ids} == {
        rule.rule_id for rule in policy.rules
    }


def test_workflow_scores_each_call_caps_counts_and_applies_deductions() -> None:
    workflow, _ = _workflow(FakeRetriever(), AlwaysReasonableReviewer())
    result = workflow.execute(
        PeriodScoreRequest(
            user_id="tenant-a",
            sales_id="001",
            date_from=date(2026, 9, 7),
            date_to=date(2026, 9, 13),
        )
    )

    assert result.call_count == 1
    assert result.average_score == 50.0  # 10 + min(6*10, 50) + 20 - 30
    assert result.daily_scores[0].average_score == 50.0
    discovery = next(
        item for item in result.calls[0].rule_scores if item.rule_id == "discovery_information"
    )
    assert discovery.matched_count == 6
    assert discovery.points == 50
    audio = next(
        item for item in result.calls[0].rule_scores if item.rule_id == "abnormal_speech"
    )
    assert audio.status == "not_applicable"
    assert audio.points == 0
    assert result.calls[0].reliability == "high"
    assert result.calls[0].assessed_rule_count == 4
    assert result.calls[0].applicable_rule_count == 13
    assert result.calls[0].assessment_coverage == 0.3077
    assert discovery.evidence_reliability == "high"


def test_review_issue_triggers_targeted_retrieval_and_only_section_rework() -> None:
    retriever = FakeRetriever()
    reviewer = RepairingReviewer()
    workflow, agents = _workflow(retriever, reviewer)
    result = workflow.execute_week(
        user_id="tenant-a",
        sales_id="001",
        week_start=date(2026, 9, 7),
    )

    assert result.date_to == date(2026, 9, 13)
    assert result.review.reasonable is True
    assert result.review_repairs == 1
    assert result.review_history[0].reasonable is False
    assert "定向补召回" in result.review_history[0].summary
    assert result.average_score == 60.0
    assert retriever.supplemental_queries == ["客户用途与销售配置推荐之间的对应关系"]
    assert agents["product_recommendation"].calls == 1
    assert agents["business_introduction"].calls == 1
    assert agents["discovery_information"].calls == 1
    assert agents["objection_unresolved"].calls == 1


def test_review_repair_only_rescores_the_issue_call() -> None:
    retriever = FiveCallRetriever()
    reviewer = RepairingReviewer()
    workflow, agents = _workflow(retriever, reviewer)

    result = workflow.execute(
        PeriodScoreRequest(
            user_id="tenant-a",
            sales_id="001",
            date_from=date(2026, 9, 7),
            date_to=date(2026, 9, 13),
        )
    )

    assert result.review.reasonable is True
    assert agents["product_recommendation"].batch_sizes == [1]


def test_each_dimension_uses_one_period_level_model_call() -> None:
    reviewer = BatchRecordingReviewer()
    workflow, agents = _workflow(FiveCallRetriever(), reviewer)
    result = workflow.execute(
        PeriodScoreRequest(
            user_id="tenant-a",
            sales_id="001",
            date_from=date(2026, 9, 7),
            date_to=date(2026, 9, 13),
        )
    )

    assert result.call_count == 5
    assert reviewer.batch_sizes == [5]
    assert result.model_call_count == 4
    for section in SALES_KEY_BEHAVIOR_POLICY_V1.sections:
        agent = agents[section.section_id]
        expected = [5] if section.section_id in {
            "business_introduction",
            "discovery_information",
            "objection_unresolved",
        } else []
        assert agent.batch_sizes == expected
        assert all(rule_ids <= set(section.rule_ids) for rule_ids in agent.seen_rule_ids)


def test_independent_section_model_calls_run_in_parallel() -> None:
    retriever = FakeRetriever()
    reviewer = AlwaysReasonableReviewer()
    barrier = Barrier(2)
    agents = {
        section.section_id: FakeSectionAgent()
        for section in SALES_KEY_BEHAVIOR_POLICY_V1.sections
    }
    agents["business_introduction"] = BarrierSectionAgent(barrier)
    agents["discovery_information"] = BarrierSectionAgent(barrier)
    workflow = SalesPerformanceScoringWorkflow(
        retriever=retriever,
        section_agents=agents,
        reviewer=reviewer,
        max_parallel_sections=2,
    )

    result = workflow.execute(
        PeriodScoreRequest(
            user_id="tenant-a",
            sales_id="001",
            date_from=date(2026, 9, 7),
            date_to=date(2026, 9, 13),
        )
    )

    assert result.average_score == 50.0
    assert result.model_call_count == 4
    assert agents["business_introduction"].calls == 1
    assert agents["discovery_information"].calls == 1


def test_parallel_section_limit_must_be_positive() -> None:
    agents = {
        section.section_id: FakeSectionAgent()
        for section in SALES_KEY_BEHAVIOR_POLICY_V1.sections
    }

    try:
        SalesPerformanceScoringWorkflow(
            retriever=FakeRetriever(),
            section_agents=agents,
            reviewer=AlwaysReasonableReviewer(),
            max_parallel_sections=0,
        )
    except ValueError as exc:
        assert str(exc) == "max_parallel_sections must be at least 1"
    else:
        raise AssertionError("expected max_parallel_sections validation to fail")


def test_summary_contains_scores_without_per_call_analysis() -> None:
    workflow, _ = _workflow(FakeRetriever(), AlwaysReasonableReviewer())
    result = workflow.execute(
        PeriodScoreRequest(
            user_id="tenant-a",
            sales_id="001",
            date_from=date(2026, 9, 7),
            date_to=date(2026, 9, 13),
        )
    )

    summary = render_period_score_summary(result)

    assert summary["weekly_score"] == 50.0
    assert summary["reliability_counts"] == {"high": 1, "medium": 0, "low": 0}
    assert summary["average_assessment_coverage"] == 0.3077
    assert summary["evidence_reliability_counts"] == {
        "high": 4,
        "medium": 0,
        "low": 0,
    }
    assert summary["rule_status_counts"] == {
        "scored": 4,
        "not_met": 0,
        "insufficient_evidence": 9,
        "not_applicable": 1,
    }
    assert summary["review"]["contested_call_count"] == 0
    assert summary["review"]["contested_points"] == 0
    assert summary["review"]["uncontested_points"] == 50
    assert len(summary["behavior_scores"]) == 14
    assert "calls" not in summary
    assert "daily_scores" not in summary
    assert "warnings" not in summary
    assert "summary" not in summary["review"]
