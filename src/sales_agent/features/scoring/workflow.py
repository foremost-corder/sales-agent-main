"""Four-section, evidence-grounded and reviewable performance workflow."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
import logging

from sales_agent.features.scoring.contracts import (
    CallScore,
    CallTarget,
    DailyScore,
    PeriodScoreRequest,
    PeriodScoreResult,
    ReviewIssue,
    ReviewReport,
    RuleJudgment,
    RuleScore,
    ScoreEvidence,
    ScoreEvidenceRetriever,
    ScoreReviewAgent,
    ScoreRule,
    ScoreSection,
    ScoringPolicy,
    SectionScoringAgent,
)
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1


logger = logging.getLogger(__name__)


class ScoringWorkflowError(ValueError):
    pass


@dataclass(frozen=True)
class _SectionScoringTask:
    section: ScoreSection
    rules: list[ScoreRule]
    calls: list[CallTarget]
    evidence: list[ScoreEvidence]
    repair_issues: list[ReviewIssue]


class SalesPerformanceScoringWorkflow:
    """Score key behaviors, audit the result, then selectively retrieve and repair."""

    def __init__(
        self,
        *,
        retriever: ScoreEvidenceRetriever,
        section_agents: dict[str, SectionScoringAgent],
        reviewer: ScoreReviewAgent,
        policy: ScoringPolicy = SALES_KEY_BEHAVIOR_POLICY_V1,
        max_parallel_sections: int = 4,
    ) -> None:
        expected = {item.section_id for item in policy.sections}
        if set(section_agents) != expected:
            raise ValueError(f"section_agents must be exactly {sorted(expected)!r}")
        if max_parallel_sections < 1:
            raise ValueError("max_parallel_sections must be at least 1")
        self._retriever = retriever
        self._section_agents = section_agents
        self._reviewer = reviewer
        self._policy = policy
        self._max_parallel_sections = max_parallel_sections
        self._rules = {item.rule_id: item for item in policy.rules}
        self._section_by_rule = {
            rule_id: section
            for section in policy.sections
            for rule_id in section.rule_ids
        }

    def execute_week(
        self,
        *,
        user_id: str,
        sales_id: str,
        week_start: date,
        **overrides,
    ) -> PeriodScoreResult:
        return self.execute(
            PeriodScoreRequest(
                user_id=user_id,
                sales_id=sales_id,
                date_from=week_start,
                date_to=week_start + timedelta(days=6),
                **overrides,
            )
        )

    def execute(self, request: PeriodScoreRequest) -> PeriodScoreResult:
        calls = self._retriever.list_calls(request)
        if not calls:
            return self._empty_result(request)

        evidence_by_pair, judgments, model_calls = self._retrieve_and_score_sections(
            request=request,
            calls=calls,
        )
        call_scores = self._assemble_calls(calls, judgments, evidence_by_pair)
        review = self._reviewer.review(policy=self._policy, calls=call_scores)
        model_calls += 1
        review_history = [review]
        repairs = 0
        budget_exhausted = False

        while (
            review.status == "completed"
            and not review.reasonable
            and repairs < request.max_review_repairs
        ):
            affected_section_ids = {
                self._section_by_rule[issue.rule_id].section_id
                for issue in review.issues
                if issue.rule_id in self._section_by_rule
                and self._rules[issue.rule_id].scoring_mode
                not in {"interaction_metric", "audio_required"}
            }
            required_calls = len(affected_section_ids) + 1
            if model_calls + required_calls > request.max_model_calls:
                budget_exhausted = True
                break
            repairs += 1
            affected_sections, repair_calls = self._supplement_and_repair(
                request=request,
                calls=calls,
                evidence_by_pair=evidence_by_pair,
                judgments=judgments,
                issues=review.issues,
            )
            model_calls += repair_calls
            if not affected_sections:
                break
            call_scores = self._assemble_calls(calls, judgments, evidence_by_pair)
            review = self._reviewer.review(policy=self._policy, calls=call_scores)
            model_calls += 1
            review_history.append(review)

        warnings: list[str] = []
        warnings.append(
            "当前 calls 表没有接通状态字段；工作流暂将日期范围内已导入通话作为候选。"
        )
        if review.status == "completed" and not review.reasonable:
            warnings.append(
                "仅评分审核列出的通话与规则仍需人工复核；其他有可靠证据且未被质疑的得分继续保留。"
            )
        elif review.status == "failed":
            warnings.append("评分主体已完成并保留，但整体模型审核调用失败。")
        if budget_exhausted:
            warnings.append(
                f"返工将超过模型调用预算 {request.max_model_calls}，工作流已停止继续调用。"
            )
        missing = sum(item.analysis_run_id is None for item in calls)
        if missing:
            warnings.append(f"{missing} 通电话在指定知识命名空间中没有成功分析。")
        if any(item.analysis_degraded for item in calls):
            warnings.append("部分通话使用了降级知识分析，评分可靠性已降低。")

        scored = [
            item
            for item in call_scores
            if _has_assessed_rule(item)
        ]
        return PeriodScoreResult(
            sales_id=request.sales_id,
            date_from=request.date_from,
            date_to=request.date_to,
            policy_id=self._policy.policy_id,
            policy_version=self._policy.version,
            knowledge_namespace=request.knowledge_namespace,
            call_count=len(calls),
            scored_call_count=len(scored),
            average_score=(
                round(sum(item.score for item in scored) / len(scored), 2)
                if scored
                else None
            ),
            total_points=sum(item.score for item in scored),
            calls=call_scores,
            daily_scores=_daily_scores(call_scores),
            review=review,
            review_history=review_history,
            review_repairs=repairs,
            model_call_count=model_calls,
            warnings=warnings,
        )

    def _retrieve_and_score_sections(
        self,
        *,
        request: PeriodScoreRequest,
        calls: list[CallTarget],
    ) -> tuple[
        dict[tuple[str, str], list[ScoreEvidence]],
        dict[tuple[str, str], RuleJudgment],
        int,
    ]:
        evidence_by_pair: dict[tuple[str, str], list[ScoreEvidence]] = {}
        judgments: dict[tuple[str, str], RuleJudgment] = {}
        scoring_tasks: list[_SectionScoringTask] = []
        for section in self._policy.sections:
            rules = [self._rules[item] for item in section.rule_ids]
            evidence: list[ScoreEvidence] = []
            for rule in rules:
                retrieved = self._retriever.retrieve_period(
                    request=request,
                    calls=calls,
                    rule=rule,
                )
                evidence.extend(retrieved)
                by_call: dict[str, list[ScoreEvidence]] = defaultdict(list)
                for item in retrieved:
                    by_call[item.call_id].append(item)
                for call in calls:
                    evidence_by_pair[(call.call_id, rule.rule_id)] = by_call.get(
                        call.call_id, []
                    )

            logger.info(
                "scoring_section_period section=%s calls=%s evidence=%s",
                section.section_id,
                len(calls),
                len(evidence),
            )
            if all(
                rule.scoring_mode in {"interaction_metric", "audio_required"}
                for rule in rules
            ):
                self._apply_computed_judgments(evidence, judgments)
            elif evidence:
                relevant_ids = {item.call_id for item in evidence}
                relevant_calls = [call for call in calls if call.call_id in relevant_ids]
                scoring_tasks.append(
                    _SectionScoringTask(
                        section=section,
                        rules=rules,
                        calls=relevant_calls,
                        evidence=evidence,
                        repair_issues=[],
                    )
                )

        for task, returned in self._run_section_tasks(scoring_tasks):
            judgments.update(
                self._normalize_judgments(task.calls, task.rules, returned)
            )

        for section in self._policy.sections:
            rules = [self._rules[item] for item in section.rule_ids]
            for call in calls:
                for rule in rules:
                    judgments.setdefault(
                        (call.call_id, rule.rule_id),
                        RuleJudgment(
                            call_id=call.call_id,
                            rule_id=rule.rule_id,
                            matched_evidence_ids=[],
                            rationale="该周期整体召回中没有足以命中该规则的证据。",
                        ),
                    )
        return evidence_by_pair, judgments, len(scoring_tasks)

    def _run_section_tasks(
        self,
        tasks: list[_SectionScoringTask],
    ) -> list[tuple[_SectionScoringTask, list[RuleJudgment]]]:
        if not tasks:
            return []
        worker_count = min(self._max_parallel_sections, len(tasks))
        logger.info(
            "scoring_sections_parallel tasks=%s workers=%s",
            len(tasks),
            worker_count,
        )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="scoring-section",
        ) as executor:
            futures = [
                executor.submit(
                    self._section_agents[task.section.section_id].score_section,
                    section=task.section,
                    rules=task.rules,
                    calls=task.calls,
                    evidence=task.evidence,
                    repair_issues=task.repair_issues,
                )
                for task in tasks
            ]
            # Resolve in policy order for deterministic merging and error reporting.
            return [
                (task, future.result())
                for task, future in zip(tasks, futures, strict=True)
            ]

    @staticmethod
    def _apply_computed_judgments(
        evidence: list[ScoreEvidence],
        judgments: dict[tuple[str, str], RuleJudgment],
    ) -> None:
        for item in evidence:
            if item.retrieved_for_rule_id != "conversation_interaction":
                continue
            matched = "规则命中=True" in item.fact
            judgments[(item.call_id, item.retrieved_for_rule_id)] = RuleJudgment(
                call_id=item.call_id,
                rule_id=item.retrieved_for_rule_id,
                matched_evidence_ids=[item.evidence_id] if matched else [],
                rationale="根据本地确定性互动指标判断。",
            )

    def _supplement_and_repair(
        self,
        *,
        request: PeriodScoreRequest,
        calls: list[CallTarget],
        evidence_by_pair: dict[tuple[str, str], list[ScoreEvidence]],
        judgments: dict[tuple[str, str], RuleJudgment],
        issues: list[ReviewIssue],
    ) -> tuple[set[str], int]:
        call_map = {item.call_id: item for item in calls}
        by_section: dict[str, list[ReviewIssue]] = defaultdict(list)
        for issue in issues:
            section = self._section_by_rule.get(issue.rule_id)
            call = call_map.get(issue.call_id)
            rule = self._rules.get(issue.rule_id)
            if section is None or call is None or rule is None:
                continue
            by_section[section.section_id].append(issue)
            if issue.supplemental_queries:
                # A review issue is scoped to one judgment. Never let its repair
                # retrieve or re-score unrelated calls from the same period.
                extra = self._retriever.retrieve_period(
                    request=request,
                    calls=[call],
                    rule=rule,
                    queries=issue.supplemental_queries,
                )
                for extra_item in extra:
                    pair = (extra_item.call_id, rule.rule_id)
                    evidence_by_pair[pair] = _merge_evidence(
                        evidence_by_pair[pair], [extra_item]
                    )

        repair_tasks: list[_SectionScoringTask] = []
        affected_sections: set[str] = set()
        for section_id, section_issues in by_section.items():
            section = next(item for item in self._policy.sections if item.section_id == section_id)
            affected_rule_ids = {item.rule_id for item in section_issues}
            rules = [
                self._rules[item]
                for item in section.rule_ids
                if item in affected_rule_ids
            ]
            affected_call_ids = {item.call_id for item in section_issues}
            affected_calls = [
                call for call in calls if call.call_id in affected_call_ids
            ]
            if not affected_calls:
                continue
            affected_sections.add(section_id)
            evidence = [
                item
                for call in affected_calls
                for rule in rules
                for item in evidence_by_pair[(call.call_id, rule.rule_id)]
            ]
            if all(
                rule.scoring_mode in {"interaction_metric", "audio_required"}
                for rule in rules
            ):
                self._apply_computed_judgments(evidence, judgments)
                continue
            repair_tasks.append(
                _SectionScoringTask(
                    section=section,
                    rules=rules,
                    calls=affected_calls,
                    evidence=evidence,
                    repair_issues=section_issues,
                )
            )

        for task, returned in self._run_section_tasks(repair_tasks):
            judgments.update(
                self._normalize_judgments(task.calls, task.rules, returned)
            )
        return affected_sections, len(repair_tasks)

    @staticmethod
    def _normalize_judgments(
        calls: list[CallTarget], rules: list[ScoreRule], returned: list[RuleJudgment]
    ) -> dict[tuple[str, str], RuleJudgment]:
        valid_pairs = {(call.call_id, rule.rule_id) for call in calls for rule in rules}
        result: dict[tuple[str, str], RuleJudgment] = {}
        for judgment in returned:
            pair = (judgment.call_id, judgment.rule_id)
            if pair not in valid_pairs or pair in result:
                continue
            result[pair] = judgment
        for call_id, rule_id in valid_pairs:
            result.setdefault(
                (call_id, rule_id),
                RuleJudgment(
                    call_id=call_id,
                    rule_id=rule_id,
                    matched_evidence_ids=[],
                    rationale="评分代理未返回该规则判断。",
                ),
            )
        return result

    def _assemble_calls(
        self,
        calls: list[CallTarget],
        judgments: dict[tuple[str, str], RuleJudgment],
        evidence_by_pair: dict[tuple[str, str], list[ScoreEvidence]],
    ) -> list[CallScore]:
        result: list[CallScore] = []
        for call in calls:
            rule_scores: list[RuleScore] = []
            warnings: list[str] = []
            for rule in self._policy.rules:
                candidates = evidence_by_pair[(call.call_id, rule.rule_id)]
                judgment = judgments[(call.call_id, rule.rule_id)]
                allowed = {item.evidence_id: item for item in candidates}
                selected: list[ScoreEvidence] = []
                for evidence_id in dict.fromkeys(judgment.matched_evidence_ids):
                    evidence = allowed.get(evidence_id)
                    if (
                        evidence is not None
                        and evidence.retrieved_for_rule_id == rule.rule_id
                        and not (
                            rule.scoring_mode == "interaction_metric"
                            and "规则命中=True" not in evidence.fact
                        )
                    ):
                        selected.append(evidence)
                if len(selected) != len(set(judgment.matched_evidence_ids)):
                    warnings.append(f"{rule.rule_id}: 评分代理引用了无效证据，已丢弃。")
                matched_count = len(selected)
                if rule.scoring_mode == "binary" and matched_count > 1:
                    selected = selected[:1]
                    matched_count = 1
                points = _calculate_points(rule, matched_count)
                if selected:
                    status = "scored"
                elif rule.scoring_mode == "audio_required" and not candidates:
                    status = "not_applicable"
                elif not candidates or call.analysis_run_id is None:
                    status = "insufficient_evidence"
                else:
                    status = "not_met"
                rule_scores.append(
                    RuleScore(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        status=status,
                        matched_count=matched_count,
                        points=points,
                        evidence=selected,
                        evidence_reliability=_evidence_reliability(selected),
                        rationale=judgment.rationale,
                    )
                )
            if call.analysis_run_id is None:
                warnings.append("该通话没有可用的知识分析结果。")
            if call.analysis_degraded:
                warnings.append("该通话的知识分析为降级结果。")
            applicable = [
                item for item in rule_scores if item.status != "not_applicable"
            ]
            assessed = [
                item for item in applicable if item.status in {"scored", "not_met"}
            ]
            assessment_coverage = len(assessed) / len(applicable) if applicable else 0.0
            if call.analysis_run_id is None:
                reliability = "low"
            elif call.analysis_degraded:
                reliability = "medium"
            else:
                reliability = "high"
            result.append(
                CallScore(
                    call_id=call.call_id,
                    external_call_id=call.external_call_id,
                    call_date=call.call_date,
                    sales_stage=call.sales_stage,
                    score=sum(item.points for item in rule_scores),
                    rule_scores=rule_scores,
                    reliability=reliability,
                    assessment_coverage=round(assessment_coverage, 4),
                    assessed_rule_count=len(assessed),
                    applicable_rule_count=len(applicable),
                    warnings=warnings,
                )
            )
        return result

    def _empty_result(self, request: PeriodScoreRequest) -> PeriodScoreResult:
        return PeriodScoreResult(
            sales_id=request.sales_id,
            date_from=request.date_from,
            date_to=request.date_to,
            policy_id=self._policy.policy_id,
            policy_version=self._policy.version,
            knowledge_namespace=request.knowledge_namespace,
            call_count=0,
            scored_call_count=0,
            average_score=None,
            total_points=0,
            calls=[],
            daily_scores=[],
            review=ReviewReport(
                status="completed",
                reasonable=True,
                issues=[],
                summary="指定范围内没有通话。",
            ),
            review_history=[
                ReviewReport(
                    status="completed",
                    reasonable=True,
                    issues=[],
                    summary="指定范围内没有通话。",
                )
            ],
            review_repairs=0,
            model_call_count=0,
            warnings=["指定销售和日期范围内没有通话。"],
        )


def _calculate_points(rule: ScoreRule, matched_count: int) -> int:
    if matched_count == 0:
        return 0
    if rule.scoring_mode == "count_capped":
        raw = matched_count * rule.points_per_match
        return min(raw, rule.maximum_points) if raw > 0 else max(raw, rule.maximum_points)
    return rule.points_per_match


def _evidence_reliability(
    evidence: list[ScoreEvidence],
) -> str:
    """Rate evidence grounding separately from analysis and assessment coverage."""
    if not evidence:
        return "not_scored"
    if any(item.confidence < 0.5 for item in evidence):
        return "low"
    if any(
        item.confidence < 0.8 or item.grounding == "turn_only"
        for item in evidence
    ):
        return "medium"
    return "high"


def _has_assessed_rule(call: CallScore) -> bool:
    return any(item.status in {"scored", "not_met"} for item in call.rule_scores)


def _merge_evidence(
    original: list[ScoreEvidence], extra: list[ScoreEvidence]
) -> list[ScoreEvidence]:
    by_id = {item.evidence_id: item for item in original}
    for item in extra:
        previous = by_id.get(item.evidence_id)
        if previous is None or (item.similarity or -1) > (previous.similarity or -1):
            by_id[item.evidence_id] = item
    return list(by_id.values())


def _daily_scores(calls: list[CallScore]) -> list[DailyScore]:
    by_date: dict[date, list[CallScore]] = defaultdict(list)
    for call in calls:
        by_date[call.call_date].append(call)
    result: list[DailyScore] = []
    for call_date, daily_calls in sorted(by_date.items()):
        scored = [
            call
            for call in daily_calls
            if _has_assessed_rule(call)
        ]
        result.append(
            DailyScore(
                call_date=call_date,
                call_count=len(daily_calls),
                scored_call_count=len(scored),
                total_points=sum(call.score for call in scored),
                average_score=(
                    round(sum(call.score for call in scored) / len(scored), 2)
                    if scored
                    else None
                ),
            )
        )
    return result
