"""Score-centric renderers kept separate from CLI and model adapters."""

from sales_agent.features.scoring.contracts import PeriodScoreResult
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1


def render_period_score_summary(result: PeriodScoreResult) -> dict:
    behavior_scores = []
    for policy_rule in SALES_KEY_BEHAVIOR_POLICY_V1.rules:
        scores = [
            rule_score
            for call in result.calls
            for rule_score in call.rule_scores
            if rule_score.rule_id == policy_rule.rule_id
        ]
        total_points = sum(score.points for score in scores)
        behavior_scores.append(
            {
                "rule_id": policy_rule.rule_id,
                "name": policy_rule.name,
                "score": (
                    round(total_points / result.scored_call_count, 2)
                    if result.scored_call_count
                    else None
                ),
                "total_points": total_points,
                "hit_call_count": sum(score.points != 0 for score in scores),
                "evidence_count": sum(score.matched_count for score in scores),
            }
        )

    reliability_counts = {
        level: sum(call.reliability == level for call in result.calls)
        for level in ("high", "medium", "low")
    }
    evidence_reliability_counts = {
        level: sum(
            score.evidence_reliability == level
            for call in result.calls
            for score in call.rule_scores
        )
        for level in ("high", "medium", "low")
    }
    rule_status_counts = {
        status: sum(
            score.status == status
            for call in result.calls
            for score in call.rule_scores
        )
        for status in (
            "scored",
            "not_met",
            "insufficient_evidence",
            "not_applicable",
        )
    }
    issue_counts: dict[str, int] = {}
    for issue in result.review.issues:
        issue_counts[issue.rule_id] = issue_counts.get(issue.rule_id, 0) + 1
    issue_pairs = {
        (issue.call_id, issue.rule_id) for issue in result.review.issues
    }
    contested_scores = [
        score
        for call in result.calls
        for score in call.rule_scores
        if (call.call_id, score.rule_id) in issue_pairs
    ]
    contested_points = sum(score.points for score in contested_scores)
    return {
        "policy_id": result.policy_id,
        "policy_version": result.policy_version,
        "knowledge_namespace": result.knowledge_namespace,
        "weekly_score": result.average_score,
        "total_points": result.total_points,
        "call_count": result.call_count,
        "scored_call_count": result.scored_call_count,
        "score_coverage": (
            round(result.scored_call_count / result.call_count, 4)
            if result.call_count
            else 0.0
        ),
        "reliability_counts": reliability_counts,
        "average_assessment_coverage": (
            round(
                sum(call.assessment_coverage for call in result.calls)
                / result.call_count,
                4,
            )
            if result.call_count
            else 0.0
        ),
        "evidence_reliability_counts": evidence_reliability_counts,
        "rule_status_counts": rule_status_counts,
        "behavior_scores": behavior_scores,
        "review": {
            "status": result.review.status,
            "reasonable": result.review.reasonable,
            "repairs": result.review_repairs,
            "issue_count": len(result.review.issues),
            "issue_counts_by_rule": issue_counts,
            "contested_call_count": len({item.call_id for item in result.review.issues}),
            "contested_points": contested_points,
            "uncontested_points": result.total_points - contested_points,
        },
        "model_call_count": result.model_call_count,
    }
