"""Score one range against baseline and lean knowledge namespaces."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from sales_agent.agent.chat_model import ChatModelRequestError
from sales_agent.core.config import get_settings
from sales_agent.core.database import SessionLocal
from sales_agent.features.knowledge.namespaces import (
    BASELINE_KNOWLEDGE_NAMESPACE,
    LEAN_KNOWLEDGE_NAMESPACE,
)
from sales_agent.features.scoring.contracts import PeriodScoreRequest
from sales_agent.features.scoring.model_agents import (
    OpenAIScoreReviewAgent,
    OpenAISectionScoringAgent,
    ScoringAgentError,
)
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1
from sales_agent.features.scoring.retrieval import PgVectorScoreEvidenceRetriever
from sales_agent.features.scoring.rendering import render_period_score_summary
from sales_agent.features.scoring.workflow import SalesPerformanceScoringWorkflow
from sales_agent.integrations.embeddings.factory import build_embedding_provider


class LimitedRetriever:
    def __init__(self, delegate: PgVectorScoreEvidenceRetriever, limit: int | None) -> None:
        self._delegate = delegate
        self._limit = limit

    def list_calls(self, request):
        calls = self._delegate.list_calls(request)
        return calls[: self._limit] if self._limit is not None else calls

    def retrieve(self, **kwargs):
        return self._delegate.retrieve(**kwargs)

    def retrieve_period(self, **kwargs):
        return self._delegate.retrieve_period(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evidence-grounded sales scoring independently against the standard "
            "and lean knowledge namespaces."
        )
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--sales-id", required=True)
    parser.add_argument("--date-from", type=date.fromisoformat, required=True)
    parser.add_argument("--date-to", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--namespace",
        action="append",
        choices=(BASELINE_KNOWLEDGE_NAMESPACE, LEAN_KNOWLEDGE_NAMESPACE),
        dest="namespaces",
        help=(
            "Knowledge namespace to score; repeat to run both. "
            "The default runs baseline-react-v4 and experiment-lean-v1."
        ),
    )
    parser.add_argument("--sample-limit", type=int, choices=range(1, 11), metavar="1-10")
    parser.add_argument(
        "--include-low-confidence",
        action="store_true",
        help=(
            "Compatibility flag; low-confidence facts are already traced back to "
            "original transcript context by default and never scored directly."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Model used only by this scoring run. Overrides SCORING_MODEL/CHAT_MODEL "
            "without changing the chat Agent."
        ),
    )
    parser.add_argument("--max-review-repairs", type=int, choices=range(0, 4), default=2)
    parser.add_argument(
        "--max-model-calls",
        type=int,
        choices=range(13, 31),
        default=18,
        metavar="13-30",
        help="Logical model-call budget for scoring, review, and rework (default: 18).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Output score-centric weekly aggregates without per-call analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the final JSON result to this file instead of stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"scoring_model": args.model})
    namespaces = list(
        dict.fromkeys(
            args.namespaces
            or [BASELINE_KNOWLEDGE_NAMESPACE, LEAN_KNOWLEDGE_NAMESPACE]
        )
    )
    results = []
    with SessionLocal() as session:
        base_retriever = PgVectorScoreEvidenceRetriever(
            session, embedder=build_embedding_provider(settings)
        )
        retriever = LimitedRetriever(base_retriever, args.sample_limit)
        for namespace in namespaces:
            logging.info(
                "scoring_run namespace=%s model=%s endpoint=%s",
                namespace,
                settings.scoring_model_name,
                settings.openai_base_url or "https://api.openai.com/v1",
            )
            workflow = SalesPerformanceScoringWorkflow(
                retriever=retriever,
                section_agents={
                    section.section_id: OpenAISectionScoringAgent(settings)
                    for section in SALES_KEY_BEHAVIOR_POLICY_V1.sections
                },
                reviewer=OpenAIScoreReviewAgent(settings),
            )
            try:
                result = workflow.execute(
                    PeriodScoreRequest(
                        user_id=args.user_id,
                        sales_id=args.sales_id,
                        date_from=args.date_from,
                        date_to=args.date_to,
                        knowledge_namespace=namespace,
                        include_low_confidence=args.include_low_confidence,
                        max_review_repairs=args.max_review_repairs,
                        max_model_calls=args.max_model_calls,
                    )
                )
            except (ChatModelRequestError, ScoringAgentError) as exc:
                failure = {
                    "event": "scoring_failed",
                    "namespace": namespace,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                    "request_id": getattr(exc, "request_id", None),
                    "provider_code": getattr(exc, "provider_code", None),
                    "provider_message": getattr(exc, "provider_message", None),
                    "attempts": getattr(exc, "attempts", None),
                    "hint": _failure_hint(exc),
                }
                print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
                return 1
            results.append(result)

    rendered = [_render_result(result, summary=args.summary) for result in results]
    identity = {
        "sales_id": args.sales_id,
        "date_from": args.date_from.isoformat(),
        "date_to": args.date_to.isoformat(),
        "model": settings.scoring_model_name,
    }
    if args.summary and len(rendered) == 1:
        payload = {**identity, **rendered[0]}
    else:
        payload = {**identity, "results": rendered}
        comparison = _compare(results, detailed=not args.summary)
        if comparison is not None:
            payload["comparison"] = comparison
        if args.sample_limit is not None:
            payload["sample_limit"] = args.sample_limit
    rendered_json = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered_json, encoding="utf-8")
        print(f"scoring result written to {args.output.resolve()}")
    else:
        print(rendered_json)
    return 0 if all(result.review.reasonable for result in results) else 2


def _render_result(result, *, summary: bool) -> dict:
    if not summary:
        return result.model_dump(mode="json")
    return render_period_score_summary(result)


def _failure_hint(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    provider_code = getattr(exc, "provider_code", None)
    if provider_code == "model_not_found" or status_code == 503:
        return "Choose a model currently enabled by the configured provider with --model."
    if status_code == 400:
        return (
            "The model route exists but rejected this request. Inspect provider_code and "
            "provider_message; common causes are context limits or provider content checks."
        )
    if status_code in {408, 409, 425, 429} or (
        isinstance(status_code, int) and 500 <= status_code <= 599
    ):
        return (
            "The upstream model remained unavailable after automatic retries. Retry the "
            "scoring run later; if it persists, inspect gateway availability and limits."
        )
    return "Inspect provider_code and provider_message for the upstream failure reason."


def _compare(results, *, detailed: bool) -> dict | None:
    if len(results) != 2:
        return None
    first, second = results
    first_calls = {item.call_id: item for item in first.calls}
    second_calls = {item.call_id: item for item in second.calls}
    common_ids = sorted(set(first_calls) & set(second_calls))
    comparison = {
        "from_namespace": first.knowledge_namespace,
        "to_namespace": second.knowledge_namespace,
        "average_score_delta": (
            round(second.average_score - first.average_score, 2)
            if first.average_score is not None and second.average_score is not None
            else None
        ),
        "common_call_count": len(common_ids),
        "only_in_from_count": len(set(first_calls) - set(second_calls)),
        "only_in_to_count": len(set(second_calls) - set(first_calls)),
    }
    if detailed:
        comparison["call_score_deltas"] = [
            {
                "call_id": call_id,
                "external_call_id": first_calls[call_id].external_call_id,
                "from_score": first_calls[call_id].score,
                "to_score": second_calls[call_id].score,
                "delta": second_calls[call_id].score - first_calls[call_id].score,
            }
            for call_id in common_ids
        ]
    return comparison


if __name__ == "__main__":
    raise SystemExit(main())
