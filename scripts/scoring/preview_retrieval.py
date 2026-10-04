"""Preview period-level scoring retrieval without calling a language model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date

from sales_agent.core.config import get_settings
from sales_agent.core.database import SessionLocal
from sales_agent.features.knowledge.namespaces import (
    BASELINE_KNOWLEDGE_NAMESPACE,
    LEAN_KNOWLEDGE_NAMESPACE,
)
from sales_agent.features.scoring.contracts import PeriodScoreRequest
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1
from sales_agent.features.scoring.retrieval import PgVectorScoreEvidenceRetriever
from sales_agent.integrations.embeddings.factory import build_embedding_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--sales-id", required=True)
    parser.add_argument("--date-from", type=date.fromisoformat, required=True)
    parser.add_argument("--date-to", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--namespace",
        choices=(BASELINE_KNOWLEDGE_NAMESPACE, LEAN_KNOWLEDGE_NAMESPACE),
        default=BASELINE_KNOWLEDGE_NAMESPACE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = PeriodScoreRequest(
        user_id=args.user_id,
        sales_id=args.sales_id,
        date_from=args.date_from,
        date_to=args.date_to,
        knowledge_namespace=args.namespace,
    )
    settings = get_settings()
    with SessionLocal() as session:
        retriever = PgVectorScoreEvidenceRetriever(
            session, embedder=build_embedding_provider(settings)
        )
        calls = retriever.list_calls(request)
        section_results = []
        rules = {item.rule_id: item for item in SALES_KEY_BEHAVIOR_POLICY_V1.rules}
        for section in SALES_KEY_BEHAVIOR_POLICY_V1.sections:
            rule_results = []
            for rule_id in section.rule_ids:
                evidence = retriever.retrieve_period(
                    request=request,
                    calls=calls,
                    rule=rules[rule_id],
                )
                rule_results.append(
                    {
                        "rule_id": rule_id,
                        "evidence_count": len(evidence),
                        "covered_call_count": len({item.call_id for item in evidence}),
                        "sources": dict(Counter(item.source for item in evidence)),
                    }
                )
            section_results.append(
                {"section_id": section.section_id, "rules": rule_results}
            )
    print(
        json.dumps(
            {
                "namespace": args.namespace,
                "call_count": len(calls),
                "analyzed_call_count": sum(
                    item.analysis_run_id is not None for item in calls
                ),
                "degraded_call_count": sum(item.analysis_degraded for item in calls),
                "sections": section_results,
                "language_model_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
