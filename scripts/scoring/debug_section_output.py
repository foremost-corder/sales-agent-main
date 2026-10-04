"""Temporarily capture one scoring section's raw model output and validation errors."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

from sales_agent.core.config import get_settings
from sales_agent.core.database import SessionLocal
from sales_agent.features.knowledge.namespaces import BASELINE_KNOWLEDGE_NAMESPACE
from sales_agent.features.scoring import model_agents
from sales_agent.features.scoring.contracts import PeriodScoreRequest
from sales_agent.features.scoring.model_agents import (
    OpenAISectionScoringAgent,
    ScoringAgentError,
    _SectionOutput,
)
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1
from sales_agent.features.scoring.retrieval import PgVectorScoreEvidenceRetriever
from sales_agent.integrations.embeddings.factory import build_embedding_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="local-demo-user")
    parser.add_argument("--sales-id", default="001")
    parser.add_argument("--date-from", type=date.fromisoformat, default=date(2026, 3, 2))
    parser.add_argument("--date-to", type=date.fromisoformat, default=date(2026, 3, 6))
    parser.add_argument("--namespace", default=BASELINE_KNOWLEDGE_NAMESPACE)
    parser.add_argument("--section", default="business_introduction")
    parser.add_argument(
        "--analyze-payload-only",
        action="store_true",
        help="Measure request fields without sending anything to the model provider.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("artifacts/scoring/debug_business_introduction.log"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    section = next(
        item
        for item in SALES_KEY_BEHAVIOR_POLICY_V1.sections
        if item.section_id == args.section
    )
    rules_by_id = {item.rule_id: item for item in SALES_KEY_BEHAVIOR_POLICY_V1.rules}
    rules = [rules_by_id[rule_id] for rule_id in section.rule_ids]
    request = PeriodScoreRequest(
        user_id=args.user_id,
        sales_id=args.sales_id,
        date_from=args.date_from,
        date_to=args.date_to,
        knowledge_namespace=args.namespace,
    )

    original_complete = model_agents._complete

    def capture_complete(settings_arg, instruction, payload, operation):
        raw = original_complete(settings_arg, instruction, payload, operation)
        record: dict[str, object] = {
            "captured_at": datetime.now().astimezone().isoformat(),
            "operation": operation,
            "model": settings_arg.scoring_model_name,
            "endpoint": settings_arg.openai_base_url or "https://api.openai.com/v1",
            "content_characters": len(raw),
            # Keep this diagnostic bounded. It may contain model-repeated evidence text.
            "content_preview": raw[:12000],
            "content_truncated": len(raw) > 12000,
        }
        try:
            _SectionOutput.model_validate_json(raw)
            record["schema_valid"] = True
            record["validation_errors"] = []
        except ValidationError as exc:
            record["schema_valid"] = False
            record["validation_errors"] = exc.errors(
                include_url=False,
                include_input=False,
            )
        args.log_file.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return raw

    model_agents._complete = capture_complete
    try:
        with SessionLocal() as session:
            retriever = PgVectorScoreEvidenceRetriever(
                session,
                embedder=build_embedding_provider(settings),
            )
            calls = retriever.list_calls(request)
            evidence = []
            for rule in rules:
                evidence.extend(
                    retriever.retrieve_period(
                        request=request,
                        calls=calls,
                        rule=rule,
                    )
                )
            relevant_call_ids = {item.call_id for item in evidence}
            relevant_calls = [
                call for call in calls if call.call_id in relevant_call_ids
            ]
            logging.info(
                "debug_section section=%s period_calls=%s relevant_calls=%s evidence=%s model=%s endpoint=%s",
                section.section_id,
                len(calls),
                len(relevant_calls),
                len(evidence),
                settings.scoring_model_name,
                settings.openai_base_url or "https://api.openai.com/v1",
            )
            payload_parts = {
                "section": {
                    "section_id": section.section_id,
                    "name": section.name,
                    "purpose": section.purpose,
                },
                "rules": [model_agents._score_rule_payload(item) for item in rules],
                "calls": [{"call_id": item.call_id} for item in relevant_calls],
                "candidate_evidence": [
                    model_agents._score_evidence_payload(item) for item in evidence
                ],
                "review_issues": [],
            }
            payload_sizes = {
                key: len(json.dumps(value, ensure_ascii=False))
                for key, value in payload_parts.items()
            }
            payload_sizes["total_serialized"] = len(
                json.dumps(payload_parts, ensure_ascii=False)
            )
            call_field_sizes = {
                field: sum(
                    len(json.dumps(item.model_dump(mode="json")[field], ensure_ascii=False))
                    for item in relevant_calls
                )
                for field in type(relevant_calls[0]).model_fields
            } if relevant_calls else {}
            evidence_field_sizes = {
                field: sum(
                    len(json.dumps(item.model_dump(mode="json")[field], ensure_ascii=False))
                    for item in evidence
                )
                for field in type(evidence[0]).model_fields
            } if evidence else {}
            logging.info("payload_character_sizes=%s", json.dumps(payload_sizes))
            if args.analyze_payload_only:
                args.log_file.write_text(
                    json.dumps(
                        {
                            "section": section.section_id,
                            "period_call_count": len(calls),
                            "relevant_call_count": len(relevant_calls),
                            "evidence_count": len(evidence),
                            "payload_character_sizes": payload_sizes,
                            "call_value_character_sizes": call_field_sizes,
                            "evidence_value_character_sizes": evidence_field_sizes,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logging.info("payload analysis log=%s", args.log_file.resolve())
                return 0
            OpenAISectionScoringAgent(settings).score_section(
                section=section,
                rules=rules,
                calls=relevant_calls,
                evidence=evidence,
            )
    except ScoringAgentError as exc:
        logging.error("captured_failure error_type=%s message=%s", type(exc).__name__, exc)
        logging.info("diagnostic_log=%s", args.log_file.resolve())
        return 1
    finally:
        model_agents._complete = original_complete

    logging.info("section output was valid; diagnostic_log=%s", args.log_file.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
