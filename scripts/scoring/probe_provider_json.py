"""Probe an OpenAI-compatible provider's JSON and Responses API support."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from sales_agent.core.config import get_settings
from sales_agent.agent.structured_json import strict_model_json_schema
from sales_agent.features.scoring.contracts import ReviewReport
from sales_agent.features.scoring.model_agents import _SectionOutput


SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "const": "provider-probe-v1"},
        "value": {"type": "integer"},
        "status": {"type": "string", "enum": ["ok"]},
    },
    "required": ["schema_version", "value", "status"],
    "additionalProperties": False,
}
PROMPT = (
    'Return provider-probe-v1 JSON with value equal to 7 and status equal to "ok".'
)
OUTPUT = Path("artifacts/scoring/provider_json_probe.json")


def _validate(text: str | None) -> dict[str, Any]:
    if not text:
        return {"valid": False, "reason": "empty output"}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "reason": f"invalid JSON: {exc.msg}",
            "preview": text[:300],
        }
    return {
        "valid": value
        == {"schema_version": "provider-probe-v1", "value": 7, "status": "ok"},
        "parsed": value,
    }


def _validate_section(text: str | None) -> dict[str, Any]:
    if not text:
        return {"valid": False, "reason": "empty output"}
    try:
        parsed = _SectionOutput.model_validate_json(text)
    except Exception as exc:
        return {
            "valid": False,
            "reason": str(exc)[:1000],
            "preview": text[:300],
        }
    return {"valid": True, "parsed": parsed.model_dump(mode="json")}


def _validate_review(text: str | None) -> dict[str, Any]:
    if not text:
        return {"valid": False, "reason": "empty output"}
    try:
        parsed = ReviewReport.model_validate_json(text)
    except Exception as exc:
        return {
            "valid": False,
            "reason": str(exc)[:1000],
            "preview": text[:300],
        }
    return {"valid": True, "parsed": parsed.model_dump(mode="json")}


def _run(
    name: str,
    request: Callable[[], str | None],
    validator: Callable[[str | None], dict[str, Any]] = _validate,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        validation = validator(request())
        return {
            "name": name,
            "request_succeeded": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            **validation,
        }
    except Exception as exc:
        body = getattr(exc, "body", None)
        return {
            "name": name,
            "request_succeeded": False,
            "valid": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "error_type": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
            "error": str(exc)[:1000],
            "provider_body": body if isinstance(body, dict) else None,
        }


def _chat_review_probe(client: OpenAI, model: str) -> dict[str, Any]:
    return _run(
        "chat_production_review_schema",
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return a completed reasonable review with no issues and "
                        "a short summary."
                    ),
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "score_review",
                    "strict": True,
                    "schema": strict_model_json_schema(ReviewReport),
                },
            },
            max_completion_tokens=1024,
        ).choices[0].message.content,
        _validate_review,
    )
def main() -> int:
    settings = get_settings()
    assert settings.openai_api_key is not None
    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout=60,
        max_retries=0,
    )

    if "--review-only" in sys.argv:
        results = [_chat_review_probe(client, settings.scoring_model_name)]
        payload = {
            "endpoint": settings.openai_base_url,
            "model": settings.scoring_model_name,
            "tests": results,
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if all(item["valid"] for item in results) else 1

    results = [
        _run(
            "chat_json_schema",
            lambda: client.chat.completions.create(
                model=settings.scoring_model_name,
                messages=[{"role": "user", "content": PROMPT}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "provider_probe",
                        "strict": True,
                        "schema": SCHEMA,
                    },
                },
                max_completion_tokens=1024,
            ).choices[0].message.content,
        ),
        _run(
            "responses_json_schema",
            lambda: client.responses.create(
                model=settings.scoring_model_name,
                input=PROMPT,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "provider_probe",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
                max_output_tokens=1024,
                store=False,
            ).output_text,
        ),
        _run(
            "chat_production_scoring_schema",
            lambda: client.chat.completions.create(
                model=settings.scoring_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Return a score-section-v1 object with an empty judgments array."
                        ),
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "score_section",
                        "strict": True,
                        "schema": strict_model_json_schema(_SectionOutput),
                    },
                },
                max_completion_tokens=1024,
            ).choices[0].message.content,
            _validate_section,
        ),
        _run(
            "responses_production_scoring_schema",
            lambda: client.responses.create(
                model=settings.scoring_model_name,
                input="Return a score-section-v1 object with an empty judgments array.",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "score_section",
                        "strict": True,
                        "schema": strict_model_json_schema(_SectionOutput),
                    }
                },
                max_output_tokens=1024,
                store=False,
            ).output_text,
            _validate_section,
        ),
        _chat_review_probe(client, settings.scoring_model_name),
    ]
    payload = {
        "endpoint": settings.openai_base_url,
        "model": settings.scoring_model_name,
        "tests": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item["valid"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
