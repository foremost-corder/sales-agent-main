import json

import pytest

from sales_agent.core.config import Settings
from sales_agent.features.knowledge.contracts import (
    SpeakerAssignment,
    SpeakerResolutionResult,
    ValidationIssue,
)
from sales_agent.features.knowledge.parser import parse_transcript_turns
from sales_agent.features.knowledge.speaker_resolver import (
    OpenAISpeakerRoleResolver,
    SpeakerResolutionError,
    apply_speaker_resolution,
    validate_speaker_resolution,
)


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql+psycopg://unused:unused@127.0.0.1/unused",
        "openai_api_key": "test-key",
        "chat_model": "test-model",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class SequenceEvaluator:
    version = "fake-evaluator-v1"

    def __init__(self, results: list[list[ValidationIssue]]) -> None:
        self.results = list(results)

    def evaluate(self, **_: object) -> list[ValidationIssue]:
        return self.results.pop(0)


def test_model_resolution_can_reverse_user_zero_and_user_one_roles() -> None:
    turns = parse_transcript_turns(
        "用户0：我们下周需要二十台电脑\n用户1：您好，我是设备租赁顾问"
    )
    result = SpeakerResolutionResult(
        schema_version="speaker-roles-v1",
        assignments=[
            SpeakerAssignment(
                source_speaker_label="用户0",
                role="customer",
                confidence=0.98,
                evidence_turn_nos=[1],
            ),
            SpeakerAssignment(
                source_speaker_label="用户1",
                role="sales",
                confidence=0.99,
                evidence_turn_nos=[2],
            ),
        ],
    )

    validate_speaker_resolution(turns, result)
    resolved = apply_speaker_resolution(turns, result)

    assert [turn.speaker.value for turn in resolved] == ["customer", "sales"]


def test_two_speaker_resolution_requires_one_sales_and_one_customer() -> None:
    turns = parse_transcript_turns("用户0：你好\n用户1：你好")
    result = SpeakerResolutionResult(
        schema_version="speaker-roles-v1",
        assignments=[
            SpeakerAssignment(source_speaker_label="用户0", role="sales", confidence=0.9, evidence_turn_nos=[1]),
            SpeakerAssignment(source_speaker_label="用户1", role="sales", confidence=0.9, evidence_turn_nos=[2]),
        ],
    )

    with pytest.raises(SpeakerResolutionError, match="one sales and one customer"):
        validate_speaker_resolution(turns, result)


def test_resolution_evidence_must_come_from_the_assigned_label() -> None:
    turns = parse_transcript_turns("用户0：你好\n用户1：你好")
    result = SpeakerResolutionResult(
        schema_version="speaker-roles-v1",
        assignments=[
            SpeakerAssignment(source_speaker_label="用户0", role="sales", confidence=0.9, evidence_turn_nos=[2]),
            SpeakerAssignment(source_speaker_label="用户1", role="customer", confidence=0.9, evidence_turn_nos=[1]),
        ],
    )

    with pytest.raises(SpeakerResolutionError, match="same label"):
        validate_speaker_resolution(turns, result)


def test_invalid_provider_fields_are_injected_into_repair_without_alias_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = parse_transcript_turns(
        "用户0：我们下周需要二十台电脑\n用户1：您好，我是设备租赁顾问"
    )
    outputs = [
        '{"call_id":"call-123","role_identifications":[]}',
        json.dumps(
            {
                "schema_version": "speaker-roles-v1",
                "assignments": [
                    {
                        "source_speaker_label": "用户0",
                        "role": "customer",
                        "confidence": 0.98,
                        "evidence_turn_nos": [1],
                    },
                    {
                        "source_speaker_label": "用户1",
                        "role": "sales",
                        "confidence": 0.99,
                        "evidence_turn_nos": [2],
                    },
                ],
            },
            ensure_ascii=False,
        ),
    ]
    requests: list[dict[str, object]] = []
    resolver = OpenAISpeakerRoleResolver(
        make_settings(), evaluator=SequenceEvaluator([[]]), max_repairs=3
    )

    def fake_complete(payload: dict[str, object], *, repairing: bool) -> str:
        requests.append({"payload": payload, "repairing": repairing})
        return outputs.pop(0)

    monkeypatch.setattr(resolver, "_complete", fake_complete)
    result = resolver.resolve(call_id="call-123", turns=turns)

    assert [item.role for item in result.assignments] == ["customer", "sales"]
    repair_context = requests[1]["payload"]["repair_context"]
    error_codes = {item["code"] for item in repair_context["validation_errors"]}
    assert {"extra_forbidden", "missing"} <= error_codes
    assert requests[1]["repairing"] is True


def test_semantic_evaluation_issue_triggers_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = parse_transcript_turns(
        "用户0：我们下周需要二十台电脑\n用户1：您好，我是设备租赁顾问"
    )
    candidate = json.dumps(
        {
            "schema_version": "speaker-roles-v1",
            "assignments": [
                {
                    "source_speaker_label": "用户0",
                    "role": "customer",
                    "confidence": 0.98,
                    "evidence_turn_nos": [1],
                },
                {
                    "source_speaker_label": "用户1",
                    "role": "sales",
                    "confidence": 0.99,
                    "evidence_turn_nos": [2],
                },
            ],
        },
        ensure_ascii=False,
    )
    semantic_issue = ValidationIssue(
        source="semantic",
        code="weak_role_evidence",
        path=["assignments", 0],
        message="角色证据不足",
    )
    evaluator = SequenceEvaluator([[semantic_issue], []])
    resolver = OpenAISpeakerRoleResolver(
        make_settings(), evaluator=evaluator, max_repairs=3
    )
    requests: list[dict[str, object]] = []

    def fake_complete(payload: dict[str, object], *, repairing: bool) -> str:
        requests.append(payload)
        return candidate

    monkeypatch.setattr(resolver, "_complete", fake_complete)
    resolver.resolve(call_id="call-123", turns=turns)

    injected = requests[1]["repair_context"]["validation_errors"]
    assert injected[0]["source"] == "semantic"
    assert injected[0]["code"] == "weak_role_evidence"


def test_speaker_resolution_stops_after_three_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = parse_transcript_turns("用户0：您好\n用户1：您好")
    resolver = OpenAISpeakerRoleResolver(
        make_settings(), evaluator=SequenceEvaluator([]), max_repairs=3
    )
    call_count = 0

    def fake_complete(payload: dict[str, object], *, repairing: bool) -> str:
        nonlocal call_count
        call_count += 1
        return '{"wrong_field":true}'

    monkeypatch.setattr(resolver, "_complete", fake_complete)

    with pytest.raises(SpeakerResolutionError):
        resolver.resolve(call_id="call-123", turns=turns)

    assert call_count == 4
