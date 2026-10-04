import json
from types import SimpleNamespace

import pytest

from sales_agent.core.config import Settings
from sales_agent.features.knowledge.contracts import SpeakerRole
from sales_agent.features.knowledge.fact_extractor import OpenAIFactExtractor, _anchor_quote_to_source
import sales_agent.features.knowledge.fact_extractor as fact_extractor_module
from sales_agent.features.knowledge.parser import parse_transcript_turns


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql+psycopg://unused:unused@127.0.0.1/unused",
        "openai_api_key": "test-key",
        "chat_model": "test-model",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def customer_turns(text: str):
    return [
        parse_transcript_turns(f"用户0：{text}")[0].model_copy(
            update={"speaker": SpeakerRole.CUSTOMER}
        )
    ]


def recall(call_id: str, facts: list[dict[str, object]]) -> str:
    return json.dumps(
        {"schema_version": "fact-recall-v1", "call_id": call_id, "candidates": facts},
        ensure_ascii=False,
    )


def annotations(items: list[dict[str, object]]) -> str:
    return json.dumps(
        {"schema_version": "fact-annotation-v1", "annotations": items},
        ensure_ascii=False,
    )


def coverage(missing: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "schema_version": "fact-coverage-v1",
            "passed": not missing,
            "missing_facts": missing,
        },
        ensure_ascii=False,
    )


def test_recall_schema_errors_are_injected_into_same_action_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = customer_turns("下周需要二十台电脑")
    outputs = {
        "recall": [
            '{"conversation_facts":[]}',
            recall(
                "call-123",
                [
                    {
                        "candidate_id": "candidate-1",
                        "fact": "客户下周需要二十台电脑。",
                        "explicit": True,
                        "evidence": [{"turn_no": 1, "quote": "下周需要二十台电脑"}],
                    }
                ],
            ),
        ],
        "annotate": [
            annotations(
                [
                    {
                        "candidate_id": "candidate-1",
                        "phase": "discovery",
                        "fact_type": "customer_need",
                        "score_tags": ["需求数量"],
                    }
                ]
            )
        ],
        "coverage_check": [coverage([])],
    }
    requests: list[dict[str, object]] = []
    extractor = OpenAIFactExtractor(make_settings(), max_repairs=3)

    def fake_complete(payload, *, action, repairing, **_):
        requests.append({"payload": payload, "action": action, "repairing": repairing})
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert result.facts[0].speaker == "customer"
    assert result.facts[0].fact_type == "customer_need"
    recall_repair = requests[1]
    assert recall_repair["action"] == "recall"
    assert recall_repair["repairing"] is True
    error_codes = {
        item["code"]
        for item in recall_repair["payload"]["repair_context"]["validation_errors"]
    }
    assert {"extra_forbidden", "missing"} <= error_codes


def test_coverage_missing_fact_is_added_and_reannotated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = customer_turns("下周需要二十台电脑，预算十万元")
    first = {
        "candidate_id": "candidate-1",
        "fact": "客户下周需要二十台电脑。",
        "explicit": True,
        "evidence": [{"turn_no": 1, "quote": "下周需要二十台电脑"}],
    }
    budget = {
        "fact": "客户预算十万元。",
        "explicit": True,
        "evidence": [{"turn_no": 1, "quote": "预算十万元"}],
    }
    outputs = {
        "recall": [recall("call-123", [first])],
        "annotate": [
            annotations(
                [
                    {
                        "candidate_id": "candidate-1",
                        "phase": "discovery",
                        "fact_type": "customer_need",
                        "score_tags": ["需求"],
                    }
                ]
            ),
            annotations(
                [
                    {
                        "candidate_id": "candidate-1",
                        "phase": "discovery",
                        "fact_type": "customer_need",
                        "score_tags": ["需求"],
                    },
                    {
                        "candidate_id": "candidate-2",
                        "phase": "unknown",
                        "fact_type": "customer_budget",
                        "score_tags": [],
                    },
                ]
            ),
        ],
        "coverage_check": [coverage([budget]), coverage([])],
    }
    actions: list[str] = []
    extractor = OpenAIFactExtractor(make_settings(), max_repairs=3)

    def fake_complete(payload, *, action, **_):
        actions.append(action)
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert actions == ["recall", "annotate", "coverage_check", "annotate", "coverage_check"]
    assert [item.fact_type for item in result.facts] == ["customer_need", "customer_budget"]
    assert result.facts[1].phase == "unknown"
    assert result.facts[1].score_tags == []


def test_bad_coverage_quote_uses_turn_level_grounding_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = customer_turns("预算十万元")
    outputs = {
        "recall": [recall("call-123", [])],
        "annotate": [annotations([]), annotations([
            {
                "candidate_id": "candidate-1",
                "phase": "discovery",
                "fact_type": "customer_budget",
                "score_tags": ["预算"],
            }
        ])],
        "coverage_check": [
            coverage([
                {
                    "fact": "客户预算十万元。",
                    "explicit": True,
                    "evidence": [{"turn_no": 1, "quote": "预算十万块"}],
                }
            ]),
            coverage([]),
        ],
    }
    requests: list[tuple[str, dict[str, object], bool]] = []
    extractor = OpenAIFactExtractor(make_settings(), max_repairs=3)

    def fake_complete(payload, *, action, repairing, **_):
        requests.append((action, payload, repairing))
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert not any(item[0] == "coverage_check" and item[2] for item in requests)
    assert len(result.facts) == 1
    assert result.facts[0].evidence[0].quote == "预算十万元"
    assert result.facts[0].evidence[0].grounding == "turn_only"
    assert result.facts[0].validation_status == "low_confidence"
    assert result.facts[0].confidence == 0.45


def test_coverage_rework_limit_returns_last_hard_valid_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = customer_turns("需要电脑，预算十万元")
    need = {
        "fact": "客户需要电脑。",
        "explicit": True,
        "evidence": [{"turn_no": 1, "quote": "需要电脑"}],
    }
    budget = {
        "fact": "客户预算十万元。",
        "explicit": True,
        "evidence": [{"turn_no": 1, "quote": "预算十万元"}],
    }
    outputs = {
        "recall": [recall("call-123", [])],
        "annotate": [
            annotations([]),
            annotations(
                [
                    {
                        "candidate_id": "candidate-1",
                        "phase": "discovery",
                        "fact_type": "customer_need",
                        "score_tags": ["需求"],
                    }
                ]
            ),
        ],
        "coverage_check": [coverage([need]), coverage([budget])],
    }
    actions: list[str] = []
    extractor = OpenAIFactExtractor(make_settings(), max_repairs=1)

    def fake_complete(payload, *, action, **_):
        actions.append(action)
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert actions == ["recall", "annotate", "coverage_check", "annotate", "coverage_check"]
    assert [item.fact for item in result.facts] == ["客户需要电脑。"]


def test_provider_completion_limit_is_detected_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_limits: list[int] = []
    response_formats: list[dict[str, object]] = []

    class ProviderLimitError(Exception):
        status_code = 400
        request_id = "request-1"

    class FakeCompletions:
        def create(self, **options):
            requested_limits.append(options["max_completion_tokens"])
            response_formats.append(options["response_format"])
            if len(requested_limits) == 1:
                raise ProviderLimitError(
                    "max_tokens is too large: 4096. This model supports at most "
                    "2048 completion tokens"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(fact_extractor_module, "OpenAI", lambda **_: fake_client)
    extractor = OpenAIFactExtractor(make_settings())

    content = extractor._complete(
        {}, action="recall", repairing=False, max_output_tokens=4096
    )

    assert content == '{"ok":true}'
    assert requested_limits == [4096, 2048]
    assert all(item["type"] == "json_schema" for item in response_formats)
    assert all(item["json_schema"]["strict"] is True for item in response_formats)


def test_insertion_only_near_quote_is_anchored_to_exact_source() -> None:
    source = "就是呢电脑您先这个先下单免押金租赁嘛"

    anchored = _anchor_quote_to_source("电脑您先下单免押金租赁嘛", source)

    assert anchored == "电脑您先这个先下单免押金租赁嘛"


def test_quote_with_changed_business_value_is_not_auto_aligned() -> None:
    assert _anchor_quote_to_source("客户预算二十万元", "客户预算三十万元") is None


def test_recall_uses_programmatically_anchored_quote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_text = "就是呢电脑您先这个先下单免押金租赁嘛"
    turns = customer_turns(source_text)
    outputs = {
        "recall": [
            recall(
                "call-123",
                [
                    {
                        "candidate_id": "candidate-1",
                        "fact": "客户可以先下单进行免押金租赁。",
                        "explicit": True,
                        "evidence": [
                            {
                                "turn_no": 1,
                                "quote": "电脑您先下单免押金租赁嘛",
                            }
                        ],
                    }
                ],
            )
        ],
        "annotate": [
            annotations(
                [
                    {
                        "candidate_id": "candidate-1",
                        "phase": "proposal_negotiation",
                        "fact_type": "product_recommendation",
                        "score_tags": ["免押金租赁"],
                    }
                ]
            )
        ],
        "coverage_check": [coverage([])],
    }
    extractor = OpenAIFactExtractor(make_settings())

    def fake_complete(payload, *, action, **_):
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert result.facts[0].evidence[0].quote == "电脑您先这个先下单免押金租赁嘛"
    assert result.facts[0].evidence[0].grounding == "aligned"


def test_unmatched_quote_falls_back_to_full_turn_without_losing_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = customer_turns("需要电脑，预算三十万元")
    outputs = {
        "recall": [
            recall(
                "call-123",
                [
                    {
                        "candidate_id": "candidate-1",
                        "fact": "客户需要电脑。",
                        "explicit": True,
                        "evidence": [{"turn_no": 1, "quote": "需要电脑"}],
                    },
                    {
                        "candidate_id": "candidate-2",
                        "fact": "客户预算二十万元。",
                        "explicit": True,
                        "evidence": [{"turn_no": 1, "quote": "预算二十万元"}],
                    },
                ],
            )
        ],
        "annotate": [
            annotations(
                [
                    {
                        "candidate_id": "candidate-1",
                        "phase": "discovery",
                        "fact_type": "customer_need",
                        "score_tags": ["需求"],
                    },
                    {
                        "candidate_id": "candidate-2",
                        "phase": "discovery",
                        "fact_type": "customer_budget",
                        "score_tags": ["预算"],
                    },
                ]
            )
        ],
        "coverage_check": [coverage([])],
    }
    extractor = OpenAIFactExtractor(make_settings(), max_repairs=0)

    def fake_complete(payload, *, action, **_):
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert [item.fact for item in result.facts] == ["客户需要电脑。", "客户预算二十万元。"]
    assert result.facts[1].evidence[0].quote == "需要电脑，预算三十万元"
    assert result.facts[1].evidence[0].grounding == "turn_only"
    assert result.facts[1].validation_status == "low_confidence"
    assert "evidence_quote_unverified" in result.facts[1].quality_issues


def test_unresolved_speaker_is_persistable_as_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = parse_transcript_turns("用户0：我们暂时不需要电脑")
    outputs = {
        "recall": [
            recall(
                "call-123",
                [
                    {
                        "candidate_id": "candidate-1",
                        "fact": "对方暂时不需要电脑。",
                        "explicit": True,
                        "evidence": [{"turn_no": 1, "quote": "暂时不需要电脑"}],
                    }
                ],
            )
        ],
        "annotate": [
            annotations(
                [
                    {
                        "candidate_id": "candidate-1",
                        "phase": "discovery",
                        "fact_type": "customer_current_state",
                        "score_tags": [],
                    }
                ]
            )
        ],
        "coverage_check": [coverage([])],
    }
    extractor = OpenAIFactExtractor(make_settings())

    def fake_complete(payload, *, action, **_):
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert result.status == "low_confidence"
    assert result.facts[0].speaker == "unknown"
    assert result.facts[0].confidence == 0.35
    assert result.facts[0].quality_issues == ["speaker_unresolved"]


def test_annotation_exhaustion_keeps_grounded_fact_as_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = customer_turns("下周需要二十台电脑")
    outputs = {
        "recall": [
            recall(
                "call-123",
                [
                    {
                        "candidate_id": "candidate-1",
                        "fact": "客户下周需要二十台电脑。",
                        "explicit": True,
                        "evidence": [{"turn_no": 1, "quote": "下周需要二十台电脑"}],
                    }
                ],
            )
        ],
        "annotate": ['{"invalid":true}'],
        "coverage_check": [coverage([])],
    }
    extractor = OpenAIFactExtractor(make_settings(), max_repairs=0)

    def fake_complete(payload, *, action, **_):
        return outputs[action].pop(0)

    monkeypatch.setattr(extractor, "_complete", fake_complete)
    result = extractor.extract(call_id="call-123", turns=turns)

    assert len(result.facts) == 1
    assert result.facts[0].phase == "unknown"
    assert result.facts[0].fact_type == "other_business_fact"
    assert result.facts[0].confidence == 0.4
    assert result.facts[0].validation_status == "low_confidence"
    assert "annotation_validation_failed" in result.facts[0].quality_issues
