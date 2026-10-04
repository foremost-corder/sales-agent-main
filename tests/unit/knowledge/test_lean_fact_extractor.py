import json

import pytest

from sales_agent.core.config import Settings
from sales_agent.features.knowledge.lean_fact_extractor import OpenAILeanFactExtractor
from sales_agent.features.knowledge.parser import parse_transcript_turns


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://unused:unused@127.0.0.1/unused",
        openai_api_key="test-key",
        chat_model="test-model",
    )


def extraction(call_id: str, candidates: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "schema_version": "conversation-facts-lean-v1",
            "call_id": call_id,
            "candidates": candidates,
        },
        ensure_ascii=False,
    )


def candidate(
    candidate_no: int,
    fact: str,
    quote: str,
    *,
    turn_no: int = 1,
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "candidate_no": candidate_no,
        "phase": "discovery",
        "fact_type": "customer_need",
        "speaker": "customer",
        "fact": fact,
        "explicit": True,
        "confidence": confidence,
        "score_tags": ["需求"],
        "evidence": [{"turn_no": turn_no, "quote": quote}],
    }


def test_lean_extractor_uses_one_call_when_all_evidence_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = parse_transcript_turns("用户0：下周需要二十台电脑")
    extractor = OpenAILeanFactExtractor(make_settings())
    calls: list[str] = []

    def fake_complete(**kwargs):
        calls.append(kwargs["operation"])
        return extraction(
            "call-1",
            [candidate(1, "客户下周需要二十台电脑。", "下周需要二十台电脑")],
        )

    monkeypatch.setattr(extractor, "_complete", fake_complete)

    result = extractor.extract(call_id="call-1", turns=turns)

    assert calls == ["lean fact extraction"]
    assert result.status == "completed"
    assert result.facts[0].validation_status == "verified"
    assert result.facts[0].speaker == "customer"


def test_lean_extractor_only_downgrades_the_fact_with_bad_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = parse_transcript_turns("用户0：需要二十台电脑，预算暂时不确定")
    extractor = OpenAILeanFactExtractor(make_settings(), review_low_confidence=False)
    monkeypatch.setattr(
        extractor,
        "_complete",
        lambda **_: extraction(
            "call-2",
            [
                candidate(1, "客户需要二十台电脑。", "需要二十台电脑"),
                candidate(2, "客户预算十万元。", "预算十万元"),
            ],
        ),
    )

    result = extractor.extract(call_id="call-2", turns=turns)

    assert [item.validation_status for item in result.facts] == [
        "verified",
        "low_confidence",
    ]
    assert result.facts[1].quality_issues == ["evidence_quote_unverified"]


def test_lean_extractor_reviews_only_local_problem_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = parse_transcript_turns(
        "用户0：您好\n用户1：我们的预算暂时不确定\n用户0：那我稍后再联系"
    )
    extractor = OpenAILeanFactExtractor(make_settings())
    requests: list[dict[str, object]] = []

    def fake_complete(**kwargs):
        requests.append(kwargs)
        if kwargs["operation"] == "lean fact extraction":
            return extraction(
                "call-3",
                [candidate(1, "客户预算暂时不确定。", "预算是十万元", turn_no=2)],
            )
        return json.dumps(
            {
                "schema_version": "conversation-facts-lean-review-v1",
                "candidates": [
                    {
                        **candidate(
                            1,
                            "客户预算暂时不确定。",
                            "预算暂时不确定",
                            turn_no=2,
                        ),
                        "supported": True,
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(extractor, "_complete", fake_complete)

    result = extractor.extract(call_id="call-3", turns=turns)

    assert [request["operation"] for request in requests] == [
        "lean fact extraction",
        "lean fact review",
    ]
    review_payload = requests[1]["payload"]
    assert "transcript" not in review_payload
    assert [item["turn_no"] for item in review_payload["nearby_transcript"]] == [1, 2, 3]
    assert result.facts[0].validation_status == "verified"
    assert result.facts[0].evidence[0].quote == "预算暂时不确定"
