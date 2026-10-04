from datetime import date
from types import SimpleNamespace

from sales_agent.features.scoring.contracts import (
    CallTarget,
    PeriodScoreRequest,
    ScoreEvidence,
)
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1
from sales_agent.features.scoring.retrieval import (
    PgVectorScoreEvidenceRetriever,
    _select_period_evidence,
)


CALL_ID = "00000000-0000-0000-0000-000000000001"
RUN_ID = "00000000-0000-0000-0000-000000000002"


class FakeScalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class FakeSession:
    def scalars(self, statement):
        return FakeScalars(
            [
                SimpleNamespace(
                    id="00000000-0000-0000-0000-000000000003",
                    call_id=CALL_ID,
                    analysis_run_id=RUN_ID,
                    ordinal=1,
                    content="[1|销售] 我们是做电脑租赁的。\n[2|客户] 好的。",
                    metadata_json={"turn_ids": [1, 2]},
                )
            ]
        )


class FakeEmbedder:
    provider_name = "fake"
    model = "fake-model"
    dimensions = 2

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


def test_low_confidence_fact_only_locates_original_transcript(monkeypatch) -> None:
    def fake_search(session, *, query, **kwargs):
        if not query.include_low_confidence:
            return []
        return [
            {
                "fact_id": "00000000-0000-0000-0000-000000000004",
                "analysis_run_id": RUN_ID,
                "call_id": CALL_ID,
                "fact": "低置信度事实摘要不应直接计分",
                "fact_type": "business_introduction",
                "speaker": "sales",
                "confidence": 0.4,
                "validation_status": "low_confidence",
                "evidence": [{"turn_no": 1, "quote": "我们是做电脑租赁的"}],
                "similarity": 0.9,
            }
        ]

    monkeypatch.setattr(
        "sales_agent.features.scoring.retrieval.search_call_knowledge", fake_search
    )
    retriever = PgVectorScoreEvidenceRetriever(
        FakeSession(), embedder=FakeEmbedder()  # type: ignore[arg-type]
    )
    rule = next(
        item
        for item in SALES_KEY_BEHAVIOR_POLICY_V1.rules
        if item.rule_id == "business_introduction"
    )
    evidence = retriever.retrieve(
        request=PeriodScoreRequest(
            user_id="tenant-a",
            sales_id="001",
            date_from=date(2026, 3, 2),
            date_to=date(2026, 3, 6),
        ),
        call=CallTarget(
            call_id=CALL_ID,
            external_call_id="call-1",
            call_date=date(2026, 3, 2),
            sales_stage="销售线索",
            analysis_run_id=RUN_ID,
        ),
        rule=rule,
    )

    assert len(evidence) == 1
    assert evidence[0].source == "transcript_context"
    assert evidence[0].document_id is not None
    assert evidence[0].trigger_validation_status == "low_confidence"
    assert "电脑租赁" in evidence[0].quote
    assert "低置信度事实摘要" not in evidence[0].quote


def test_period_selection_enforces_final_call_and_low_confidence_limits() -> None:
    def candidate(index: int, call_id: str, source: str, similarity: float):
        return ScoreEvidence(
            evidence_id=f"e-{index}",
            retrieved_for_rule_id="business_introduction",
            source=source,
            call_id=call_id,
            fact="evidence",
            quote="evidence",
            grounding="exact",
            confidence=1.0,
            similarity=similarity,
        )

    selected = _select_period_evidence(
        [
            candidate(1, "call-a", "vector_fact", 0.99),
            candidate(2, "call-a", "vector_fact", 0.98),
            candidate(3, "call-b", "transcript_context", 0.97),
            candidate(4, "call-c", "transcript_context", 0.96),
            candidate(5, "call-d", "vector_fact", 0.95),
        ],
        final_k=3,
        per_call_k=1,
        low_confidence_k=1,
    )

    assert [item.evidence_id for item in selected] == ["e-1", "e-3", "e-5"]
