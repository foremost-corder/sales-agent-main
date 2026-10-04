from datetime import date, datetime, timezone
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import delete

from sales_agent.domain.models import (
    Call,
    CallAnalysisRun,
    CallFact,
    Document,
    DocumentEmbedding,
)
from sales_agent.core.database import SessionLocal
from sales_agent.features.knowledge.read_tools import KnowledgeReadToolExecutor


class FakeEmbeddingProvider:
    provider_name = "fake-search"
    model = "fake-search-512"
    dimensions = 512

    def embed(self, texts):
        assert len(texts) == 1
        return [[1.0] + [0.0] * 511]


def _call(user_id: str, sales_id: str) -> Call:
    return Call(
        user_id=user_id,
        external_call_id=str(uuid4()),
        sales_id=sales_id,
        call_date=date(2026, 9, 5),
        sales_stage="销售线索",
        source_filename="facts.txt",
        source_encoding="utf-8",
        raw_source_text="客户需要二十台电脑",
        transcript_text="客户需要二十台电脑",
        source_hash=sha256(str(uuid4()).encode()).hexdigest(),
        analysis_status="completed",
    )


def _publish_fact(
    session,
    *,
    call: Call,
    fact_text: str,
    fact_type: str,
    vector: list[float],
    knowledge_namespace: str = "baseline-react-v4",
) -> CallAnalysisRun:
    run = CallAnalysisRun(
        call_id=call.id,
        input_fingerprint=sha256(str(uuid4()).encode()).hexdigest(),
        status="completed",
        parser_version="test",
        speaker_resolver_version="test",
        extractor_version="test",
        document_builder_version="test",
        embedding_model="fake-search-512",
        embedding_dimensions=512,
        knowledge_namespace=knowledge_namespace,
        finished_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.flush()
    fact = CallFact(
        call_id=call.id,
        analysis_run_id=run.id,
        fact_key="fact-1",
        phase="discovery",
        fact_type=fact_type,
        speaker="customer",
        fact_text=fact_text,
        explicit=True,
        score_tags=["需求数量"],
        evidence_json=[{"turn_no": 2, "quote": fact_text, "grounding": "exact"}],
    )
    document = Document(
        call_id=call.id,
        analysis_run_id=run.id,
        knowledge_namespace=knowledge_namespace,
        user_id=call.user_id,
        sales_id=call.sales_id,
        call_date=call.call_date,
        doc_type="structured_fact",
        source_key="fact:fact-1",
        ordinal=1,
        content=f"事实：{fact_text}",
        content_hash=sha256(fact_text.encode()).hexdigest(),
        metadata_json={},
        is_active=True,
    )
    session.add_all([fact, document])
    session.flush()
    session.add(
        DocumentEmbedding(
            document_id=document.id,
            embedding=vector,
            embedding_provider="fake-search",
            embedding_model="fake-search-512",
            embedding_dimensions=512,
            embedding_strategy="test",
        )
    )
    return run


@pytest.mark.integration
def test_get_call_facts_filters_and_enforces_user_scope() -> None:
    user_id = f"facts-user-{uuid4()}"
    other_user_id = f"facts-user-{uuid4()}"
    try:
        with SessionLocal.begin() as session:
            call = _call(user_id, "001")
            hidden_call = _call(other_user_id, "002")
            session.add_all([call, hidden_call])
            session.flush()
            run = _publish_fact(
                session,
                call=call,
                fact_text="客户下周需要二十台电脑",
                fact_type="customer_need",
                vector=[1.0] + [0.0] * 511,
            )
            _publish_fact(
                session,
                call=hidden_call,
                fact_text="其他用户的事实",
                fact_type="customer_need",
                vector=[1.0] + [0.0] * 511,
            )

        with SessionLocal() as session:
            tools = KnowledgeReadToolExecutor(
                session, user_id=user_id, embedder=FakeEmbeddingProvider()
            )
            result = tools.execute(
                "get_call_facts",
                {
                    "call_id": str(call.id),
                    "fact_type": "customer_need",
                    "score_tags": ["需求数量"],
                },
            )
            hidden = tools.execute("get_call_facts", {"call_id": str(hidden_call.id)})
            no_match = tools.execute(
                "get_call_facts",
                {"call_id": str(call.id), "fact_type": "customer_budget"},
            )

        assert result.ok is True
        assert result.result is not None
        assert result.result["analysis_run_id"] == str(run.id)
        assert result.result["facts"][0]["fact"] == "客户下周需要二十台电脑"
        assert result.result["facts"][0]["evidence"][0]["turn_no"] == 2
        assert hidden.ok is False
        assert hidden.error is not None
        assert hidden.error.code == "not_found"
        assert no_match.ok is True
        assert no_match.result is not None
        assert no_match.result["status"] == "no_matching_facts"
        assert "search_call_knowledge" in no_match.result["retry_hint"]
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id.in_([user_id, other_user_id])))


@pytest.mark.integration
def test_search_call_knowledge_ranks_facts_and_enforces_user_scope() -> None:
    user_id = f"search-user-{uuid4()}"
    other_user_id = f"search-user-{uuid4()}"
    try:
        with SessionLocal.begin() as session:
            relevant = _call(user_id, "001")
            irrelevant = _call(user_id, "002")
            hidden = _call(other_user_id, "003")
            session.add_all([relevant, irrelevant, hidden])
            session.flush()
            _publish_fact(
                session,
                call=relevant,
                fact_text="客户需要二十台电脑",
                fact_type="customer_need",
                vector=[1.0] + [0.0] * 511,
            )
            _publish_fact(
                session,
                call=irrelevant,
                fact_text="客户询问售后服务",
                fact_type="after_sales_question",
                vector=[0.0, 1.0] + [0.0] * 510,
            )
            _publish_fact(
                session,
                call=hidden,
                fact_text="其他用户也需要电脑",
                fact_type="customer_need",
                vector=[1.0] + [0.0] * 511,
            )

        with SessionLocal() as session:
            result = KnowledgeReadToolExecutor(
                session, user_id=user_id, embedder=FakeEmbeddingProvider()
            ).execute(
                "search_call_knowledge",
                {"query": "客户的电脑数量需求", "min_similarity": 0.5},
            )

        assert result.ok is True
        assert result.result is not None
        assert result.result["returned_count"] == 1
        assert result.result["matches"][0]["call_id"] == str(relevant.id)
        assert result.result["matches"][0]["fact"] == "客户需要二十台电脑"
        assert result.result["matches"][0]["similarity"] == 1.0
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id.in_([user_id, other_user_id])))


@pytest.mark.integration
def test_fact_queries_are_isolated_by_knowledge_namespace() -> None:
    user_id = f"namespace-query-user-{uuid4()}"
    try:
        with SessionLocal.begin() as session:
            call = _call(user_id, "001")
            session.add(call)
            session.flush()
            _publish_fact(
                session,
                call=call,
                fact_text="基线事实",
                fact_type="customer_need",
                vector=[1.0] + [0.0] * 511,
            )
            _publish_fact(
                session,
                call=call,
                fact_text="实验事实",
                fact_type="customer_need",
                vector=[1.0] + [0.0] * 511,
                knowledge_namespace="experiment-lean-v1",
            )

        with SessionLocal() as session:
            tools = KnowledgeReadToolExecutor(
                session, user_id=user_id, embedder=FakeEmbeddingProvider()
            )
            baseline = tools.execute("get_call_facts", {"call_id": str(call.id)})
            experiment = tools.execute(
                "get_call_facts",
                {
                    "call_id": str(call.id),
                    "knowledge_namespace": "experiment-lean-v1",
                },
            )

        assert baseline.result is not None
        assert experiment.result is not None
        assert [item["fact"] for item in baseline.result["facts"]] == ["基线事实"]
        assert [item["fact"] for item in experiment.result["facts"]] == ["实验事实"]
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id == user_id))


def test_knowledge_tools_reject_unbounded_or_invalid_arguments() -> None:
    tools = KnowledgeReadToolExecutor(
        session=None,  # type: ignore[arg-type]
        user_id="user-1",
        embedder=FakeEmbeddingProvider(),
    )

    invalid_facts = tools.execute("get_call_facts", {"call_id": str(uuid4()), "limit": 101})
    invalid_search = tools.execute(
        "search_call_knowledge",
        {"query": "需求", "date_from": "2026-09-06", "date_to": "2026-09-05"},
    )
    invented_type = tools.execute(
        "get_call_facts",
        {"call_id": str(uuid4()), "fact_type": "联系方式验证"},
    )

    assert invalid_facts.ok is False
    assert invalid_facts.error is not None
    assert invalid_facts.error.code == "invalid_arguments"
    assert invalid_search.ok is False
    assert invalid_search.error is not None
    assert invalid_search.error.code == "invalid_arguments"
    assert invented_type.ok is False
    assert invented_type.error is not None
    assert invented_type.error.code == "invalid_arguments"
