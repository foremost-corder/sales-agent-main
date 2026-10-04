"""Read-only persistence queries for extracted and embedded call facts."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sales_agent.domain.models import (
    Call,
    CallAnalysisRun,
    CallFact,
    Document,
    DocumentEmbedding,
)
from sales_agent.features.knowledge.query import GetCallFactsInput, SearchCallKnowledgeInput


@dataclass(frozen=True)
class FactPage:
    call_found: bool
    analysis_run_id: UUID | None
    rows: list[dict[str, Any]]
    next_offset: int | None


def get_call_facts(
    session: Session,
    *,
    user_id: str,
    query: GetCallFactsInput,
) -> FactPage:
    """Read one authorized call's facts without exposing arbitrary SQL."""
    call_exists = session.scalar(
        select(Call.id).where(Call.id == query.call_id, Call.user_id == user_id)
    )
    if call_exists is None:
        return FactPage(False, None, [], None)

    run_statement = select(CallAnalysisRun.id).where(
        CallAnalysisRun.call_id == query.call_id,
        CallAnalysisRun.status == "completed",
        CallAnalysisRun.knowledge_namespace == query.knowledge_namespace,
    )
    if query.analysis_run_id is not None:
        run_statement = run_statement.where(CallAnalysisRun.id == query.analysis_run_id)
    else:
        run_statement = run_statement.order_by(
            CallAnalysisRun.finished_at.desc().nullslast(),
            CallAnalysisRun.started_at.desc(),
        )
    run_id = session.scalar(run_statement.limit(1))
    if run_id is None:
        return FactPage(True, None, [], None)

    statement = select(CallFact).where(CallFact.analysis_run_id == run_id)
    statement = _apply_fact_filters(statement, query)
    facts = session.scalars(
        statement.order_by(CallFact.fact_key).offset(query.offset).limit(query.limit + 1)
    ).all()
    has_more = len(facts) > query.limit
    rows = [_serialize_fact(fact) for fact in facts[: query.limit]]
    return FactPage(
        True,
        run_id,
        rows,
        query.offset + query.limit if has_more else None,
    )


def search_call_knowledge(
    session: Session,
    *,
    user_id: str,
    query: SearchCallKnowledgeInput,
    query_vector: list[float],
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[dict[str, Any]]:
    """Run user-scoped cosine search over active structured-fact documents."""
    distance = DocumentEmbedding.embedding.cosine_distance(query_vector)
    similarity = (1.0 - distance).label("similarity")
    statement = (
        select(Document, CallFact, similarity)
        .join(DocumentEmbedding, DocumentEmbedding.document_id == Document.id)
        .join(
            CallFact,
            (CallFact.analysis_run_id == Document.analysis_run_id)
            & (Document.source_key == func.concat("fact:", CallFact.fact_key)),
        )
        .where(
            Document.user_id == user_id,
            Document.knowledge_namespace == query.knowledge_namespace,
            Document.doc_type == "structured_fact",
            Document.is_active.is_(True),
            DocumentEmbedding.embedding_provider == embedding_provider,
            DocumentEmbedding.embedding_model == embedding_model,
            DocumentEmbedding.embedding_dimensions == embedding_dimensions,
            distance <= 1.0 - query.min_similarity,
        )
    )
    if query.call_id is not None:
        statement = statement.where(Document.call_id == query.call_id)
    if query.sales_id is not None:
        statement = statement.where(Document.sales_id == query.sales_id)
    if query.date_from is not None:
        statement = statement.where(Document.call_date >= query.date_from)
    if query.date_to is not None:
        statement = statement.where(Document.call_date <= query.date_to)
    statement = _apply_fact_filters(statement, query)

    rows = session.execute(statement.order_by(distance).limit(query.limit)).all()
    return [
        {
            **_serialize_fact(fact),
            "call_id": str(document.call_id),
            "analysis_run_id": str(document.analysis_run_id),
            "knowledge_namespace": document.knowledge_namespace,
            "sales_id": document.sales_id,
            "call_date": document.call_date.isoformat(),
            "similarity": round(float(score), 6),
        }
        for document, fact, score in rows
    ]


def _apply_fact_filters(statement, query):
    validation_status = getattr(query, "validation_status", None)
    if validation_status is not None:
        statement = statement.where(
            CallFact.validation_status == validation_status
        )
    elif not query.include_low_confidence:
        statement = statement.where(CallFact.validation_status == "verified")
    if query.phase is not None:
        statement = statement.where(CallFact.phase == query.phase)
    if query.fact_type is not None:
        statement = statement.where(CallFact.fact_type == query.fact_type)
    if query.speaker is not None:
        statement = statement.where(CallFact.speaker == query.speaker)
    if query.score_tags:
        statement = statement.where(CallFact.score_tags.contains(query.score_tags))
    return statement


def _serialize_fact(fact: CallFact) -> dict[str, Any]:
    return {
        "fact_id": str(fact.id),
        "fact_key": fact.fact_key,
        "phase": fact.phase,
        "fact_type": fact.fact_type,
        "speaker": fact.speaker,
        "fact": fact.fact_text,
        "explicit": fact.explicit,
        "confidence": fact.confidence,
        "validation_status": fact.validation_status,
        "quality_issues": fact.quality_issues,
        "score_tags": fact.score_tags,
        "evidence": fact.evidence_json,
    }
