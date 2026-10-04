"""Metadata-first evidence retrieval from the existing call knowledge store."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from sales_agent.domain.models import Call, CallAnalysisRun, CallTurn, Document
from sales_agent.features.knowledge.contracts import EmbeddingProvider
from sales_agent.features.knowledge.repository import search_call_knowledge
from sales_agent.features.scoring.contracts import (
    CallTarget,
    PeriodScoreRequest,
    ScoreEvidence,
    ScoreRule,
)
from sales_agent.features.knowledge.query import SearchCallKnowledgeInput


class PgVectorScoreEvidenceRetriever:
    """Filter calls by tenant/sales/date before doing rule-specific vector search."""

    def __init__(self, session: Session, *, embedder: EmbeddingProvider) -> None:
        self._session = session
        self._embedder = embedder
        self._transcript_chunks_by_run: dict[str, list[Document]] = {}
        self._query_vectors: dict[str, list[float]] = {}

    def list_calls(self, request: PeriodScoreRequest) -> list[CallTarget]:
        calls = self._session.scalars(
            select(Call)
            .where(
                Call.user_id == request.user_id,
                Call.sales_id == request.sales_id,
                Call.call_date >= request.date_from,
                Call.call_date <= request.date_to,
            )
            .order_by(Call.call_date, Call.external_call_id, Call.id)
        ).all()
        result: list[CallTarget] = []
        for call in calls:
            run = self._session.scalar(
                select(CallAnalysisRun)
                .where(
                    CallAnalysisRun.call_id == call.id,
                    CallAnalysisRun.knowledge_namespace == request.knowledge_namespace,
                    CallAnalysisRun.status == "completed",
                )
                .order_by(
                    CallAnalysisRun.finished_at.desc().nullslast(),
                    CallAnalysisRun.started_at.desc(),
                )
                .limit(1)
            )
            result.append(
                CallTarget(
                    call_id=str(call.id),
                    external_call_id=call.external_call_id,
                    call_date=call.call_date,
                    sales_stage=call.sales_stage,
                    analysis_run_id=str(run.id) if run else None,
                    analysis_degraded=bool(run.degraded) if run else False,
                )
            )
        return result

    def retrieve_period(
        self,
        *,
        request: PeriodScoreRequest,
        calls: list[CallTarget],
        rule: ScoreRule,
        queries: list[str] | None = None,
    ) -> list[ScoreEvidence]:
        if rule.scoring_mode == "audio_required":
            return []
        if rule.scoring_mode == "interaction_metric":
            return [
                metric
                for call in calls
                if (metric := self._interaction_metric(call)) is not None
            ]

        selected_runs = {
            call.call_id: call.analysis_run_id
            for call in calls
            if call.analysis_run_id is not None
        }
        if not selected_runs:
            return []

        matches_by_fact: dict[str, dict] = {}
        low_confidence_matches: dict[str, dict] = {}
        trace_low_confidence = (
            request.trace_low_confidence_to_transcript
            or request.include_low_confidence
        ) and rule.retrieval_low_confidence_k > 0
        min_similarity = max(
            request.min_similarity, rule.retrieval_min_similarity
        )
        for query_text in queries or rule.retrieval_queries:
            vector = self._query_vectors.get(query_text)
            if vector is None:
                vector = self._embedder.embed([query_text])[0]
                self._query_vectors[query_text] = vector
            verified_matches = search_call_knowledge(
                self._session,
                user_id=request.user_id,
                query=SearchCallKnowledgeInput(
                    query=query_text,
                    knowledge_namespace=request.knowledge_namespace,
                    sales_id=request.sales_id,
                    date_from=request.date_from,
                    date_to=request.date_to,
                    include_low_confidence=False,
                    validation_status="verified",
                    min_similarity=min_similarity,
                    limit=rule.retrieval_per_query_k,
                ),
                query_vector=vector,
                embedding_provider=self._embedder.provider_name,
                embedding_model=self._embedder.model,
                embedding_dimensions=self._embedder.dimensions,
            )
            for match in verified_matches:
                if selected_runs.get(str(match["call_id"])) != str(
                    match["analysis_run_id"]
                ):
                    continue
                fact_id = str(match["fact_id"])
                previous = matches_by_fact.get(fact_id)
                if previous is None or match["similarity"] > previous["similarity"]:
                    matches_by_fact[fact_id] = match

            if not trace_low_confidence:
                continue
            low_matches = search_call_knowledge(
                self._session,
                user_id=request.user_id,
                query=SearchCallKnowledgeInput(
                    query=query_text,
                    knowledge_namespace=request.knowledge_namespace,
                    sales_id=request.sales_id,
                    date_from=request.date_from,
                    date_to=request.date_to,
                    include_low_confidence=True,
                    validation_status="low_confidence",
                    min_similarity=min_similarity,
                    limit=rule.retrieval_per_query_k,
                ),
                query_vector=vector,
                embedding_provider=self._embedder.provider_name,
                embedding_model=self._embedder.model,
                embedding_dimensions=self._embedder.dimensions,
            )
            for match in low_matches:
                if selected_runs.get(str(match["call_id"])) != str(
                    match["analysis_run_id"]
                ):
                    continue
                fact_id = str(match["fact_id"])
                previous = low_confidence_matches.get(fact_id)
                if previous is None or match["similarity"] > previous["similarity"]:
                    low_confidence_matches[fact_id] = match

        ordered = sorted(
            matches_by_fact.values(), key=lambda item: item["similarity"], reverse=True
        )[: request.evidence_limit_per_rule]
        evidence: list[ScoreEvidence] = []
        for match in ordered:
            anchors = match.get("evidence") or []
            anchor = anchors[0] if anchors else {}
            evidence.append(
                ScoreEvidence(
                    evidence_id=f"{rule.rule_id}:{match['fact_id']}",
                    retrieved_for_rule_id=rule.rule_id,
                    source="vector_fact",
                    call_id=str(match["call_id"]),
                    analysis_run_id=str(match["analysis_run_id"]),
                    fact_id=str(match["fact_id"]),
                    fact_type=match.get("fact_type"),
                    speaker=match.get("speaker"),
                    fact=match["fact"],
                    turn_no=anchor.get("turn_no"),
                    quote=anchor.get("quote") or match["fact"],
                    grounding=anchor.get("grounding", "turn_only"),
                    confidence=float(match.get("confidence", 1.0)),
                    similarity=float(match["similarity"]),
                )
            )
        contexts: dict[str, ScoreEvidence] = {}
        for match in low_confidence_matches.values():
            for context in self._transcript_contexts(rule=rule, match=match):
                previous = contexts.get(context.evidence_id)
                if previous is None or (context.similarity or -1) > (
                    previous.similarity or -1
                ):
                    contexts[context.evidence_id] = context
        return _select_period_evidence(
            [*evidence, *contexts.values()],
            final_k=min(
                request.evidence_limit_per_rule,
                rule.retrieval_final_k,
            ),
            per_call_k=rule.retrieval_per_call_k,
            low_confidence_k=rule.retrieval_low_confidence_k,
        )

    def retrieve(
        self,
        *,
        request: PeriodScoreRequest,
        call: CallTarget,
        rule: ScoreRule,
        queries: list[str] | None = None,
    ) -> list[ScoreEvidence]:
        """Compatibility wrapper for callers that intentionally scope one call."""
        return self.retrieve_period(
            request=request,
            calls=[call],
            rule=rule,
            queries=queries,
        )

    def _transcript_contexts(
        self, *, rule: ScoreRule, match: dict
    ) -> list[ScoreEvidence]:
        run_id = str(match["analysis_run_id"])
        chunks = self._transcript_chunks_by_run.get(run_id)
        if chunks is None:
            chunks = list(
                self._session.scalars(
                    select(Document)
                    .where(
                        Document.analysis_run_id == run_id,
                        Document.doc_type == "transcript_chunk",
                        Document.is_active.is_(True),
                    )
                    .order_by(Document.ordinal)
                ).all()
            )
            self._transcript_chunks_by_run[run_id] = chunks
        anchor_turns = {
            item.get("turn_no")
            for item in match.get("evidence") or []
            if item.get("turn_no") is not None
        }
        if not anchor_turns:
            return []
        result: list[ScoreEvidence] = []
        for chunk in chunks:
            chunk_turns = set((chunk.metadata_json or {}).get("turn_ids") or [])
            if not anchor_turns.intersection(chunk_turns):
                continue
            result.append(
                ScoreEvidence(
                    evidence_id=f"{rule.rule_id}:transcript:{chunk.id}",
                    retrieved_for_rule_id=rule.rule_id,
                    source="transcript_context",
                    call_id=str(chunk.call_id),
                    analysis_run_id=run_id,
                    fact_id=str(match["fact_id"]),
                    document_id=str(chunk.id),
                    trigger_validation_status=str(match.get("validation_status") or "low_confidence"),
                    fact=chunk.content,
                    quote=chunk.content,
                    grounding="exact",
                    confidence=1.0,
                    similarity=float(match["similarity"]),
                )
            )
        return result

    def _interaction_metric(self, call: CallTarget) -> ScoreEvidence | None:
        if call.analysis_run_id is None:
            return None
        turns = self._session.scalars(
            select(CallTurn)
            .where(CallTurn.analysis_run_id == call.analysis_run_id)
            .order_by(CallTurn.turn_no)
        ).all()
        role_turns = [item for item in turns if item.speaker_role in {"sales", "customer"}]
        if not role_turns:
            return None

        blocks: list[str] = []
        for turn in role_turns:
            if not blocks or blocks[-1] != turn.speaker_role:
                blocks.append(turn.speaker_role)
        block_counts: dict[str, int] = defaultdict(int)
        for role in blocks:
            block_counts[role] += 1
        round_trips = min(block_counts["sales"], block_counts["customer"])
        customer_chars = sum(len(item.text) for item in role_turns if item.speaker_role == "customer")
        total_chars = sum(len(item.text) for item in role_turns)
        customer_ratio = customer_chars / total_chars if total_chars else 0.0
        timestamps = [item.timestamp_ms for item in role_turns if item.timestamp_ms is not None]
        duration_seconds = (
            (max(timestamps) - min(timestamps)) / 1000 if len(timestamps) >= 2 else None
        )
        lead_stage = call.sales_stage == "销售线索"
        required_round_trips = 3 if lead_stage else 5
        required_ratio = 0.20 if lead_stage else 0.35
        met = round_trips >= required_round_trips or (
            duration_seconds is not None
            and duration_seconds >= 40
            and customer_ratio >= required_ratio
        )
        fact = (
            f"互动来回={round_trips}，时长={duration_seconds if duration_seconds is not None else '未知'}秒，"
            f"客户发言占比={customer_ratio:.3f}，阶段={call.sales_stage}，规则命中={met}"
        )
        return ScoreEvidence(
            evidence_id=f"conversation_interaction:metric:{call.call_id}",
            retrieved_for_rule_id="conversation_interaction",
            source="call_metric",
            call_id=call.call_id,
            analysis_run_id=call.analysis_run_id,
            fact_id=None,
            fact_type="computed_interaction_metric",
            speaker=None,
            fact=fact,
            turn_no=None,
            quote=fact,
            grounding="computed",
            confidence=1.0,
            similarity=None,
        )


def _select_period_evidence(
    candidates: list[ScoreEvidence],
    *,
    final_k: int,
    per_call_k: int,
    low_confidence_k: int,
) -> list[ScoreEvidence]:
    """Select high-similarity evidence without letting one call dominate."""
    ordered = sorted(
        candidates,
        key=lambda item: item.similarity if item.similarity is not None else -1,
        reverse=True,
    )
    selected: list[ScoreEvidence] = []
    selected_ids: set[str] = set()
    call_counts: dict[str, int] = defaultdict(int)
    low_confidence_count = 0
    for item in ordered:
        if len(selected) >= final_k:
            break
        if item.evidence_id in selected_ids:
            continue
        if call_counts[item.call_id] >= per_call_k:
            continue
        is_low_confidence_context = item.source == "transcript_context"
        if is_low_confidence_context and low_confidence_count >= low_confidence_k:
            continue
        selected.append(item)
        selected_ids.add(item.evidence_id)
        call_counts[item.call_id] += 1
        if is_low_confidence_context:
            low_confidence_count += 1
    return selected
