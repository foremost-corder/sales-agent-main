"""Single-call parsing, fact extraction, document building and vector ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import logging
import math
from time import perf_counter
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from sales_agent.agent.chat_model import (
    ChatModelNotConfiguredError,
    ChatModelRequestError,
)
from sales_agent.domain.models import (
    Call,
    CallAnalysisRun,
    CallFact,
    CallTurn,
    Document,
    DocumentEmbedding,
)
from sales_agent.features.knowledge.contracts import (
    CallKnowledgeResult,
    EmbeddingProvider,
    ExtractedFact,
    FactExtractionResult,
    FactExtractor,
    KnowledgeDocumentSpec,
    SpeakerRoleResolver,
)
from sales_agent.features.knowledge.documents import DOCUMENT_BUILDER_VERSION, build_knowledge_documents
from sales_agent.features.knowledge.fact_extractor import FactExtractionError
from sales_agent.features.knowledge.namespaces import (
    BASELINE_KNOWLEDGE_NAMESPACE,
    validate_knowledge_namespace,
)
from sales_agent.features.knowledge.parser import PARSER_VERSION, parse_transcript_turns
from sales_agent.features.knowledge.speaker_resolver import (
    SpeakerResolutionError,
    apply_speaker_resolution,
)


logger = logging.getLogger(__name__)


class CallKnowledgeNotFoundError(LookupError):
    pass


class CallKnowledgeConfigurationError(ValueError):
    pass


class AnalyzeCallKnowledgeWorkflow:
    """Domain workflow; independent of the chat tool protocol."""

    def __init__(
        self,
        session: Session,
        *,
        speaker_resolver: SpeakerRoleResolver | None,
        extractor: FactExtractor,
        embedder: EmbeddingProvider,
        full_document_direct_limit: int = 12_000,
        degrade_single_source_speaker_label: bool = True,
    ) -> None:
        if embedder.dimensions != 512:
            raise CallKnowledgeConfigurationError(
                "document_embeddings is fixed at 512 dimensions"
            )
        self._session = session
        self._speaker_resolver = speaker_resolver
        self._extractor = extractor
        self._embedder = embedder
        self._full_document_direct_limit = full_document_direct_limit
        self._degrade_single_source_speaker_label = degrade_single_source_speaker_label

    @property
    def _speaker_resolver_version(self) -> str:
        if self._speaker_resolver is None:
            return "fact-level-speaker-v1"
        return self._speaker_resolver.version

    def execute(
        self,
        *,
        call_id: UUID,
        user_id: str,
        force: bool = False,
        knowledge_namespace: str = BASELINE_KNOWLEDGE_NAMESPACE,
    ) -> CallKnowledgeResult:
        knowledge_namespace = validate_knowledge_namespace(knowledge_namespace)
        workflow_started = perf_counter()
        logger.info(
            "call_knowledge_start call_id=%s user_id=%s force=%s namespace=%s",
            call_id,
            user_id,
            force,
            knowledge_namespace,
        )
        call = self._session.scalar(
            select(Call).where(Call.id == call_id, Call.user_id == user_id)
        )
        if call is None:
            logger.warning(
                "call_knowledge_not_found call_id=%s user_id=%s", call_id, user_id
            )
            raise CallKnowledgeNotFoundError(str(call_id))

        fingerprint = self._fingerprint(call)
        if not force:
            existing = self._session.scalar(
                select(CallAnalysisRun)
                .where(
                    CallAnalysisRun.call_id == call.id,
                    CallAnalysisRun.knowledge_namespace == knowledge_namespace,
                    CallAnalysisRun.input_fingerprint == fingerprint,
                    CallAnalysisRun.status == "completed",
                )
                .order_by(CallAnalysisRun.finished_at.desc())
                .limit(1)
            )
            if existing is not None:
                logger.info(
                    "call_knowledge_reused call_id=%s analysis_run_id=%s elapsed_ms=%d",
                    call.id,
                    existing.id,
                    _elapsed_ms(workflow_started),
                )
                return self._result_for_run(call, existing, reused=True)

        run = CallAnalysisRun(
            call_id=call.id,
            input_fingerprint=fingerprint,
            status="running",
            parser_version=PARSER_VERSION,
            speaker_resolver_version=self._speaker_resolver_version,
            extractor_version=self._extractor.version,
            document_builder_version=DOCUMENT_BUILDER_VERSION,
            embedding_model=self._embedder.model,
            embedding_dimensions=self._embedder.dimensions,
            knowledge_namespace=knowledge_namespace,
        )
        call.analysis_status = "running"
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        logger.info(
            "call_knowledge_run_created call_id=%s analysis_run_id=%s",
            call.id,
            run.id,
        )

        speaker_resolution = None
        degradations: list[dict[str, str]] = []
        try:
            stage_started = perf_counter()
            unresolved_turns = parse_transcript_turns(call.transcript_text)
            logger.info(
                "call_knowledge_parse_completed call_id=%s analysis_run_id=%s turn_count=%d elapsed_ms=%d",
                call.id,
                run.id,
                len(unresolved_turns),
                _elapsed_ms(stage_started),
            )
            stage_started = perf_counter()
            if self._speaker_resolver is None:
                turns = unresolved_turns
                run.speaker_mapping_json = {
                    "status": "fact_level",
                    "source_speaker_labels": sorted(
                        {turn.source_speaker_label for turn in unresolved_turns}
                    ),
                    "strategy": "assigned_per_fact_by_extractor",
                }
                logger.info(
                    "call_knowledge_speakers_skipped call_id=%s analysis_run_id=%s elapsed_ms=%d",
                    call.id,
                    run.id,
                    _elapsed_ms(stage_started),
                )
            else:
                try:
                    speaker_resolution = self._speaker_resolver.resolve(
                        call_id=str(call.id), turns=unresolved_turns
                    )
                except (
                    SpeakerResolutionError,
                    ChatModelRequestError,
                    ChatModelNotConfiguredError,
                ) as exc:
                    turns = unresolved_turns
                    issue = _degradation("speaker_resolution", exc)
                    degradations.append(issue)
                    run.speaker_mapping_json = {
                        "status": "unresolved",
                        "source_speaker_labels": sorted(
                            {turn.source_speaker_label for turn in unresolved_turns}
                        ),
                        "issue": issue,
                    }
                    logger.warning(
                        "call_knowledge_speakers_degraded call_id=%s analysis_run_id=%s error_type=%s elapsed_ms=%d",
                        call.id,
                        run.id,
                        type(exc).__name__,
                        _elapsed_ms(stage_started),
                    )
                else:
                    turns = apply_speaker_resolution(unresolved_turns, speaker_resolution)
                    run.speaker_mapping_json = {
                        "status": "resolved",
                        **speaker_resolution.model_dump(mode="json"),
                    }
                    logger.info(
                        "call_knowledge_speakers_completed call_id=%s analysis_run_id=%s labels=%d elapsed_ms=%d",
                        call.id,
                        run.id,
                        len(speaker_resolution.assignments),
                        _elapsed_ms(stage_started),
                    )
            stage_started = perf_counter()
            try:
                facts = self._extractor.extract(call_id=str(call.id), turns=turns)
            except (
                FactExtractionError,
                ChatModelRequestError,
                ChatModelNotConfiguredError,
            ) as exc:
                issue = _degradation("fact_extraction", exc)
                degradations.append(issue)
                facts = FactExtractionResult(
                    schema_version="conversation-facts-v1",
                    call_id=str(call.id),
                    facts=[],
                    status="skipped",
                    quality_issues=[issue["code"]],
                )
                logger.warning(
                    "call_knowledge_facts_degraded call_id=%s analysis_run_id=%s error_type=%s elapsed_ms=%d",
                    call.id,
                    run.id,
                    type(exc).__name__,
                    _elapsed_ms(stage_started),
                )
            if (
                self._degrade_single_source_speaker_label
                and len({turn.source_speaker_label for turn in unresolved_turns}) < 2
            ):
                facts = _degrade_facts(
                    facts,
                    issue="single_source_speaker_label",
                    maximum_confidence=0.6,
                )
            if facts.status == "low_confidence" and not any(
                item["stage"] == "fact_quality" for item in degradations
            ):
                degradations.append(
                    {
                        "stage": "fact_quality",
                        "code": "low_confidence_facts",
                        "summary": "; ".join(facts.quality_issues)[:1000],
                    }
                )
            enrichment_status = (
                "skipped"
                if facts.status == "skipped"
                else "low_confidence"
                if degradations or facts.status == "low_confidence"
                else "completed"
            )
            run.enrichment_status = enrichment_status
            run.degraded = enrichment_status != "completed"
            run.degradation_json = degradations
            logger.info(
                "call_knowledge_facts_completed call_id=%s analysis_run_id=%s fact_count=%d low_confidence_count=%d enrichment_status=%s elapsed_ms=%d",
                call.id,
                run.id,
                len(facts.facts),
                sum(item.validation_status == "low_confidence" for item in facts.facts),
                enrichment_status,
                _elapsed_ms(stage_started),
            )
            common_metadata = {
                "call_id": str(call.id),
                "external_call_id": call.external_call_id,
                "user_id": call.user_id,
                "sales_id": call.sales_id,
                "call_date": call.call_date.isoformat(),
                "sales_stage": call.sales_stage,
                "source_filename": call.source_filename,
                "source_encoding": call.source_encoding,
                "source_hash": call.source_hash,
                "metadata_is_synthetic": call.metadata_is_synthetic,
                "parser_version": PARSER_VERSION,
                "speaker_resolver_version": self._speaker_resolver_version,
                "speaker_mapping": run.speaker_mapping_json,
                "extractor_version": self._extractor.version,
                "document_builder_version": DOCUMENT_BUILDER_VERSION,
                "enrichment_status": enrichment_status,
                "degraded": run.degraded,
                "degradation": degradations,
                "knowledge_namespace": knowledge_namespace,
            }
            specs = build_knowledge_documents(
                call_id=str(call.id),
                raw_source_text=call.raw_source_text,
                turns=turns,
                facts=facts,
                common_metadata=common_metadata,
            )
            logger.info(
                "call_knowledge_documents_built call_id=%s analysis_run_id=%s document_count=%d",
                call.id,
                run.id,
                len(specs),
            )
            stage_started = perf_counter()
            vectors, strategies = self._embed_documents(specs)
            logger.info(
                "call_knowledge_embeddings_completed call_id=%s analysis_run_id=%s vector_count=%d model=%s elapsed_ms=%d",
                call.id,
                run.id,
                len(vectors),
                self._embedder.model,
                _elapsed_ms(stage_started),
            )
            stage_started = perf_counter()
            self._publish(call, run, turns, facts.facts, specs, vectors, strategies)
            logger.info(
                "call_knowledge_publish_completed call_id=%s analysis_run_id=%s elapsed_ms=%d total_elapsed_ms=%d",
                call.id,
                run.id,
                _elapsed_ms(stage_started),
                _elapsed_ms(workflow_started),
            )
        except Exception as exc:
            logger.exception(
                "call_knowledge_failed call_id=%s analysis_run_id=%s error_type=%s status_code=%s request_id=%s elapsed_ms=%d",
                call.id,
                run.id,
                type(exc).__name__,
                getattr(exc, "status_code", None),
                getattr(exc, "request_id", None),
                _elapsed_ms(workflow_started),
            )
            self._session.rollback()
            failed_run = self._session.get(CallAnalysisRun, run.id)
            failed_call = self._session.get(Call, call.id)
            if failed_run is not None:
                failed_run.status = "failed"
                if run.speaker_mapping_json is not None:
                    failed_run.speaker_mapping_json = run.speaker_mapping_json
                failed_run.error_code = type(exc).__name__[:128]
                failed_run.error_summary = _error_summary(exc)[:2000]
                failed_run.finished_at = datetime.now(timezone.utc)
            if failed_call is not None:
                failed_call.analysis_status = "failed"
            self._session.commit()
            raise

        return self._result_for_run(call, run, reused=False)

    def _fingerprint(self, call: Call) -> str:
        database_text_hash = sha256(
            f"{call.raw_source_text}\0{call.transcript_text}".encode("utf-8")
        ).hexdigest()
        value = "|".join(
            [
                call.source_hash,
                database_text_hash,
                PARSER_VERSION,
                self._speaker_resolver_version,
                self._extractor.version,
                DOCUMENT_BUILDER_VERSION,
                self._embedder.provider_name,
                self._embedder.model,
                str(self._embedder.dimensions),
            ]
        )
        return sha256(value.encode("utf-8")).hexdigest()

    def _embed_documents(
        self, specs: list[KnowledgeDocumentSpec]
    ) -> tuple[list[list[float]], list[str]]:
        full_index = next(index for index, item in enumerate(specs) if item.doc_type == "full_transcript")
        direct_full = len(specs[full_index].embedding_text) <= self._full_document_direct_limit
        direct_indexes = [index for index in range(len(specs)) if index != full_index or direct_full]
        direct_vectors = self._embedder.embed([specs[index].embedding_text for index in direct_indexes])
        by_index = dict(zip(direct_indexes, direct_vectors, strict=True))
        strategies = ["direct_v1"] * len(specs)
        if not direct_full:
            chunk_vectors = [
                by_index[index]
                for index, item in enumerate(specs)
                if item.doc_type == "transcript_chunk"
            ]
            by_index[full_index] = _normalized_mean(chunk_vectors, self._embedder.dimensions)
            strategies[full_index] = "mean_transcript_chunks_v1"
        vectors = [by_index[index] for index in range(len(specs))]
        if any(len(vector) != self._embedder.dimensions for vector in vectors):
            raise ValueError("embedding provider returned an unexpected dimension")
        return vectors, strategies

    def _publish(self, call, run, turns, facts, specs, vectors, strategies) -> None:
        self._session.execute(
            update(Document)
            .where(
                Document.call_id == call.id,
                Document.knowledge_namespace == run.knowledge_namespace,
                Document.is_active.is_(True),
            )
            .values(is_active=False)
        )
        self._session.add_all(
            [
                CallTurn(
                    call_id=call.id,
                    analysis_run_id=run.id,
                    turn_no=turn.turn_no,
                    source_speaker_label=turn.source_speaker_label,
                    speaker_role=turn.speaker.value,
                    timestamp_ms=turn.timestamp_ms,
                    text=turn.text,
                    raw_line=turn.raw_line,
                )
                for turn in turns
            ]
        )
        self._session.add_all(
            [
                CallFact(
                    call_id=call.id,
                    analysis_run_id=run.id,
                    fact_key=fact.fact_id,
                    phase=fact.phase,
                    fact_type=fact.fact_type,
                    speaker=fact.speaker,
                    fact_text=fact.fact,
                    explicit=fact.explicit,
                    confidence=fact.confidence,
                    validation_status=fact.validation_status,
                    quality_issues=fact.quality_issues,
                    score_tags=fact.score_tags,
                    evidence_json=[item.model_dump(mode="json") for item in fact.evidence],
                )
                for fact in facts
            ]
        )
        documents: list[Document] = []
        for spec, strategy in zip(specs, strategies, strict=True):
            document = Document(
                call_id=call.id,
                analysis_run_id=run.id,
                knowledge_namespace=run.knowledge_namespace,
                user_id=call.user_id,
                sales_id=call.sales_id,
                call_date=call.call_date,
                doc_type=spec.doc_type,
                source_key=spec.source_key,
                ordinal=spec.ordinal,
                content=spec.content,
                content_hash=sha256(spec.content.encode("utf-8")).hexdigest(),
                metadata_json={**spec.metadata, "embedding_strategy": strategy},
                is_active=True,
            )
            self._session.add(document)
            documents.append(document)
        self._session.flush()
        self._session.add_all(
            [
                DocumentEmbedding(
                    document_id=document.id,
                    embedding=vector,
                    embedding_provider=self._embedder.provider_name,
                    embedding_model=self._embedder.model,
                    embedding_dimensions=self._embedder.dimensions,
                    embedding_strategy=strategy,
                )
                for document, vector, strategy in zip(documents, vectors, strategies, strict=True)
            ]
        )
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        call.analysis_status = "completed"
        self._session.add_all([run, call])
        self._session.commit()

    def _result_for_run(
        self, call: Call, run: CallAnalysisRun, *, reused: bool
    ) -> CallKnowledgeResult:
        fact_count = self._session.scalar(
            select(func.count()).select_from(CallFact).where(CallFact.analysis_run_id == run.id)
        ) or 0
        low_confidence_fact_count = self._session.scalar(
            select(func.count())
            .select_from(CallFact)
            .where(
                CallFact.analysis_run_id == run.id,
                CallFact.validation_status == "low_confidence",
            )
        ) or 0
        counts = dict(
            self._session.execute(
                select(Document.doc_type, func.count())
                .where(Document.analysis_run_id == run.id)
                .group_by(Document.doc_type)
            ).all()
        )
        document_count = sum(counts.values())
        return CallKnowledgeResult(
            analysis_run_id=str(run.id),
            call_id=str(call.id),
            status="completed",
            reused=reused,
            degraded=run.degraded,
            enrichment_status=run.enrichment_status,
            fact_count=fact_count,
            low_confidence_fact_count=low_confidence_fact_count,
            transcript_chunk_count=counts.get("transcript_chunk", 0),
            full_transcript_count=counts.get("full_transcript", 0),
            document_count=document_count,
            embedding_model=run.embedding_model,
            knowledge_namespace=run.knowledge_namespace,
        )


def _normalized_mean(vectors: list[list[float]], dimensions: int) -> list[float]:
    if not vectors:
        raise ValueError("cannot aggregate an empty vector list")
    mean = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in mean)) or 1.0
    return [value / norm for value in mean]


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


def _error_summary(exc: Exception) -> str:
    parts = [str(exc)]
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if status_code is not None:
        parts.append(f"status_code={status_code}")
    if request_id:
        parts.append(f"request_id={request_id}")
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"cause={type(cause).__name__}: {cause}")
    return " | ".join(parts)


def _degradation(stage: str, exc: Exception) -> dict[str, str]:
    return {
        "stage": stage,
        "code": type(exc).__name__[:128],
        "summary": _error_summary(exc)[:1000],
    }


def _degrade_facts(
    result: FactExtractionResult,
    *,
    issue: str,
    maximum_confidence: float,
) -> FactExtractionResult:
    if result.status == "skipped":
        return result
    facts: list[ExtractedFact] = [
        fact.model_copy(
            update={
                "confidence": min(fact.confidence, maximum_confidence),
                "validation_status": "low_confidence",
                "quality_issues": (
                    fact.quality_issues
                    if issue in fact.quality_issues
                    else [*fact.quality_issues, issue]
                ),
            }
        )
        for fact in result.facts
    ]
    return result.model_copy(
        update={
            "facts": facts,
            "status": "low_confidence",
            "quality_issues": (
                result.quality_issues
                if issue in result.quality_issues
                else [*result.quality_issues, issue]
            ),
        }
    )
