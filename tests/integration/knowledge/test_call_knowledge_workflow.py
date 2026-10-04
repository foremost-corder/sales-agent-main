from datetime import date
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select

from sales_agent.domain.models import (
    Call,
    CallAnalysisRun,
    CallFact,
    CallTurn,
    Document,
    DocumentEmbedding,
)
from sales_agent.core.database import SessionLocal
from sales_agent.features.knowledge.contracts import (
    ExtractedFact,
    FactEvidence,
    FactExtractionResult,
    SpeakerAssignment,
    SpeakerResolutionResult,
)
from sales_agent.features.knowledge.fact_extractor import FactExtractionError
from sales_agent.features.knowledge.standard_workflow import AnalyzeCallKnowledgeWorkflow
from sales_agent.features.knowledge.lean_workflow import AnalyzeCallKnowledgeLeanWorkflow


class FakeFactExtractor:
    version = "fake-facts-v1"

    def extract(self, *, call_id, turns):
        assert [turn.text for turn in turns] == ["您好，想了解需求", "下周需要二十台电脑"]
        return FactExtractionResult(
            schema_version="conversation-facts-v1",
            call_id=call_id,
            facts=[
                ExtractedFact(
                    fact_id="fact-1",
                    phase="discovery",
                    fact_type="customer_need",
                    speaker="customer",
                    fact="客户下周需要二十台电脑。",
                    explicit=True,
                    score_tags=["需求时间", "需求数量"],
                    evidence=[FactEvidence(turn_no=2, quote="下周需要二十台电脑")],
                )
            ],
        )


class FakeSpeakerResolver:
    version = "fake-speaker-roles-v1"

    def resolve(self, *, call_id, turns):
        assert all(turn.speaker.value == "unknown" for turn in turns)
        return SpeakerResolutionResult(
            schema_version="speaker-roles-v1",
            assignments=[
                SpeakerAssignment(
                    source_speaker_label="用户0",
                    role="sales",
                    confidence=0.99,
                    evidence_turn_nos=[1],
                ),
                SpeakerAssignment(
                    source_speaker_label="用户1",
                    role="customer",
                    confidence=0.99,
                    evidence_turn_nos=[2],
                ),
            ],
        )


class FakeEmbeddingProvider:
    provider_name = "fake"
    model = "fake-512"
    dimensions = 512

    def embed(self, texts):
        return [[1.0] + [0.0] * 511 for _ in texts]


class FailingFactExtractor:
    version = "fake-facts-failing-v1"

    def extract(self, *, call_id, turns):
        raise FactExtractionError("facts did not pass validation")


class FakeLeanFactExtractor:
    version = "fake-lean-facts-v1"

    def extract(self, *, call_id, turns):
        assert all(turn.speaker.value == "unknown" for turn in turns)
        return FactExtractionResult(
            schema_version="conversation-facts-v1",
            call_id=call_id,
            facts=[
                ExtractedFact(
                    fact_id="fact-1",
                    phase="discovery",
                    fact_type="customer_need",
                    speaker="customer",
                    fact="客户需要二十台电脑。",
                    explicit=True,
                    confidence=0.9,
                    evidence=[FactEvidence(turn_no=2, quote="需要二十台电脑")],
                )
            ],
        )


@pytest.mark.integration
def test_workflow_reads_call_text_and_publishes_three_document_types() -> None:
    user_id = f"knowledge-user-{uuid4()}"
    raw_source = "【通话ID：knowledge】\n用户0：您好，想了解需求， 0:00:01\n用户1：下周需要二十台电脑， 0:00:03"
    try:
        with SessionLocal.begin() as session:
            call = Call(
                user_id=user_id,
                external_call_id=str(uuid4()),
                sales_id="001",
                call_date=date(2026, 9, 4),
                sales_stage="销售线索",
                source_filename="knowledge.txt",
                source_encoding="utf-8",
                raw_source_text=raw_source,
                transcript_text="用户0：您好，想了解需求， 0:00:01\n用户1：下周需要二十台电脑， 0:00:03",
                source_hash="b" * 64,
                metadata_is_synthetic=False,
                analysis_status="pending",
            )
            session.add(call)

        with SessionLocal() as session:
            workflow = AnalyzeCallKnowledgeWorkflow(
                session,
                speaker_resolver=FakeSpeakerResolver(),
                extractor=FakeFactExtractor(),
                embedder=FakeEmbeddingProvider(),
            )
            first = workflow.execute(call_id=call.id, user_id=user_id)
            second = workflow.execute(call_id=call.id, user_id=user_id)

            documents = session.scalars(
                select(Document)
                .where(Document.call_id == call.id, Document.is_active.is_(True))
                .order_by(Document.doc_type)
            ).all()
            turn_count = session.scalar(select(func.count()).select_from(CallTurn).where(CallTurn.call_id == call.id))
            fact_count = session.scalar(select(func.count()).select_from(CallFact).where(CallFact.call_id == call.id))
            embedding_count = session.scalar(
                select(func.count())
                .select_from(DocumentEmbedding)
                .join(Document, Document.id == DocumentEmbedding.document_id)
                .where(Document.call_id == call.id)
            )

        assert first.reused is False
        assert second.reused is True
        assert first.analysis_run_id == second.analysis_run_id
        assert {document.doc_type for document in documents} == {
            "structured_fact", "transcript_chunk", "full_transcript"
        }
        full_document = next(item for item in documents if item.doc_type == "full_transcript")
        assert full_document.content == raw_source
        assert full_document.metadata_json["source_filename"] == "knowledge.txt"
        assert full_document.metadata_json["speaker_mapping"]["assignments"][0]["role"] == "sales"
        assert turn_count == 2
        assert fact_count == 1
        assert embedding_count == 3
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id == user_id))


@pytest.mark.integration
def test_fact_failure_publishes_raw_documents_as_degraded_success() -> None:
    user_id = f"knowledge-degraded-user-{uuid4()}"
    try:
        with SessionLocal.begin() as session:
            call = Call(
                user_id=user_id,
                external_call_id=str(uuid4()),
                sales_id="001",
                call_date=date(2026, 9, 4),
                sales_stage="销售线索",
                source_filename="degraded.txt",
                source_encoding="utf-8",
                raw_source_text="用户0：您好\n用户1：暂时不需要电脑",
                transcript_text="用户0：您好\n用户1：暂时不需要电脑",
                source_hash="c" * 64,
                metadata_is_synthetic=False,
                analysis_status="pending",
            )
            session.add(call)

        with SessionLocal() as session:
            workflow = AnalyzeCallKnowledgeWorkflow(
                session,
                speaker_resolver=FakeSpeakerResolver(),
                extractor=FailingFactExtractor(),
                embedder=FakeEmbeddingProvider(),
            )
            result = workflow.execute(call_id=call.id, user_id=user_id)
            run = session.get(CallAnalysisRun, UUID(result.analysis_run_id))
            documents = session.scalars(
                select(Document).where(
                    Document.analysis_run_id == result.analysis_run_id
                )
            ).all()

        assert result.status == "completed"
        assert result.degraded is True
        assert result.enrichment_status == "skipped"
        assert result.fact_count == 0
        assert {item.doc_type for item in documents} == {
            "transcript_chunk",
            "full_transcript",
        }
        assert run is not None
        assert run.status == "completed"
        assert run.degraded is True
        assert run.degradation_json[0]["stage"] == "fact_extraction"
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id == user_id))


@pytest.mark.integration
def test_lean_workflow_skips_call_level_speaker_resolution() -> None:
    user_id = f"knowledge-lean-user-{uuid4()}"
    try:
        with SessionLocal.begin() as session:
            call = Call(
                user_id=user_id,
                external_call_id=str(uuid4()),
                sales_id="001",
                call_date=date(2026, 9, 4),
                sales_stage="销售线索",
                source_filename="lean.txt",
                source_encoding="utf-8",
                raw_source_text="用户0：您好\n用户0：需要二十台电脑",
                transcript_text="用户0：您好\n用户0：需要二十台电脑",
                source_hash="d" * 64,
                metadata_is_synthetic=False,
                analysis_status="pending",
            )
            session.add(call)

        with SessionLocal() as session:
            workflow = AnalyzeCallKnowledgeLeanWorkflow(
                session,
                extractor=FakeLeanFactExtractor(),
                embedder=FakeEmbeddingProvider(),
            )
            result = workflow.execute(call_id=call.id, user_id=user_id)
            run = session.get(CallAnalysisRun, UUID(result.analysis_run_id))
            fact = session.scalar(
                select(CallFact).where(CallFact.analysis_run_id == run.id)
            )

        assert result.degraded is False
        assert result.low_confidence_fact_count == 0
        assert run is not None
        assert run.speaker_resolver_version == "fact-level-speaker-v1"
        assert run.speaker_mapping_json["status"] == "fact_level"
        assert fact is not None
        assert fact.speaker == "customer"
        assert fact.validation_status == "verified"
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id == user_id))


@pytest.mark.integration
def test_workflows_keep_active_documents_isolated_by_namespace() -> None:
    user_id = f"knowledge-namespace-user-{uuid4()}"
    try:
        with SessionLocal.begin() as session:
            call = Call(
                user_id=user_id,
                external_call_id=str(uuid4()),
                sales_id="001",
                call_date=date(2026, 9, 4),
                sales_stage="销售线索",
                source_filename="namespace.txt",
                source_encoding="utf-8",
                raw_source_text="用户0：您好，想了解需求\n用户1：下周需要二十台电脑",
                transcript_text="用户0：您好，想了解需求\n用户1：下周需要二十台电脑",
                source_hash="e" * 64,
                metadata_is_synthetic=False,
                analysis_status="pending",
            )
            session.add(call)

        with SessionLocal() as session:
            standard = AnalyzeCallKnowledgeWorkflow(
                session,
                speaker_resolver=FakeSpeakerResolver(),
                extractor=FakeFactExtractor(),
                embedder=FakeEmbeddingProvider(),
            )
            lean = AnalyzeCallKnowledgeLeanWorkflow(
                session,
                extractor=FakeLeanFactExtractor(),
                embedder=FakeEmbeddingProvider(),
            )
            standard.execute(
                call_id=call.id,
                user_id=user_id,
                knowledge_namespace="baseline-react-v4",
            )
            lean.execute(
                call_id=call.id,
                user_id=user_id,
                knowledge_namespace="experiment-lean-v1",
            )
            active_counts = dict(
                session.execute(
                    select(Document.knowledge_namespace, func.count())
                    .where(Document.call_id == call.id, Document.is_active.is_(True))
                    .group_by(Document.knowledge_namespace)
                ).all()
            )

        assert active_counts == {
            "baseline-react-v4": 3,
            "experiment-lean-v1": 3,
        }
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id == user_id))
