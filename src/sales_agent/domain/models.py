import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_id", "user_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    summary: Mapped[str | None] = mapped_column(Text)
    summary_through_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "messages.id",
            name="fk_conversations_summary_through_message_id",
            use_alter=True,
        ),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('system', 'user', 'assistant', 'tool')",
            name="ck_messages_role",
        ),
        UniqueConstraint(
            "conversation_id", "sequence_no", name="uq_messages_conversation_sequence"
        ),
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_conversation_started", "conversation_id", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    graph_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_run_id", "run_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    arguments_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    approval_status: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (
        CheckConstraint(
            "analysis_status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_calls_analysis_status",
        ),
        UniqueConstraint(
            "user_id", "external_call_id", name="uq_calls_user_external_id"
        ),
        Index("ix_calls_user_date", "user_id", "call_date"),
        Index("ix_calls_sales_date", "sales_id", "call_date"),
        Index("ix_calls_analysis_status", "analysis_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_call_id: Mapped[str] = mapped_column(String(255), nullable=False)
    sales_id: Mapped[str] = mapped_column(String(64), nullable=False)
    call_date: Mapped[date] = mapped_column(Date, nullable=False)
    sales_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_encoding: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_source_text: Mapped[str] = mapped_column(Text, nullable=False)
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_is_synthetic: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    analysis_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'pending'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CallAnalysisRun(Base):
    __tablename__ = "call_analysis_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_call_analysis_runs_status",
        ),
        Index("ix_call_analysis_runs_call_started", "call_id", "started_at"),
        Index("ix_call_analysis_runs_fingerprint", "call_id", "input_fingerprint"),
        Index(
            "ix_call_analysis_runs_namespace_call_started",
            "knowledge_namespace",
            "call_id",
            "started_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False)
    speaker_resolver_version: Mapped[str] = mapped_column(String(64), nullable=False)
    speaker_mapping_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    document_builder_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    knowledge_namespace: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=text("'baseline-react-v4'")
    )
    enrichment_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'completed'")
    )
    degraded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    degradation_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_summary: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CallTurn(Base):
    __tablename__ = "call_turns"
    __table_args__ = (
        UniqueConstraint("analysis_run_id", "turn_no", name="uq_call_turns_run_turn"),
        Index("ix_call_turns_call_turn", "call_id", "turn_no"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("call_analysis_runs.id", ondelete="CASCADE"), nullable=False)
    turn_no: Mapped[int] = mapped_column(Integer, nullable=False)
    source_speaker_label: Mapped[str] = mapped_column(String(64), nullable=False)
    speaker_role: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp_ms: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_line: Mapped[str] = mapped_column(Text, nullable=False)


class CallFact(Base):
    __tablename__ = "call_facts"
    __table_args__ = (
        UniqueConstraint("analysis_run_id", "fact_key", name="uq_call_facts_run_key"),
        Index("ix_call_facts_call_type", "call_id", "fact_type"),
        Index(
            "ix_call_facts_validation_confidence",
            "validation_status",
            "confidence",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("call_analysis_runs.id", ondelete="CASCADE"), nullable=False)
    fact_key: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str] = mapped_column(String(64), nullable=False)
    fact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    speaker: Mapped[str] = mapped_column(String(32), nullable=False)
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)
    explicit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float] = mapped_column(
        nullable=False, server_default=text("1.0")
    )
    validation_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'verified'")
    )
    quality_issues: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    score_tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "doc_type IN ('structured_fact', 'transcript_chunk', 'full_transcript')",
            name="ck_documents_doc_type",
        ),
        UniqueConstraint("analysis_run_id", "source_key", name="uq_documents_run_source_key"),
        Index("ix_documents_user_type_active", "user_id", "doc_type", "is_active"),
        Index("ix_documents_call_active", "call_id", "is_active"),
        Index(
            "ix_documents_namespace_user_type_active",
            "knowledge_namespace",
            "user_id",
            "doc_type",
            "is_active",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("call_analysis_runs.id", ondelete="CASCADE"), nullable=False)
    knowledge_namespace: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=text("'baseline-react-v4'")
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    sales_id: Mapped[str] = mapped_column(String(64), nullable=False)
    call_date: Mapped[date] = mapped_column(Date, nullable=False)
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_key: Mapped[str] = mapped_column(String(255), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
