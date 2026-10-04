"""Create versioned call knowledge and pgvector tables.

Revision ID: 0004_call_knowledge
Revises: 0003_calls_user_date
"""
from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0004_call_knowledge"
down_revision: str | Sequence[str] | None = "0003_calls_user_date"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "call_analysis_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("extractor_version", sa.String(length=64), nullable=False),
        sa.Column("document_builder_version", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=128)),
        sa.Column("error_summary", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_call_analysis_runs_status"),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_call_analysis_runs_call_started", "call_analysis_runs", ["call_id", "started_at"])
    op.create_index("ix_call_analysis_runs_fingerprint", "call_analysis_runs", ["call_id", "input_fingerprint"])

    op.create_table(
        "call_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_no", sa.Integer(), nullable=False),
        sa.Column("source_speaker_label", sa.String(length=64), nullable=False),
        sa.Column("speaker_role", sa.String(length=32), nullable=False),
        sa.Column("timestamp_ms", sa.Integer()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("raw_line", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["analysis_run_id"], ["call_analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_run_id", "turn_no", name="uq_call_turns_run_turn"),
    )
    op.create_index("ix_call_turns_call_turn", "call_turns", ["call_id", "turn_no"])

    op.create_table(
        "call_facts",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fact_key", sa.String(length=64), nullable=False),
        sa.Column("phase", sa.String(length=64), nullable=False),
        sa.Column("fact_type", sa.String(length=64), nullable=False),
        sa.Column("speaker", sa.String(length=32), nullable=False),
        sa.Column("fact_text", sa.Text(), nullable=False),
        sa.Column("explicit", sa.Boolean(), nullable=False),
        sa.Column("score_tags", postgresql.JSONB(), nullable=False),
        sa.Column("evidence_json", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["analysis_run_id"], ["call_analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_run_id", "fact_key", name="uq_call_facts_run_key"),
    )
    op.create_index("ix_call_facts_call_type", "call_facts", ["call_id", "fact_type"])

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("call_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("sales_id", sa.String(length=64), nullable=False),
        sa.Column("call_date", sa.Date(), nullable=False),
        sa.Column("doc_type", sa.String(length=32), nullable=False),
        sa.Column("source_key", sa.String(length=255), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("doc_type IN ('structured_fact', 'transcript_chunk', 'full_transcript')", name="ck_documents_doc_type"),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["analysis_run_id"], ["call_analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_run_id", "source_key", name="uq_documents_run_source_key"),
    )
    op.create_index("ix_documents_user_type_active", "documents", ["user_id", "doc_type", "is_active"])
    op.create_index("ix_documents_call_active", "documents", ["call_id", "is_active"])

    op.create_table(
        "document_embeddings",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("embedding_provider", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding_strategy", sa.String(length=64), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id"),
    )
    op.create_index(
        "ix_document_embeddings_embedding_hnsw",
        "document_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_document_embeddings_embedding_hnsw", table_name="document_embeddings", postgresql_using="hnsw")
    op.drop_table("document_embeddings")
    op.drop_index("ix_documents_call_active", table_name="documents")
    op.drop_index("ix_documents_user_type_active", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_call_facts_call_type", table_name="call_facts")
    op.drop_table("call_facts")
    op.drop_index("ix_call_turns_call_turn", table_name="call_turns")
    op.drop_table("call_turns")
    op.drop_index("ix_call_analysis_runs_fingerprint", table_name="call_analysis_runs")
    op.drop_index("ix_call_analysis_runs_call_started", table_name="call_analysis_runs")
    op.drop_table("call_analysis_runs")
