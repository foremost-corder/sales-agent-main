"""Isolate baseline and experimental knowledge corpora.

Revision ID: 0009_knowledge_namespaces
Revises: 0008_fact_confidence_index
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0009_knowledge_namespaces"
down_revision: str | Sequence[str] | None = "0008_fact_confidence_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "call_analysis_runs",
        sa.Column(
            "knowledge_namespace",
            sa.String(length=64),
            nullable=False,
            server_default="baseline-react-v4",
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "knowledge_namespace",
            sa.String(length=64),
            nullable=False,
            server_default="baseline-react-v4",
        ),
    )
    op.create_index(
        "ix_call_analysis_runs_namespace_call_started",
        "call_analysis_runs",
        ["knowledge_namespace", "call_id", "started_at"],
    )
    op.create_index(
        "ix_documents_namespace_user_type_active",
        "documents",
        ["knowledge_namespace", "user_id", "doc_type", "is_active"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_namespace_user_type_active",
        table_name="documents",
    )
    op.drop_index(
        "ix_call_analysis_runs_namespace_call_started",
        table_name="call_analysis_runs",
    )
    op.drop_column("documents", "knowledge_namespace")
    op.drop_column("call_analysis_runs", "knowledge_namespace")
