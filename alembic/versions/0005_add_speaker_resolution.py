"""Persist model-resolved speaker roles for each analysis run.

Revision ID: 0005_speaker_resolution
Revises: 0004_call_knowledge
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0005_speaker_resolution"
down_revision: str | Sequence[str] | None = "0004_call_knowledge"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "call_analysis_runs",
        sa.Column(
            "speaker_resolver_version",
            sa.String(length=64),
            server_default="speaker-roles-v1",
            nullable=False,
        ),
    )
    op.add_column(
        "call_analysis_runs",
        sa.Column("speaker_mapping_json", postgresql.JSONB(), nullable=True),
    )
    op.alter_column(
        "call_analysis_runs", "speaker_resolver_version", server_default=None
    )


def downgrade() -> None:
    op.drop_column("call_analysis_runs", "speaker_mapping_json")
    op.drop_column("call_analysis_runs", "speaker_resolver_version")
