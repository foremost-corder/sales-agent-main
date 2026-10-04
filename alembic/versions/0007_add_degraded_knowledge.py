"""Track degraded enrichment and low-confidence facts.

Revision ID: 0007_degraded_knowledge
Revises: 0006_bge_small_zh
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0007_degraded_knowledge"
down_revision: str | Sequence[str] | None = "0006_bge_small_zh"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "call_analysis_runs",
        sa.Column(
            "enrichment_status",
            sa.String(length=32),
            nullable=False,
            server_default="completed",
        ),
    )
    op.add_column(
        "call_analysis_runs",
        sa.Column("degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "call_analysis_runs",
        sa.Column(
            "degradation_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "call_facts",
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
    )
    op.add_column(
        "call_facts",
        sa.Column(
            "validation_status",
            sa.String(length=32),
            nullable=False,
            server_default="verified",
        ),
    )
    op.add_column(
        "call_facts",
        sa.Column(
            "quality_issues",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("call_facts", "quality_issues")
    op.drop_column("call_facts", "validation_status")
    op.drop_column("call_facts", "confidence")
    op.drop_column("call_analysis_runs", "degradation_json")
    op.drop_column("call_analysis_runs", "degraded")
    op.drop_column("call_analysis_runs", "enrichment_status")
