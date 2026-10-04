"""Create the calls table for manually imported transcripts.

Revision ID: 0002_create_calls
Revises: 0001_core_relational
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_create_calls"
down_revision: str | Sequence[str] | None = "0001_core_relational"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "calls",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("external_call_id", sa.String(length=255), nullable=False),
        sa.Column("sales_id", sa.String(length=64), nullable=False),
        sa.Column("call_date", sa.Date(), nullable=False),
        sa.Column("sales_stage", sa.String(length=64), nullable=False),
        sa.Column("source_filename", sa.String(length=1000), nullable=False),
        sa.Column("source_encoding", sa.String(length=32), nullable=False),
        sa.Column("raw_source_text", sa.Text(), nullable=False),
        sa.Column("transcript_text", sa.Text(), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "metadata_is_synthetic",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "analysis_status",
            sa.String(length=32),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "analysis_status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_calls_analysis_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id", "external_call_id", name="uq_calls_user_external_id"
        ),
    )
    op.create_index("ix_calls_sales_date", "calls", ["sales_id", "call_date"])
    op.create_index("ix_calls_analysis_status", "calls", ["analysis_status"])


def downgrade() -> None:
    op.drop_index("ix_calls_analysis_status", table_name="calls")
    op.drop_index("ix_calls_sales_date", table_name="calls")
    op.drop_table("calls")

