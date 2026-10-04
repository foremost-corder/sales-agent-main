"""Add the common user-and-date lookup index for call tools.

Revision ID: 0003_calls_user_date
Revises: 0002_create_calls
"""
from collections.abc import Sequence

from alembic import op


revision: str = "0003_calls_user_date"
down_revision: str | Sequence[str] | None = "0002_create_calls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_calls_user_date", "calls", ["user_id", "call_date"])


def downgrade() -> None:
    op.drop_index("ix_calls_user_date", table_name="calls")
