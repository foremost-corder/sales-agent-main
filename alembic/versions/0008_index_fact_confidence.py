"""Index fact validation status and confidence.

Revision ID: 0008_fact_confidence_index
Revises: 0007_degraded_knowledge
"""

from collections.abc import Sequence

from alembic import op


revision: str = "0008_fact_confidence_index"
down_revision: str | Sequence[str] | None = "0007_degraded_knowledge"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_call_facts_validation_confidence",
        "call_facts",
        ["validation_status", "confidence"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_call_facts_validation_confidence",
        table_name="call_facts",
    )
