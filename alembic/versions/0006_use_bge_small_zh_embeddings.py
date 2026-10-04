"""Switch document embeddings from OpenAI 1536d to local BGE Chinese 512d.

Revision ID: 0006_bge_small_zh
Revises: 0005_speaker_resolution
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


revision: str = "0006_bge_small_zh"
down_revision: str | Sequence[str] | None = "0005_speaker_resolution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index(
        "ix_document_embeddings_embedding_hnsw",
        table_name="document_embeddings",
        postgresql_using="hnsw",
    )
    op.execute("DELETE FROM document_embeddings")
    op.drop_column("document_embeddings", "embedding")
    op.add_column(
        "document_embeddings",
        sa.Column("embedding", Vector(512), nullable=False),
    )
    op.create_index(
        "ix_document_embeddings_embedding_hnsw",
        "document_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_embeddings_embedding_hnsw",
        table_name="document_embeddings",
        postgresql_using="hnsw",
    )
    op.execute("DELETE FROM document_embeddings")
    op.drop_column("document_embeddings", "embedding")
    op.add_column(
        "document_embeddings",
        sa.Column("embedding", Vector(1536), nullable=False),
    )
    op.create_index(
        "ix_document_embeddings_embedding_hnsw",
        "document_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
