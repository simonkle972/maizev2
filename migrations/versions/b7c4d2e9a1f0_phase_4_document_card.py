"""Phase 4: document card columns, chunk doc_label + lexical index, HNSW

The document card is one typed identity per document (title as students say it, number,
part, term, aliases), decided once at indexing by a strong model that sees the sibling
documents, and served identically to the document search, the reranker, the generator
and the professor UI. See attached_assets/maize-phase4-document-card-plan-2026-09-16.md.

Additive only:
- documents.card_title / card_number / card_part / card_term / card_aliases (JSON list) /
  card_source ('llm' | 'professor') / card_generated_at
- document_chunks.doc_label (the card label, denormalised) and
  document_chunks.search_tsvector (label + aliases + section path + chunk text) with a GIN
  index -- the chunk-level lexical index the 2026-09-13 review asked for
- an HNSW index on document_chunks.embedding (the IVFFlat index stays until a cleanup)

Revision ID: b7c4d2e9a1f0
Revises: 4a91c7d3be05
Create Date: 2026-09-16 21:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR


revision = 'b7c4d2e9a1f0'
down_revision = '4a91c7d3be05'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('card_title', sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column('card_number', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('card_part', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('card_term', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('card_aliases', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('card_source', sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column('card_generated_at', sa.DateTime(), nullable=True))

    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('doc_label', sa.String(length=256), nullable=True))
        batch_op.add_column(sa.Column('search_tsvector', TSVECTOR(), nullable=True))

    op.execute("CREATE INDEX IF NOT EXISTS ix_document_chunks_search_tsvector "
               "ON document_chunks USING gin (search_tsvector)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
               "ON document_chunks USING hnsw (embedding vector_cosine_ops)")


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_search_tsvector")
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.drop_column('search_tsvector')
        batch_op.drop_column('doc_label')
    with op.batch_alter_table('documents', schema=None) as batch_op:
        for col in ('card_generated_at', 'card_source', 'card_aliases', 'card_term',
                    'card_part', 'card_number', 'card_title'):
            batch_op.drop_column(col)
