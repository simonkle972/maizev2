"""Phase A retrieval refactor: doc_role + provenance + bm25_tsvector

Adds the schema substrate for the Phase A retrieval refactor (gap analysis
2026-05-22). doc_role becomes the PRIMARY semantic axis for retrieval;
doc_role_provenance captures classifier source/confidence; bm25_tsvector
backs the hybrid Stage 1 BM25+dense+RRF document search. denormalized
doc_role on DocumentChunk so the retriever can filter chunks without
joining back to documents.

A GIN index on documents.bm25_tsvector is added explicitly so full-text
queries hit an index (Alembic autogen doesn't infer GIN indexes for
TSVECTOR columns).

Revision ID: 6616a6d2eb0a
Revises: 509f7774eea9
Create Date: 2026-05-22 13:24:35.881720

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '6616a6d2eb0a'
down_revision = '509f7774eea9'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('doc_role', sa.String(length=32), nullable=True))

    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('doc_role', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('doc_role_provenance', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('bm25_tsvector', postgresql.TSVECTOR(), nullable=True))

    # GIN index for full-text search performance over the BM25 tsvector.
    # Required for fast hybrid Stage 1 retrieval at any meaningful corpus size.
    op.create_index(
        'ix_documents_bm25_tsvector',
        'documents',
        ['bm25_tsvector'],
        postgresql_using='gin',
    )


def downgrade():
    op.drop_index('ix_documents_bm25_tsvector', table_name='documents')

    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_column('bm25_tsvector')
        batch_op.drop_column('doc_role_provenance')
        batch_op.drop_column('doc_role')

    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.drop_column('doc_role')
