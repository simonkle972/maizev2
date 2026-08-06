"""Phase B latency Phase 1 — Document.full_text cache column

Revision ID: 0e8c3497d34a
Revises: c05842b2af9c
Create Date: 2026-08-06

Adds Document.full_text (Text, nullable) — cached extracted text populated
once at indexing time. Read by src/retriever.py:get_full_document_text as the
fast path for the hybrid_full_doc fallback, avoiding 30-40s per-fire PDF
vision re-extraction observed in the July 2026 ECON S1117 pilot.

down_revision is set to c05842b2af9c (B10) rather than the more-recent
c9a05001c6b8 (B9) deliberately — the B9 migration file was removed by the
2026-05-26 revert (commit 6a0975a) but the local dev DB still has that
migration marked as applied. Threading directly from B10 bypasses that
alembic wrinkle without needing to un-revert B9.

Hand-crafted (not autogen) so we avoid the recurring chat_message_images FK
+ bm25_tsvector GIN false-positives Alembic surfaces on every diff.
"""
from alembic import op
import sqlalchemy as sa


revision = '0e8c3497d34a'
down_revision = 'c05842b2af9c'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('full_text', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_column('full_text')
