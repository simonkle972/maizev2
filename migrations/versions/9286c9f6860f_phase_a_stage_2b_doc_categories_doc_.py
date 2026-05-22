"""Phase A Stage 2B: doc_categories + doc_category (additive; doc_role stays)

Adds the per-TA configurable document-classification schema (gap analysis
2026-05-22 + research 2026-05-22). Replaces the Stage 2 doc_role enum
SHAPE without dropping the column — additive only, for safe rollback.

Adds:
- TeachingAssistant.doc_categories (JSON) — per-TA array of {slug, label}
  objects defining the controlled vocabulary for this course's documents.
- Document.doc_category (String(64)) — stores the SLUG of one of the
  parent TA's doc_categories. Replaces doc_role as the load-bearing
  classification axis for retrieval (per the Stage 2B design pivot).
- DocumentChunk.doc_category (String(64)) — denormalized copy of
  Document.doc_category so the retriever can filter chunks without
  joining back to documents.

Keeps (for backwards-compat + rollback safety; no longer load-bearing):
- Document.doc_role + Document.doc_role_provenance — to be dropped in
  a future cleanup migration after v1 is stable.
- DocumentChunk.doc_role — same.

Hand-cleaned to:
- Strip the autogen noise on chat_message_images FK (re-creates without
  ondelete=CASCADE — silent regression).
- Preserve the GIN index on documents.bm25_tsvector (Alembic autogen
  doesn't recognize GIN; would have dropped + recreated it as a btree).

Revision ID: 9286c9f6860f
Revises: 6616a6d2eb0a
Create Date: 2026-05-22 15:43:17.044046

"""
from alembic import op
import sqlalchemy as sa


revision = '9286c9f6860f'
down_revision = '6616a6d2eb0a'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('doc_category', sa.String(length=64), nullable=True))

    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('doc_category', sa.String(length=64), nullable=True))

    with op.batch_alter_table('teaching_assistants', schema=None) as batch_op:
        batch_op.add_column(sa.Column('doc_categories', sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table('teaching_assistants', schema=None) as batch_op:
        batch_op.drop_column('doc_categories')

    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_column('doc_category')

    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.drop_column('doc_category')
