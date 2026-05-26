"""Phase B B9 contextual_prefix on chunks

Revision ID: c9a05001c6b8
Revises: c05842b2af9c
Create Date: 2026-05-25 19:30:39.585835

Phase B Stage B9 — Anthropic Contextual Retrieval. Adds one nullable Text column
on document_chunks for the 1-2 sentence LLM-generated context blob that gets
prepended to chunk text at embedding time. Pre-B9 chunks have NULL here until
the backfill script (scripts/backfill_chunk_contextual_prefix.py) repopulates
them; the indexing pipeline populates this column for all new chunks going
forward.

See attached_assets/maize-b9-contextual-retrieval-plan.md.

Alembic also flagged the recurring chat_message_images FK + bm25_tsvector GIN
index false-positives — stripped from this migration by hand.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c9a05001c6b8'
down_revision = 'c05842b2af9c'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('contextual_prefix', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.drop_column('contextual_prefix')
