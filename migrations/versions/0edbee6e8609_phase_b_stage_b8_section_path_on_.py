"""Phase B Stage B8 — section_path on DocumentChunk

Adds the multi-level structural path column (architecture audit 2026-05-23,
Group B finding #1). Replaces single-header `chunk_context` as the load-bearing
structural-navigation signal, but kept additively: chunk_context survives for
backwards-compat with the existing B15 structural injection block until that
block is deleted in a follow-up cleanup commit.

Hand-cleaned to remove Alembic autogen noise:
- chat_message_images FK drop+recreate would silently lose the ondelete='CASCADE'
  directive. Stripped.
- bm25_tsvector GIN index is correctly defined in models.py but Alembic doesn't
  recognize the postgresql_using='gin' clause and proposes dropping + recreating
  as a btree. Stripped.

Revision ID: 0edbee6e8609
Revises: 9286c9f6860f
Create Date: 2026-05-24 23:22:50.347055

"""
from alembic import op
import sqlalchemy as sa


revision = '0edbee6e8609'
down_revision = '9286c9f6860f'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('section_path', sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table('document_chunks', schema=None) as batch_op:
        batch_op.drop_column('section_path')
