"""Phase B Stage B10 — doc summary + summary_embedding on Document

Adds per-doc summary text + summary embedding for the audit's Group B finding #3
(cross-cutting indexing change #3 in `attached_assets/maize-architecture-review-2026-05-23.md`).
Sets up the future hybrid_doc_search refactor (~300 lines deletable): doc-routing
via summary-cosine + LLM tiebreaker replaces BM25 + dense + filename RRF + Stage 5
short-circuit. Today the columns are populated at index time + via backfill but
NOT yet read by retrieval — ships as indexing-only infrastructure.

Hand-cleaned to remove Alembic autogen noise:
- chat_message_images FK drop+recreate would silently lose the ondelete='CASCADE'
  directive. Stripped.
- bm25_tsvector GIN index is correctly defined in models.py but Alembic doesn't
  recognize the postgresql_using='gin' clause and proposes dropping + recreating
  as a btree. Stripped.

Revision ID: c05842b2af9c
Revises: 0edbee6e8609
Create Date: 2026-05-25 11:13:08.607622
"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


revision = 'c05842b2af9c'
down_revision = '0edbee6e8609'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('summary', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('summary_embedding', pgvector.sqlalchemy.Vector(dim=1536), nullable=True))


def downgrade():
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_column('summary_embedding')
        batch_op.drop_column('summary')
