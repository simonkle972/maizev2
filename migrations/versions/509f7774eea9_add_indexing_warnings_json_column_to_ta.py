"""Add indexing_warnings JSON column to TA

Captures per-document indexing outcomes (success/failure with reason) so the
manage_ta UI can surface which docs failed silently during a re-index. See
Phase A indexing-bugs work in .claude/plans/humble-mapping-biscuit.md.

Revision ID: 509f7774eea9
Revises: 0d5e2a36f74c
Create Date: 2026-05-22 01:39:57.224637

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '509f7774eea9'
down_revision = '0d5e2a36f74c'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('teaching_assistants', schema=None) as batch_op:
        batch_op.add_column(sa.Column('indexing_warnings', sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table('teaching_assistants', schema=None) as batch_op:
        batch_op.drop_column('indexing_warnings')
