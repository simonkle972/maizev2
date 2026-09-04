"""Add vendor_deletions queue for post-commit external cleanup

Revision ID: 4a91c7d3be05
Revises: 0e8c3497d34a
Create Date: 2026-09-03 00:00:00.000000

Durable record of Stripe / Auth0 / filesystem cleanups owed after an account or
TA delete commits, so a failed vendor call is retried instead of silently
leaving a subscription billing. See utils/vendor_deletion.py.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4a91c7d3be05'
down_revision = '0e8c3497d34a'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'vendor_deletions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('target', sa.String(length=32), nullable=False),
        sa.Column('external_id', sa.String(length=1024), nullable=True),
        sa.Column('origin', sa.String(length=128), nullable=False),
        sa.Column('attempts', sa.Integer(), server_default='0', nullable=False),
        sa.Column('last_attempt_at', sa.DateTime(), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('vendor_deletions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_vendor_deletions_completed_at'),
                              ['completed_at'], unique=False)


def downgrade():
    with op.batch_alter_table('vendor_deletions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_vendor_deletions_completed_at'))
    op.drop_table('vendor_deletions')
