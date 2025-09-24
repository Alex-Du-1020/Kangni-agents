"""
add rewritten_question column to query_history

Revision ID: a1b2c3d4e5f6
Revises: 5a6b7c8d9e0f
Create Date: 2025-09-23
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '5a6b7c8d9e0f'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('query_history', sa.Column('rewritten_question', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('query_history', 'rewritten_question')
