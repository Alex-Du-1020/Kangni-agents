"""Add memory system tables and fields

Revision ID: 5a6b7c8d9e0f
Revises: e1af37c3d2d7
Create Date: 2025-01-11 15:45:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import json

# revision identifiers, used by Alembic.
revision = '5a6b7c8d9e0f'
down_revision = 'e1af37c3d2d7'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns to query_history table
    with op.batch_alter_table('query_history', schema=None) as batch_op:
        batch_op.add_column(sa.Column('memory_summary', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('context_used', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('importance_score', sa.Float(), nullable=True))

    # Create memories table
    op.create_table('memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_email', sa.String(length=255), nullable=False),
        sa.Column('session_id', sa.String(length=100), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.Column('memory_type', sa.Enum('SHORT_TERM', 'LONG_TERM', 'EPISODIC', 'SEMANTIC', name='memorytype'), nullable=False),
        sa.Column('importance', sa.Enum('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='memoryimportance'), nullable=False),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('access_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_accessed', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('source_query_id', sa.Integer(), nullable=True),
        sa.Column('related_entities', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['source_query_id'], ['query_history.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    with op.batch_alter_table('memories', schema=None) as batch_op:
        batch_op.create_index('idx_user_memory_type', ['user_email', 'memory_type'])
        batch_op.create_index('idx_session_memory', ['session_id', 'memory_type'])
        batch_op.create_index('idx_memory_importance', ['importance', 'relevance_score'])
        batch_op.create_index(batch_op.f('ix_memories_created_at'), ['created_at'])
        batch_op.create_index(batch_op.f('ix_memories_session_id'), ['session_id'])
        batch_op.create_index(batch_op.f('ix_memories_user_email'), ['user_email'])


def downgrade():
    # Drop indexes
    with op.batch_alter_table('memories', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_memories_user_email'))
        batch_op.drop_index(batch_op.f('ix_memories_session_id'))
        batch_op.drop_index(batch_op.f('ix_memories_created_at'))
        batch_op.drop_index('idx_memory_importance')
        batch_op.drop_index('idx_session_memory')
        batch_op.drop_index('idx_user_memory_type')
    
    # Drop memories table
    op.drop_table('memories')
    
    # Remove columns from query_history table
    with op.batch_alter_table('query_history', schema=None) as batch_op:
        batch_op.drop_column('importance_score')
        batch_op.drop_column('context_used')
        batch_op.drop_column('memory_summary')