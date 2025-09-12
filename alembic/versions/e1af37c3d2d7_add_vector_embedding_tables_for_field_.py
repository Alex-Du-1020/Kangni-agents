"""Add vector embedding tables for field value search

Revision ID: e1af37c3d2d7
Revises: 006477fcc163
Create Date: 2025-09-09 17:39:33.322904

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import os


# revision identifiers, used by Alembic.
revision: str = 'e1af37c3d2d7'
down_revision: Union[str, None] = '006477fcc163'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # Enable pgvector extension if not already enabled
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    op.create_table('field_value_embeddings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('table_name', sa.String(length=255), nullable=False),
        sa.Column('field_name', sa.String(length=255), nullable=False),
        sa.Column('field_value', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('table_name', 'field_name', 'field_value', name='uq_table_field_value')
    )
    
    # Create indexes for PostgreSQL with vector support
    op.create_index('idx_table_field', 'field_value_embeddings', ['table_name', 'field_name'])
    # Note: IVFFlat index for vector similarity search requires manual creation after data insertion
    # op.execute('CREATE INDEX idx_field_value_embedding ON field_value_embeddings USING ivfflat (embedding vector_l2_ops)')

    # Create record_field_mappings table (same for both databases)
    op.create_table('record_field_mappings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('record_id', sa.String(length=255), nullable=False),
        sa.Column('table_name', sa.String(length=255), nullable=False),
        sa.Column('field_name', sa.String(length=255), nullable=False),
        sa.Column('field_value', sa.Text(), nullable=False),
        sa.Column('embedding_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), 
                  server_default=sa.text('now()' if db_type == 'postgresql' else 'CURRENT_TIMESTAMP'), 
                  nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_record_table', 'record_field_mappings', ['record_id', 'table_name'])
    op.create_index('idx_embedding_ref', 'record_field_mappings', ['embedding_id'])
    op.create_index('idx_table_field_mapping', 'record_field_mappings', ['table_name', 'field_name'])


def downgrade() -> None:
    op.drop_index('idx_table_field_mapping', table_name='record_field_mappings')
    op.drop_index('idx_embedding_ref', table_name='record_field_mappings')
    op.drop_index('idx_record_table', table_name='record_field_mappings')
    op.drop_table('record_field_mappings')
    
    op.drop_index('idx_table_field', table_name='field_value_embeddings')
    op.drop_table('field_value_embeddings')
