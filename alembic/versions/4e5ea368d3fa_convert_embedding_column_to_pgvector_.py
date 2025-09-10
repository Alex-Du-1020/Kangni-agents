"""Convert embedding column to pgvector type

Revision ID: 4e5ea368d3fa
Revises: e1af37c3d2d7
Create Date: 2025-09-10 11:13:18.867642

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4e5ea368d3fa'
down_revision: Union[str, None] = 'e1af37c3d2d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the vector extension if it doesn't exist
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Convert the embedding column from ARRAY to vector type
    # First, drop any existing data since we're changing the type
    op.execute('TRUNCATE TABLE field_value_embeddings')
    
    # Now alter the column type to vector(1024)
    op.execute('ALTER TABLE field_value_embeddings ALTER COLUMN embedding TYPE vector(1024) USING embedding::vector(1024)')


def downgrade() -> None:
    # Convert back to ARRAY type
    op.execute('ALTER TABLE field_value_embeddings ALTER COLUMN embedding TYPE double precision[] USING ARRAY[embedding]')
