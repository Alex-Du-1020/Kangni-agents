"""
Vector embedding service for semantic search on database field values.
"""
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, text, select, and_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert
import logging
from datetime import datetime

from ..models.vector_embedding import FieldValueEmbedding
from ..services.database_service import db_service
from ..config import settings

logger = logging.getLogger(__name__)


class VectorEmbeddingService:
    """Service for managing vector embeddings of database field values."""
    
    def __init__(self):
        """
        Initialize the vector embedding service.
        
        Args:
            database_service: Database service for data access
        """

        db_url = (
            f"postgresql://{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST')}:"
            f"{os.getenv('POSTGRES_PORT')}/"
            f"{os.getenv('POSTGRES_DATABASE')}"
        )

        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using BGE-M3 model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            import httpx
            import os
            
            # BGE-M3 embedding model configuration
            embedding_base_url = os.getenv('EMBEDDING_BASE_URL', 'http://158.158.4.66:4432/v1')
            # Use the API key from test script
            embedding_api_key = getattr(settings, 'embedding_api_key', None) or os.getenv('EMBEDDING_API_KEY')
            
            # Prepare request to BGE-M3 embedding API
            headers = {
                'Authorization': f'Bearer {embedding_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'bge-m3',
                'input': text,
                'encoding_format': 'float'
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f'{embedding_base_url}/embeddings',
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Extract embedding from response
                    # BGE-M3 returns embeddings in OpenAI-compatible format
                    if 'data' in data and len(data['data']) > 0:
                        embedding = data['data'][0].get('embedding', [])
                        
                        if embedding:
                            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
                            return embedding
                        else:
                            raise ValueError("No embedding returned from BGE-M3 model")
                    else:
                        raise ValueError("Invalid response format from BGE-M3 model")
                else:
                    logger.error(f"BGE-M3 API error: {response.status_code} - {response.text}")
                    raise Exception(f"Failed to generate embedding: {response.status_code}")
                    
        except httpx.RequestError as e:
            logger.error(f"Network error connecting to BGE-M3: {e}")
            # Fallback to hash-based embedding if BGE-M3 is unavailable
            logger.warning("Falling back to hash-based embedding")
            return await self._generate_fallback_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback to hash-based embedding
            logger.warning("Falling back to hash-based embedding")
            return await self._generate_fallback_embedding(text)
    
    async def _generate_fallback_embedding(self, text: str) -> List[float]:
        """
        Generate fallback embedding using hash-based approach.
        Used when BGE-M3 model is unavailable.
        """
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # BGE-M3 typically generates 1024-dimensional embeddings
        # We'll match that dimension for consistency
        embedding = []
        for i in range(1024):
            byte_idx = i % len(hash_bytes)
            value = hash_bytes[byte_idx] / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(value)
        
        return embedding
    
    async def store_field_embedding(
        self, 
        table_name: str, 
        field_name: str, 
        field_value: str
    ) -> int:
        """
        Store or retrieve embedding for a field value.
        
        Args:
            table_name: Name of the table
            field_name: Name of the field
            field_value: Value of the field
            
        Returns:
            ID of the embedding record
        """
        with self.SessionLocal() as session:
            # Check if embedding already exists
            existing = session.query(FieldValueEmbedding).filter(
                and_(
                    FieldValueEmbedding.table_name == table_name,
                    FieldValueEmbedding.field_name == field_name,
                    FieldValueEmbedding.field_value == field_value
                )
            ).first()
            
            if existing:
                return existing.id
            
            # Generate new embedding
            embedding = await self.generate_embedding(field_value)

            # For PostgreSQL with pgvector, store as array directly
            # pgvector will handle the conversion
            embedding_data = embedding
            
            # Create new embedding record
            new_embedding = FieldValueEmbedding(
                table_name=table_name,
                field_name=field_name,
                field_value=field_value,
                embedding=embedding_data
            )
            
            session.add(new_embedding)
            session.commit()
            
            return new_embedding.id
    
    async def sync_table_field_data(
        self, 
        table_name: str, 
        field_name: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sync data from business table to embedding tables.
        
        Args:
            table_name: Name of the table to sync
            field_name: Name of the field to sync
            limit: Optional limit on number of records to sync
            
        Returns:
            Dictionary with sync statistics
        """
        try:
            # Query distinct values from the business table
            query = f"""
                SELECT DISTINCT {field_name} as field_value
                FROM {table_name}
                WHERE {field_name} IS NOT NULL
                AND {field_name} != ''
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Executing sync query: {query}")
            
            # Execute the SQL query directly
            try:
                rows = await db_service.execute_sql_query(query)
            except Exception as e:
                logger.error(f"Failed to execute query: {e}")
                return {
                    'success': False,
                    'error': f'Failed to query data: {str(e)}'
                }
            
            if not rows:
                return {
                    'success': True,
                    'synced': 0,
                    'skipped': 0,
                    'errors': 0,
                    'total': 0,
                    'message': 'No data to sync'
                }
            
            # Process each unique value
            synced_count = 0
            skipped_count = 0
            error_count = 0
            
            for row in rows:
                try:
                    field_value = str(row.get('field_value', ''))
                    
                    if not field_value or field_value.strip() == '':
                        skipped_count += 1
                        continue
                    
                    # Store or get embedding
                    embedding_id = await self.store_field_embedding(
                        table_name, field_name, field_value
                    )
                    
                    # For sync operations, we don't need record mappings
                    # since we're only storing unique field values
                    # The mapping will be created when actual queries happen
                    
                    synced_count += 1
                    
                    if synced_count % 100 == 0:
                        logger.info(f"Synced {synced_count} records...")
                    
                except Exception as e:
                    logger.error(f"Error syncing field value '{field_value}': {e}")
                    error_count += 1
            
            logger.info(f"Sync completed: synced={synced_count}, skipped={skipped_count}, errors={error_count}")
            
            return {
                'success': True,
                'synced': synced_count,
                'skipped': skipped_count,
                'errors': error_count,
                'total': len(rows),
                'message': f'Successfully synced {synced_count} unique values from {table_name}.{field_name}'
            }
            
        except Exception as e:
            logger.error(f"Error syncing table field data: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def search_similar_values(
        self,
        search_text: str,
        table_name: str,
        field_name: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar field values using vector similarity.
        
        Args:
            search_text: Text to search for
            table_name: Table to search in
            field_name: Field to search in
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar values with scores
        """
        try:
            # Generate embedding for search text
            search_embedding = await self.generate_embedding(search_text)
            
            with self.SessionLocal() as session:

                # Check if we're using pgvector or ARRAY
                # Try pgvector query first
                # Convert Python list to PostgreSQL array format string
                embedding_str = '[' + ','.join(map(str, search_embedding)) + ']'
                    
                query = text("""
                    SELECT id, field_value, 
                            1 - (embedding <-> CAST(:search_embedding AS vector)) as similarity
                    FROM field_value_embeddings
                    WHERE table_name = :table_name 
                        AND field_name = :field_name
                    ORDER BY embedding <-> CAST(:search_embedding AS vector)
                    LIMIT :top_k
                """)
                    
                results = session.execute(
                    query,
                    {
                        'search_embedding': embedding_str,
                        'table_name': table_name,
                        'field_name': field_name,
                        'top_k': top_k
                    }
                ).fetchall()
        
                
                # Format PostgreSQL results
                return [
                    {
                        'id': r.id,
                        'field_value': r.field_value,
                        'similarity': r.similarity
                    }
                    for r in results
                    if r.similarity >= similarity_threshold
                ]
                
        except Exception as e:
            logger.error(f"Error searching similar values: {e}")
            return []
    
    async def get_field_values_for_query(
        self,
        query_text: str,
        table_name: str = "kn_quality_trace_prod_order",
        field_name: str = "projectname_s",
        similarity_threshold: float = 0.7
    ) -> List[str]:
        """
        Get relevant field values for a query using vector search.
        
        Args:
            query_text: User's query text
            table_name: Table to search in
            field_name: Field to search in
            
        Returns:
            List of relevant field values
        """
        # Extract potential project names from query
        similar_values = await self.search_similar_values(
            query_text,
            table_name,
            field_name,
            top_k=5,
            similarity_threshold=similarity_threshold
        )
        
        return [v['field_value'] for v in similar_values]
    
vector_service = VectorEmbeddingService()