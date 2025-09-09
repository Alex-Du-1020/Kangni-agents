"""
API routes for vector embedding operations.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging

from ..services.vector_embedding_service import VectorEmbeddingService
from ..services.database_service import db_service
from ..services.llm_service import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/embeddings", tags=["embeddings"])

# Initialize the vector embedding service
vector_service = None

def get_vector_service():
    """Get or create the vector embedding service instance."""
    global vector_service
    if vector_service is None:
        vector_service = VectorEmbeddingService(db_service, llm_service)
    return vector_service

@router.post("/sync")
async def sync_field_data(
    table_name: str = Query(..., description="Table name to sync"),
    field_name: str = Query(..., description="Field name to sync"),
    limit: Optional[int] = Query(None, description="Limit number of records to sync")
):
    """
    Sync field data from business table to vector embedding tables.
    
    Args:
        table_name: Name of the business table
        field_name: Name of the field to sync
        limit: Optional limit on number of records
        
    Returns:
        Sync statistics
    """
    try:
        service = get_vector_service()
        result = await service.sync_table_field_data(table_name, field_name, limit)
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Sync failed')
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error syncing field data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/production-orders")
async def sync_production_orders(limit: Optional[int] = Query(None, description="Limit number of records")):
    """
    Sync production order project names to vector embeddings.
    This is a convenience endpoint for the specific use case.
    
    Args:
        limit: Optional limit on number of records
        
    Returns:
        Sync statistics
    """
    try:
        service = get_vector_service()
        result = await service.sync_table_field_data(
            table_name="kn_quality_trace_prod_order",
            field_name="projectname_s",
            limit=limit
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Sync failed')
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error syncing production orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_similar_values(
    query: str = Query(..., description="Search query text"),
    table_name: str = Query("kn_quality_trace_prod_order", description="Table name"),
    field_name: str = Query("projectname_s", description="Field name"),
    top_k: int = Query(10, description="Number of results to return"),
    threshold: float = Query(0.7, description="Similarity threshold (0-1)")
) -> List[Dict[str, Any]]:
    """
    Search for similar field values using vector similarity.
    
    Args:
        query: Text to search for
        table_name: Table to search in
        field_name: Field to search in
        top_k: Number of top results
        threshold: Minimum similarity score
        
    Returns:
        List of similar values with scores
    """
    try:
        service = get_vector_service()
        results = await service.search_similar_values(
            search_text=query,
            table_name=table_name,
            field_name=field_name,
            top_k=top_k,
            similarity_threshold=threshold
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching similar values: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/values")
async def get_field_values(
    query: str = Query(..., description="User query text"),
    table_name: str = Query("kn_quality_trace_prod_order", description="Table name"),
    field_name: str = Query("projectname_s", description="Field name")
) -> List[str]:
    """
    Get relevant field values for a query using vector search.
    
    Args:
        query: User's query text
        table_name: Table to search in
        field_name: Field to search in
        
    Returns:
        List of relevant field values
    """
    try:
        service = get_vector_service()
        values = await service.get_field_values_for_query(
            query_text=query,
            table_name=table_name,
            field_name=field_name
        )
        
        return values
        
    except Exception as e:
        logger.error(f"Error getting field values: {e}")
        raise HTTPException(status_code=500, detail=str(e))