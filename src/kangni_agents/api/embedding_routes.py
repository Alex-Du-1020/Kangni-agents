"""
API routes for vector embedding operations.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging

from ..services.vector_embedding_service import vector_service
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qomo/v1/embeddings", tags=["embeddings"])

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
        result = await vector_service.sync_table_field_data(table_name, field_name, limit)
        
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
        result = await vector_service.sync_table_field_data(
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
    top_k: int = Query(settings.max_suggestions, description="Number of results to return"),
    threshold: float = Query(settings.similarity_threshold, description="Similarity threshold (0-1)")
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
        results = await vector_service.search_similar_values(
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
