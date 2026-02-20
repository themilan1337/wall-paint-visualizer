"""Color search routes"""
from fastapi import APIRouter, Query, HTTPException
from typing import List
from app.models.schemas import ColorSearchResponse, ColorInfo
from app.services.color_database import color_db

router = APIRouter(prefix="/colors", tags=["colors"])


@router.get("/search", response_model=ColorSearchResponse)
async def search_colors(
    q: str = Query(default="", description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results")
):
    """
    Search colors by code

    - **q**: Search query (e.g., "F00" finds F001, F002, etc.)
    - **limit**: Maximum number of results (1-100)
    """
    try:
        results = color_db.search(q, limit=limit)
        return ColorSearchResponse(
            success=True,
            results=results,
            count=len(results),
            total=color_db.get_count()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}", response_model=ColorInfo)
async def get_color(code: str):
    """
    Get specific color by code

    - **code**: Color code (e.g., "F001", "K059")
    """
    color = color_db.get_by_code(code)
    if not color:
        raise HTTPException(status_code=404, detail="Color not found")
    return color


@router.get("/", response_model=List[str])
async def get_all_codes():
    """Get all available color codes"""
    return color_db.get_all_codes()
