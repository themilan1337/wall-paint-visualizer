"""Pydantic schemas for API requests/responses"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ColorInfo(BaseModel):
    """Color information schema"""
    code: str
    hex: str
    rgb: List[int]
    lab: List[float]
    page: int


class ColorSearchRequest(BaseModel):
    """Color search request"""
    query: str = Field(default="", description="Search query")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")


class ColorSearchResponse(BaseModel):
    """Color search response"""
    success: bool
    results: List[ColorInfo]
    count: int
    total: int


class ProcessImageRequest(BaseModel):
    """Image processing request"""
    image: str = Field(..., description="Image filename")
    color: Optional[str] = Field(None, description="RGB color as 'R,G,B'")
    pattern: Optional[str] = Field(None, description="Pattern filename")


class ProcessImageResponse(BaseModel):
    """Image processing response"""
    success: bool
    edited_image: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class UploadResponse(BaseModel):
    """File upload response"""
    success: bool
    filename: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    redis_connected: bool
    model_loaded: bool
    colors_loaded: bool
    total_colors: int
