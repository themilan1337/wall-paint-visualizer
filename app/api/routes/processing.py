"""Image processing routes"""
import os
import uuid
from fastapi import APIRouter, HTTPException
from app.models.schemas import ProcessImageRequest, ProcessImageResponse
from app.services.image_processor import process_image
from app.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/process", tags=["processing"])


@router.post("/", response_model=ProcessImageResponse)
async def process_wall_image(request: ProcessImageRequest):
    """
    Process image with wall color/pattern change

    - **image**: Image filename (must be uploaded first)
    - **color**: RGB color as "R,G,B" string (e.g., "220,180,170")
    - **pattern**: Pattern filename (optional, instead of color)

    Either color or pattern must be provided, but not both.
    """
    # Validate input
    if not request.color and not request.pattern:
        raise HTTPException(
            status_code=400,
            detail="Either 'color' or 'pattern' must be provided"
        )

    if request.color and request.pattern:
        raise HTTPException(
            status_code=400,
            detail="Only one of 'color' or 'pattern' can be provided"
        )

    # Check if input image exists
    input_path = os.path.join(settings.upload_folder, request.image)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Generate output filename
    output_name = f"edited_{uuid.uuid4().hex[:8]}_{request.image}"
    output_path = os.path.join(settings.edited_folder, output_name)

    # Ensure output directory exists
    os.makedirs(settings.edited_folder, exist_ok=True)

    try:
        # Parse color if provided
        new_color = None
        if request.color:
            try:
                rgb_parts = [int(c.strip()) for c in request.color.split(',')]
                if len(rgb_parts) != 3:
                    raise ValueError("Color must be 'R,G,B' format")
                new_color = tuple(rgb_parts)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid color format: {str(e)}"
                )

        # Get pattern path if provided
        pattern_path = None
        if request.pattern:
            pattern_path = os.path.join(settings.patterns_folder, request.pattern)
            if not os.path.exists(pattern_path):
                raise HTTPException(status_code=404, detail="Pattern not found")

        # Process image
        _, processing_time = process_image(
            input_path,
            output_path,
            new_color=new_color,
            pattern_path=pattern_path
        )

        return ProcessImageResponse(
            success=True,
            edited_image=output_name,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
