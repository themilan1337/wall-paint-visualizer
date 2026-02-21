"""Image upload routes"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.models.schemas import UploadResponse
from app.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/images", tags=["images"])

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file

    - **file**: Image file (PNG, JPG, JPEG, WEBP)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: PNG, JPG, JPEG, WEBP")

    try:
        # Generate unique filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"upload_{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(settings.upload_folder, filename)

        # Save file
        os.makedirs(settings.upload_folder, exist_ok=True)
        with open(filepath, "wb") as f:
            content = await file.read()
            if len(content) > settings.max_file_size:
                raise HTTPException(status_code=413, detail="File too large (max 16MB)")
            f.write(content)

        return UploadResponse(success=True, filename=filename)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/list")
async def list_images():
    """List all uploaded images"""
    try:
        if not os.path.exists(settings.upload_folder):
            return []

        images = [
            f for f in os.listdir(settings.upload_folder)
            if allowed_file(f)
        ]
        images.sort()
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preload/list")
async def list_preload_images():
    """
    List preloaded demo images.
    These images have wall masks cached at startup for instant color changes.
    Use image path as "preload/{filename}" in process API.
    """
    try:
        preload_dir = settings.preload_folder
        if not os.path.isdir(preload_dir):
            return []

        images = [
            f for f in os.listdir(preload_dir)
            if allowed_file(f)
        ]
        images.sort()
        return [{"filename": f, "path": f"preload/{f}"} for f in images]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{filename}")
async def get_image(filename: str):
    """
    Get uploaded image file

    - **filename**: Image filename
    """
    if os.path.isabs(filename) or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    upload_dir = os.path.realpath(settings.upload_folder)
    filepath = os.path.realpath(os.path.join(upload_dir, filename))
    
    if not filepath.startswith(upload_dir):
        raise HTTPException(status_code=400, detail="Invalid path")
        
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(filepath)


@router.get("/edited/{filename}")
async def get_edited_image(filename: str):
    """
    Get edited/processed image file

    - **filename**: Edited image filename
    """
    if os.path.isabs(filename) or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    edited_dir = os.path.realpath(settings.edited_folder)
    filepath = os.path.realpath(os.path.join(edited_dir, filename))
    
    if not filepath.startswith(edited_dir):
        raise HTTPException(status_code=400, detail="Invalid path")
        
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Edited image not found")

    return FileResponse(filepath)
