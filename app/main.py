"""Main FastAPI application"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import get_settings
from app.core.cache import cache
from app.services import color_db, warmup_model
from app.services.preload import preload_images
from app.api.routes import colors, images, processing
from app.models.schemas import HealthResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("=" * 50)
    print("Starting Wall Paint Visualizer API")
    print("=" * 50)

    # Ensure directories exist
    os.makedirs(settings.upload_folder, exist_ok=True)
    os.makedirs(settings.edited_folder, exist_ok=True)
    os.makedirs(settings.patterns_folder, exist_ok=True)

    # Load color database
    print("\n📚 Loading color database...")
    try:
        color_db.load()
        print(f"✓ Loaded {color_db.get_count()} colors")
    except Exception as e:
        print(f"⚠ Warning: Could not load color database: {e}")

    # Warmup AI model
    print("\n🤖 Loading and warming up AI model...")
    try:
        warmup_model()
        print("✓ Model ready")
    except Exception as e:
        print(f"⚠ Warning: Model warmup failed: {e}")

    # Preload demo images (cache wall masks for instant processing)
    try:
        preload_images()
    except Exception as e:
        print(f"⚠ Warning: Preload failed: {e}")

    # Check Redis connection
    if cache.is_connected():
        print("\n✓ Redis cache connected")
    else:
        print("\n⚠ Redis cache not available (will run without caching)")

    print("\n" + "=" * 50)
    print(f"🚀 Server ready on {settings.host}:{settings.port}")
    print("=" * 50 + "\n")

    yield

    # Shutdown
    print("\nShutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered wall paint visualization API",
    lifespan=lifespan
)

# CORS middleware - allow_credentials=False required when using allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(colors.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(processing.router, prefix="/api")

# Mount static files
if os.path.exists("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")


@app.get("/", tags=["root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Wall Paint Visualizer API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        redis_connected=cache.is_connected(),
        model_loaded=True,
        colors_loaded=color_db.loaded,
        total_colors=color_db.get_count() if color_db.loaded else 0
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
