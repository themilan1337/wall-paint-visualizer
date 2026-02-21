"""Application configuration"""
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    model_config = ConfigDict(
        protected_namespaces=(),
        env_file=".env",
        case_sensitive=False,
    )

    # API Settings
    app_name: str = "Wall Paint Visualizer API"
    app_version: str = "2.0.0"
    debug: bool = False

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Redis Settings
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_enabled: bool = True

    # Cache Settings
    cache_ttl: int = 3600  # 1 hour
    mask_cache_ttl: int = 7200  # 2 hours

    # File Settings
    upload_folder: str = "public/images/upload"
    preload_folder: str = "public/images/preload"
    edited_folder: str = "public/edited"
    patterns_folder: str = "public/patterns"
    colors_file: str = "data/colors.json"
    max_file_size: int = 16 * 1024 * 1024  # 16MB

    # AI Model Settings
    model_name: str = "nvidia/segformer-b4-finetuned-ade-512-512"
    device: str = "cpu"  # or "cuda" if GPU available

    # Preload demo images (gentle on weak servers)
    preload_enabled: bool = True
    preload_max_images: int = 4  # Limit for low-RAM; 0 = no limit


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
