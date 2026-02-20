"""Application configuration"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""

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
    upload_folder: str = "public/images"
    edited_folder: str = "public/edited"
    patterns_folder: str = "public/patterns"
    colors_file: str = "data/colors.json"
    max_file_size: int = 16 * 1024 * 1024  # 16MB

    # AI Model Settings
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    device: str = "cpu"  # or "cuda" if GPU available

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
