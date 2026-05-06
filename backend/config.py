from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/urbansense"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    
    # Logging
    log_level: str = "INFO"
    
    # Google Earth Engine
    gee_key_path: str = ""
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    model_config = ConfigDict(
        env_file=str(Path(__file__).with_name(".env")),
        env_file_encoding="utf-8"
    )

settings = Settings()
