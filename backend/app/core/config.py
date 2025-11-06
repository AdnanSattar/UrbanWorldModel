"""
Configuration management for UrbanSim WM Backend
Uses pydantic-settings for environment variable validation
"""

import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"

    # Model Configuration
    MODEL_CHECKPOINT_PATH: str = "./checkpoints/model_best.json"
    # Warm model on startup to reduce first-request latency ("true"/"false")
    WARM_MODEL_ON_STARTUP: str = "true"

    # ETL bootstrap on API startup ("true"/"false")
    RUN_ETL_ON_STARTUP: str = "true"
    # Comma-separated city list for ETL bootstrap
    ETL_CITIES: str = "Lahore,Karachi,Islamabad,Peshawar,Quetta"
    # Lookback hours for OpenAQ bootstrap (reduced default for faster runs)
    ETL_OPENAQ_HOURS: int = 6
    # Max sensors to query per city (limits API calls; 0 = no limit)
    ETL_OPENAQ_MAX_SENSORS: int = 50
    # Max seconds to wait on ETL during startup before continuing
    ETL_STARTUP_TIMEOUT_SECS: int = 10
    # Request timeout for OpenAQ calls (seconds)
    ETL_REQUEST_TIMEOUT_SECS: int = 10

    # Paths (container paths; bind-mounted via docker-compose)
    LOGS_DIR: str = "/logs"
    PROCESSED_DATA_DIR: str = "/etl/processed_data"

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://frontend:3000"]

    # External API Keys
    WAQI_API_TOKEN: str = ""  # World Air Quality Index (primary)
    OPENAQ_API_KEY: str = ""  # OpenAQ (fallback)
    MOBILITY_API_KEY: str = ""
    ENERGY_API_KEY: str = ""

    # Redis Configuration (for future use with Celery)
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # Simulation Defaults
    DEFAULT_HORIZON_HOURS: int = 48
    DEFAULT_CITY: str = "Lahore"
    # Simulation mode: "model" (uses world model inference) or "baseline" (uses simple deterministic dynamics)
    SIMULATION_MODE: str = "model"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton settings instance
settings = Settings()
