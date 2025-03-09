import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class DeepSeekSettings(LLMSettings):
    """DeepSeek-specific settings extending LLMSettings."""

    base_url: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:11434"))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    default_model: str = Field(default="deepseek-r1:14b")
    temperature: float = 0.7
    max_retries: int = 3
    max_tokens: int = 4096

class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 1999
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
