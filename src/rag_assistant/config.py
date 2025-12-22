"""Configuration loader for RAG Study Assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = Field(default="rag-study-assistant")
    environment: str = Field(default="development")
    data_root: str = Field(default="./data")
    logs_dir: str = Field(default="./logs")


class DatabaseConfig(BaseModel):
    sqlite_path: str = Field(default="./data/db/rag_assistant.db")


class QdrantConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    collection_name: str = Field(default="rag_study")


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")


class Settings(BaseModel):
    app: AppConfig
    database: DatabaseConfig
    qdrant: QdrantConfig
    logging: LoggingConfig


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(data: dict) -> dict:
    overrides = {
        ("app", "environment"): os.getenv("APP_ENV"),
        ("app", "data_root"): os.getenv("DATA_ROOT"),
        ("app", "logs_dir"): os.getenv("LOGS_DIR"),
        ("database", "sqlite_path"): os.getenv("DB_PATH"),
        ("logging", "level"): os.getenv("LOG_LEVEL"),
        ("qdrant", "host"): os.getenv("QDRANT_HOST"),
        ("qdrant", "port"): os.getenv("QDRANT_PORT"),
        ("qdrant", "collection_name"): os.getenv("QDRANT_COLLECTION"),
    }
    for (section, key), value in overrides.items():
        if value is None:
            continue
        if section not in data:
            data[section] = {}
        if key == "port":
            try:
                data[section][key] = int(value)
                continue
            except ValueError:
                # keep original if conversion fails
                pass
        data[section][key] = value
    return data


def load_config(path: Optional[Path] = None) -> Settings:
    """Load configuration from YAML and environment variables."""
    load_dotenv()
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    merged = _apply_env_overrides(raw)
    settings = Settings(**merged)

    # Ensure directories exist
    data_root = Path(settings.app.data_root)
    logs_dir = Path(settings.app.logs_dir)
    db_path = Path(settings.database.sqlite_path)
    for directory in [data_root, logs_dir, db_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = ["Settings", "AppConfig", "DatabaseConfig", "QdrantConfig", "LoggingConfig", "load_config"]
