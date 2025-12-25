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
    url: str = Field(default="http://localhost:6333")
    collection: str = Field(default="rag_chunks")
    vector_size: int = Field(default=1536)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")


class IngestConfig(BaseModel):
    pdf_dpi: int = Field(default=200)
    ocr_lang: str = Field(default="en")
    max_chunk_chars: int = Field(default=2000)
    min_chunk_chars: int = Field(default=200)
    overlap_blocks: int = Field(default=2)
    ocr_engine: str = Field(default="auto")
    tesseract_cmd: str = Field(default="")
    tessdata_dir: str = Field(default="")


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=6)
    neighbor_window: int = Field(default=1)
    max_neighbor_chunks: int = Field(default=12)
    min_score: float = Field(default=0.0)


class LLMConfig(BaseModel):
    provider: str = Field(default="ollama")
    model: str = Field(default="llama3.1:8b")
    base_url: str = Field(default="http://127.0.0.1:11434")
    temperature: float = Field(default=0.2)
    timeout_s: int = Field(default=60)
    chat_model: str = Field(default="gpt-4.1-mini")
    embed_model: str = Field(default="text-embedding-3-small")


class EmbeddingsConfig(BaseModel):
    provider: str = Field(default="local")
    model: str = Field(default="intfloat/multilingual-e5-small")
    vector_size: int = Field(default=384)


class WebConfig(BaseModel):
    enabled: bool = Field(default=False)
    provider: str = Field(default="serpapi")
    api_key: str = Field(default="")
    max_results: int = Field(default=5)
    timeout_s: int = Field(default=20)
    min_rag_score_to_skip_web: float = Field(default=0.65)
    min_rag_hits_to_skip_web: int = Field(default=3)
    max_web_queries_per_question: int = Field(default=2)
    force_even_if_rag_strong: bool = Field(default=False)
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    app: AppConfig
    database: DatabaseConfig
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    web: WebConfig = Field(default_factory=WebConfig)


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
        ("qdrant", "url"): os.getenv("QDRANT_URL"),
        ("qdrant", "vector_size"): os.getenv("QDRANT_VECTOR_SIZE"),
        ("retrieval", "top_k"): os.getenv("RETRIEVAL_TOP_K"),
        ("llm", "provider"): os.getenv("LLM_PROVIDER"),
        ("llm", "model"): os.getenv("LLM_MODEL"),
        ("llm", "base_url"): os.getenv("LLM_BASE_URL"),
        ("llm", "temperature"): os.getenv("LLM_TEMPERATURE"),
        ("llm", "timeout_s"): os.getenv("LLM_TIMEOUT_S"),
        ("embeddings", "provider"): os.getenv("EMBEDDINGS_PROVIDER"),
        ("embeddings", "model"): os.getenv("EMBEDDINGS_MODEL"),
        ("embeddings", "vector_size"): os.getenv("EMBEDDINGS_VECTOR_SIZE"),
        ("ingest", "ocr_engine"): os.getenv("OCR_ENGINE"),
        ("ingest", "tesseract_cmd"): os.getenv("TESSERACT_CMD"),
        ("ingest", "tessdata_dir"): os.getenv("TESSDATA_DIR"),
        ("retrieval", "neighbor_window"): os.getenv("RETRIEVAL_NEIGHBOR_WINDOW"),
        ("retrieval", "max_neighbor_chunks"): os.getenv("RETRIEVAL_MAX_NEIGHBOR_CHUNKS"),
        ("retrieval", "min_score"): os.getenv("RETRIEVAL_MIN_SCORE"),
        ("web", "enabled"): os.getenv("WEB_ENABLED"),
        ("web", "provider"): os.getenv("WEB_PROVIDER"),
        ("web", "api_key"): os.getenv("WEB_API_KEY"),
        ("web", "max_results"): os.getenv("WEB_MAX_RESULTS"),
        ("web", "timeout_s"): os.getenv("WEB_TIMEOUT_S"),
        ("web", "min_rag_score_to_skip_web"): os.getenv("WEB_MIN_RAG_SCORE_TO_SKIP_WEB"),
        ("web", "min_rag_hits_to_skip_web"): os.getenv("WEB_MIN_RAG_HITS_TO_SKIP_WEB"),
        ("web", "max_web_queries_per_question"): os.getenv("WEB_MAX_WEB_QUERIES_PER_QUESTION"),
        ("web", "force_even_if_rag_strong"): os.getenv("WEB_FORCE_EVEN_IF_RAG_STRONG"),
    }
    for (section, key), value in overrides.items():
        if value is None:
            continue
        if section not in data:
            data[section] = {}
        if key in {"enabled"}:
            data[section][key] = str(value).strip().lower() in {"1", "true", "yes", "on"}
            continue
        if key in {"port", "vector_size", "top_k", "neighbor_window", "max_neighbor_chunks", "max_results", "timeout_s", "min_rag_hits_to_skip_web", "max_web_queries_per_question"}:
            try:
                data[section][key] = int(value)
                continue
            except ValueError:
                # keep original if conversion fails
                pass
        if key in {"temperature", "timeout_s", "min_score", "min_rag_score_to_skip_web"}:
            try:
                if key in {"temperature"}:
                    data[section][key] = float(value)
                elif key == "timeout_s":
                    data[section][key] = int(value)
                else:
                    data[section][key] = float(value)
                continue
            except ValueError:
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


__all__ = [
    "Settings",
    "AppConfig",
    "DatabaseConfig",
    "QdrantConfig",
    "LoggingConfig",
    "IngestConfig",
    "RetrievalConfig",
    "LLMConfig",
    "load_config",
    "WebConfig",
]
