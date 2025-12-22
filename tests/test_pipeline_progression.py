from pathlib import Path

import cv2
import numpy as np
import pytest

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import init_db
from rag_assistant.ingest import pipeline
from rag_assistant.services import asset_service, subject_service


def test_pipeline_progression(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_root = tmp_path / "data"
    db_path = data_root / "db" / "test.db"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DB_PATH", str(db_path))
    cfg = load_config()
    init_db(db_path)
    subject = subject_service.create_subject("Test Subject")

    img_path = data_root / "sample.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    asset = asset_service.add_asset(subject["subject_id"], "sample.png", img_path.read_bytes(), "image/png")

    class DummyOCR:
        def __init__(self, lang: str = "en"):
            pass

        def ocr_page(self, image_path: str, page_num: int):
            return {"page": page_num, "width": 10, "height": 10, "blocks": [{"text": "hello", "bbox": [0, 0, 5, 5], "confidence": 0.9}]}

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def embed_texts(self, texts):
            return [[0.0] * cfg.qdrant.vector_size for _ in texts]

    class DummyStore:
        def __init__(self):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            return None

    monkeypatch.setattr(pipeline, "get_ocr_engine", lambda lang="en": (DummyOCR(), None))
    monkeypatch.setattr(pipeline, "Embedder", DummyEmbedder)
    monkeypatch.setattr(pipeline, "QdrantStore", DummyStore)

    pipeline.process_asset(subject["subject_id"], asset, cfg)

    status = asset_service.get_index_status(asset["asset_id"])
    assert status["stage"] == "indexed"
