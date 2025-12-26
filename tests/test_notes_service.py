from pathlib import Path

import pytest

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import execute, init_db
from rag_assistant.services import asset_service, notes_quality, notes_service, subject_service
from rag_assistant.services.notes_quality import SectionGap
from rag_assistant.web.search_client import WebResult
from rag_assistant.rag.judge import JudgeDecision


def _setup_subject_and_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "data"
    db_path = data_root / "db" / "test.db"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DB_PATH", str(db_path))
    cfg = load_config()
    init_db(db_path)
    subject = subject_service.create_subject("Test Subject")
    asset = asset_service.add_asset(subject["subject_id"], "sample.pdf", b"file-bytes", "application/pdf")
    execute(
        db_path,
        "INSERT INTO chunks (chunk_id, subject_id, asset_id, page_num, text, bbox_json, start_block, end_block, created_at) VALUES (?, ?, ?, ?, ?, '{}', 0, 0, 0.0);",
        ("chunk1", subject["subject_id"], asset["asset_id"], 1, "Intro to testing"),
    )
    return cfg, subject, asset, db_path


def test_notes_web_enabled_by_default():
    cfg = load_config()
    assert cfg.notes.web_augmentation_enabled is True


def test_generate_notes_creates_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)

    class DummyEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed_texts(self, texts):
            return [[0.1] * 2 for _ in texts]

    store_calls = {"upserts": 0, "payloads": [], "deleted": []}
    calls = {"llm": [], "revise": []}

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            store_calls["deleted"].append(notes_id)

        def upsert_chunks(self, vectors, payloads, ids):
            store_calls["upserts"] += 1
            store_calls["payloads"] = payloads

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())

    def fake_answer(prompt, cfg, **kwargs):
        calls["llm"].append(prompt)
        return "## Heading\nSome bullet point"

    def fake_quality(draft, cfg, trace=None, base_query=None):
        calls["revise"].append(draft)
        return draft + "\n\n## Improvements\n- Added clarity", {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}

    monkeypatch.setattr(notes_service, "generate_answer", fake_answer)
    monkeypatch.setattr(notes_service, "run_quality_loop", fake_quality)

    res = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    notes_row = execute(db_path, "SELECT * FROM notes WHERE notes_id = ?;", (res["notes_id"],), fetchone=True)
    chunks = execute(db_path, "SELECT * FROM notes_chunks WHERE notes_id = ?;", (res["notes_id"],), fetchall=True)

    assert notes_row is not None
    assert notes_row["version"] == 1
    assert chunks
    assert store_calls["upserts"] == 1
    first_payload = store_calls["payloads"][0]
    assert first_payload["source_type"] == "notes"
    assert first_payload["source_label"] == "Generated Notes"
    assert first_payload["version"] == 1
    assert store_calls["deleted"]
    assert len(calls["llm"]) == 1  # draft
    assert calls["revise"]  # critique loop invoked


def test_generate_traces_quality(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.notes.generation.min_chars = 0
    trace: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.21] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")
    monkeypatch.setattr(notes_quality, "generate_answer", lambda prompt, cfg, **kwargs: "## Revised\nBody")

    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert any("draft_generate:start" in m for m in trace)
    assert any("judge_review:start round=1" in m for m in trace)
    assert any("judge_review:start round=2" in m for m in trace)
    assert any("revise:start round=1" in m for m in trace or [])
    assert any("revise:done round=1" in m for m in trace or [])
    draft_idx = trace.index(next(m for m in trace if "draft_generate:start" in m))
    judge1_idx = trace.index(next(m for m in trace if "judge_review:start round=1" in m))
    judge2_idx = trace.index(next(m for m in trace if "judge_review:start round=2" in m))
    assert draft_idx < judge1_idx < judge2_idx


def test_regenerate_pipeline_matches_generate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.notes.generation.min_chars = 0
    trace: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.22] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")
    monkeypatch.setattr(notes_quality, "generate_answer", lambda prompt, cfg, **kwargs: "## Revised\nBody")

    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    draft_starts = [m for m in trace if "draft_generate:start" in m]
    revise_starts = [m for m in trace if "revise:start round=" in m]
    judge_starts = [m for m in trace if "judge_review:start" in m]
    assert len(draft_starts) == 2
    assert len(revise_starts) >= 2  # may include length-expansion pass
    assert len(judge_starts) == 4  # two rounds per generate call


def test_generation_uses_notes_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.notes.generation.temperature = 0.05
    cfg.notes.generation.top_p = 0.7
    cfg.notes.generation.seed = 123
    cfg.notes.generation.max_tokens = 321
    cfg.notes.generation.target_chars = 500
    cfg.notes.generation.min_chars = 0

    class DummyEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed_texts(self, texts):
            return [[0.11] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    calls = []

    def fake_generate(prompt, cfg, **kwargs):
        calls.append(kwargs)
        if "Draft notes" in prompt:
            return "## Revised\nContent"
        return "## Draft\nBody"

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", fake_generate)
    monkeypatch.setattr(notes_quality, "generate_answer", fake_generate)

    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    assert len(calls) >= 3  # draft + judge + critique (+ possible second judge/revise)
    for call_kwargs in calls:
        assert call_kwargs["temperature"] == 0.05
        assert call_kwargs["top_p"] == 0.7
        assert call_kwargs["seed"] == 123
        assert call_kwargs["max_tokens"] == 321
        assert call_kwargs.get("target_chars") in {None, 500}
        assert call_kwargs.get("min_chars") in {None, 0}


def test_length_expansion_only_when_min_positive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.notes.generation.min_chars = 10
    trace: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.12] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    def fake_generate(prompt, cfg, **kwargs):
        # return very short text to trigger expansion
        return "short"

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", fake_generate)
    monkeypatch.setattr(notes_quality, "generate_answer", fake_generate)

    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert any("expand_for_length" in m for m in trace)


def test_notes_web_augmentation_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.notes.web_augmentation_enabled = True
    cfg.notes.max_web_queries_per_notes = 1
    cfg.notes.max_web_results_per_query = 1
    trace: list[str] = []
    prompts = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.5] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")

    def fake_generate_answer(prompt, cfg, **kwargs):
        prompts.append(prompt)
        return "## Revised\nBody"

    def fake_judge(draft_md, config, trace=None, round_num=1):
        if trace is not None:
            trace.append(f"[NOTES] judge_review:start round={round_num}")
        if round_num == 1:
            return {"needs_revision": True, "critique": "missing info", "needs_web": True, "suggested_queries": ["topic"]}
        return {"needs_revision": False, "critique": "", "needs_web": False, "suggested_queries": []}

    search_calls = {"calls": 0, "allow": None, "block": None}

    def fake_search(query, config=None, allowlist=None, blocklist=None, max_results=None):
        search_calls["calls"] += 1
        search_calls["allow"] = allowlist
        search_calls["block"] = blocklist
        return [WebResult(title="t", url="http://example.com", snippet="snippet", source="example")]

    monkeypatch.setattr(notes_quality, "judge_notes", fake_judge)
    monkeypatch.setattr(notes_quality, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(notes_quality.search_client, "search", fake_search)

    res = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert res["chunk_count"] > 0
    assert search_calls["calls"] == 1
    assert any("web_search:start" in m for m in trace)
    assert any("External Context" in p for p in prompts)
    assert any("judge_review:start round=1" in m for m in trace)
    assert any("judge_review:start round=2" in m for m in trace)
    assert any("judge:done decision=search" in m for m in trace)


def test_notes_web_augmentation_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.notes.web_augmentation_enabled = False
    trace: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.51] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")

    def fake_generate_answer(prompt, cfg, **kwargs):
        return "## Revised\nBody"

    def fake_judge(draft_md, config, trace=None, round_num=1):
        return {"needs_revision": True, "critique": "missing info", "needs_web": True, "suggested_queries": ["topic"]}

    search_calls = {"calls": 0}

    def fake_search(query, config=None, allowlist=None, blocklist=None, max_results=None):
        search_calls["calls"] += 1
        return []

    monkeypatch.setattr(notes_quality, "judge_notes", fake_judge)
    monkeypatch.setattr(notes_quality, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(notes_quality.search_client, "search", fake_search)

    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert search_calls["calls"] == 0


def test_notes_web_augmentation_respects_domains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.notes.web_augmentation_enabled = True
    cfg.notes.web_allow_domains = ["example.com"]
    cfg.notes.web_block_domains = ["blocked.com"]
    trace: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.6] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")

    def fake_generate_answer(prompt, cfg, **kwargs):
        return "## Revised\nBody"

    def fake_judge(draft_md, config, trace=None, round_num=1):
        return {"needs_revision": True, "critique": "missing info", "needs_web": True, "suggested_queries": ["topic"]}

    search_calls = {"allow": None, "block": None}

    def fake_search(query, config=None, allowlist=None, blocklist=None, max_results=None):
        search_calls["allow"] = allowlist
        search_calls["block"] = blocklist
        return [WebResult(title="t", url="http://example.com", snippet="s", source="example.com")]

    monkeypatch.setattr(notes_quality, "judge_notes", fake_judge)
    monkeypatch.setattr(notes_quality, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(notes_quality.search_client, "search", fake_search)

    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert search_calls["allow"] == ["example.com"]
    assert search_calls["block"] == ["blocked.com"]


def test_notes_web_telemetry_logged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.notes.web_augmentation_enabled = True
    trace: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.7] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")

    def fake_judge(draft_md, config, trace=None, round_num=1):
        return {"needs_revision": True, "critique": "fine", "needs_web": False, "web_reason": "judge_says_no"}

    monkeypatch.setattr(notes_quality, "judge_notes", fake_judge)
    monkeypatch.setattr(notes_quality, "generate_answer", lambda prompt, cfg, **kwargs: "## Revised\nBody")

    notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert any("judge:done decision=no_search" in m for m in trace)
    assert any("reason=judge_no_web" in m for m in trace)


def test_notes_web_rescue_after_round2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.notes.web_augmentation_enabled = True
    cfg.notes.max_web_queries_per_notes = 1
    trace: list[str] = []
    prompts: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.8] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")

    def fake_generate_answer(prompt, cfg, **kwargs):
        prompts.append(prompt)
        return "## Revised\nBody"

    def fake_judge(draft_md, config, trace=None, round_num=1):
        if round_num == 1:
            if trace is not None:
                trace.append(f"[NOTES] judge_review:start round={round_num}")
            return {"needs_revision": True, "critique": "needs work", "needs_web": False, "suggested_queries": []}
        if trace is not None:
            trace.append(f"[NOTES] judge_review:start round={round_num}")
        return {"needs_revision": True, "critique": "still needs web", "needs_web": False, "suggested_queries": []}

    search_calls = {"calls": 0}

    def fake_search(query, config=None, allowlist=None, blocklist=None, max_results=None):
        search_calls["calls"] += 1
        return [WebResult(title="t", url="http://rescue.com", snippet="snippet", source="rescue.com")]

    monkeypatch.setattr(notes_quality, "judge_notes", fake_judge)
    monkeypatch.setattr(notes_quality, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(notes_quality.search_client, "search", fake_search)

    res = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert res["used_web"] is True
    assert search_calls["calls"] == 1
    assert any("rescue_with_web_context" in m for m in trace)
    assert any("with_web_context=True" in m for m in trace)
    assert any("web_rescue:search:done" in m for m in trace)


def test_section_gap_patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.notes.web_augmentation_enabled = True
    trace: list[str] = []

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.9] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")

    def fake_generate_answer(prompt, cfg, **kwargs):
        return "Rewritten Section with citations [^1]"

    def fake_detect(markdown, slide_context, cfg, trace=None):
        return [
            SectionGap(
                section_title="Draft",
                section_anchor=None,
                gap_type="missing",
                what_to_add="Add example",
                priority=1,
                suggested_queries=["example gap"],
            )
        ]

    def fake_search(query, config=None, allowlist=None, blocklist=None, max_results=None):
        return [WebResult(title="t", url="http://example.com", snippet="snippet", source="example.com")]

    monkeypatch.setattr(notes_quality, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(notes_quality, "detect_section_gaps", fake_detect)
    monkeypatch.setattr(notes_quality.search_client, "search", fake_search)

    res = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg, trace=trace)
    assert res["used_web"] is True
    assert any("patch_section:applied" in m for m in trace)


def test_regenerate_notes_runs_quality_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)

    class DummyEmbedder:
        def __init__(self, *a, **k):
            pass

        def embed_texts(self, texts):
            return [[0.15] * 2 for _ in texts]

    calls = {"revise": 0}

    class DummyStore:
        def __init__(self):
            self.deleted = []
            self.versions = []

        def delete_by_notes_id(self, notes_id):
            self.deleted.append(notes_id)

        def upsert_chunks(self, vectors, payloads, ids):
            if payloads:
                self.versions.append(payloads[0]["version"])

    store = DummyStore()
    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: store)
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Draft\nBody")

    def fake_quality(draft, cfg, trace=None, base_query=None):
        calls["revise"] += 1
        return draft + "\n\nMore detail", {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}

    monkeypatch.setattr(notes_service, "run_quality_loop", fake_quality)

    first = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    second = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)

    assert first["version"] == 1
    assert second["version"] == 2
    assert calls["revise"] == 2  # both first gen and regenerate hit critique pass
    assert 1 in store.versions and 2 in store.versions


def test_update_notes_increments_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.2] * 2 for _ in texts]

    calls = {"revise": 0}

    class DummyStore:
        def __init__(self):
            self.deleted = []
            self.upserts = 0
            self.payloads = []

        def delete_by_notes_id(self, notes_id):
            self.deleted.append(notes_id)

        def upsert_chunks(self, vectors, payloads, ids):
            self.upserts += 1
            self.payloads = payloads

    store = DummyStore()
    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: store)
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## Intro\nDetails")

    def fake_quality(draft, cfg, trace=None, base_query=None):
        calls["revise"] += 1
        return draft, {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None}

    monkeypatch.setattr(notes_service, "run_quality_loop", fake_quality)

    initial = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    updated = notes_service.update_notes(initial["notes_id"], "# Updated\nNew content", config=cfg)

    row = execute(db_path, "SELECT version, markdown FROM notes WHERE notes_id = ?;", (initial["notes_id"],), fetchone=True)
    assert updated["version"] == 2
    assert row["version"] == 2
    assert "Updated" in row["markdown"]
    assert store.deleted  # deletion before re-upsert
    assert store.upserts >= 1
    assert store.payloads[0]["source_label"] == "From User Notes"
    assert store.payloads[0]["version"] == 2
    assert calls["revise"] == 1


def test_web_augmentation_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)
    cfg.web.enabled = True
    cfg.web.max_web_queries_per_question = 1

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.3] * 2 for _ in texts]

    class DummyStore:
        def delete_by_notes_id(self, notes_id):
            pass

        def upsert_chunks(self, vectors, payloads, ids):
            pass

    counter = {"calls": 0}

    def fake_search(query, config=None, allowlist=None, blocklist=None):
        counter["calls"] += 1
        return [WebResult(title=query, url="http://example.com", snippet="snippet", source="example")]

    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: DummyStore())
    monkeypatch.setattr(notes_service, "generate_answer", lambda prompt, cfg, **kwargs: "## With web\ndata")
    monkeypatch.setattr(
        notes_service,
        "run_quality_loop",
        lambda draft, cfg, trace=None, base_query=None: (
            draft,
            {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None},
        ),
    )
    monkeypatch.setattr(notes_service.search_client, "search", fake_search)
    monkeypatch.setattr(
        notes_service.judge,
        "should_search_web",
        lambda *a, **k: JudgeDecision(do_search=True, reason="force", suggested_queries=["q1", "q2"]),
    )

    res = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    assert counter["calls"] == 1  # bounded by max_web_queries_per_question
    assert res["used_web"] in {True, False}
    notes_row = execute(db_path, "SELECT meta_json FROM notes WHERE notes_id = ?;", (res["notes_id"],), fetchone=True)
    assert notes_row is not None


def test_diff_preserves_labels_for_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg, subject, asset, db_path = _setup_subject_and_asset(tmp_path, monkeypatch)

    class DummyEmbedder:
        def embed_texts(self, texts):
            return [[0.4] * 2 for _ in texts]

    class DummyStore:
        def __init__(self):
            self.deleted = []
            self.payloads = []

        def delete_by_notes_id(self, notes_id):
            self.deleted.append(notes_id)

        def upsert_chunks(self, vectors, payloads, ids):
            self.payloads = payloads

    store = DummyStore()
    monkeypatch.setattr(notes_service, "Embedder", lambda *a, **k: DummyEmbedder())
    monkeypatch.setattr(notes_service, "QdrantStore", lambda: store)

    def fake_answer(prompt, cfg, **kwargs):
        return "# Section One\nKeep line\n\n# Section Two\nStay put"

    monkeypatch.setattr(notes_service, "generate_answer", fake_answer)
    monkeypatch.setattr(
        notes_service,
        "run_quality_loop",
        lambda draft, cfg, trace=None, base_query=None: (
            draft,
            {"used_web": False, "web_queries": 0, "web_results": 0, "web_error": None},
        ),
    )

    initial = notes_service.generate_notes_for_asset(subject["subject_id"], asset["asset_id"], config=cfg)
    row_before = execute(db_path, "SELECT meta_json FROM notes WHERE notes_id = ?;", (initial["notes_id"],), fetchone=True)
    assert row_before

    updated_markdown = "# Section One\nEdited line\n\n# Section Two\nStay put"
    notes_service.update_notes(initial["notes_id"], updated_markdown, edited_by="user", config=cfg)
    # payloads from last upsert
    labels = [p["source_label"] for p in store.payloads]
    assert "From User Notes" in labels
    assert "Generated Notes" in labels  # unchanged section retains original provenance
    # ensure old vectors removed before upsert
    assert initial["notes_id"] in store.deleted
