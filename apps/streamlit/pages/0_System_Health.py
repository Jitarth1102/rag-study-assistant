import streamlit as st

from rag_assistant.config import load_config
from rag_assistant.ingest.ocr.resolve import resolve_tesseract
from rag_assistant.services import health_service


st.set_page_config(page_title="System Health", layout="wide")

try:
    cfg = load_config()
except Exception as exc:  # pragma: no cover - UI guard
    st.error(f"Failed to load config: {exc}")
    st.stop()


def render_check(name: str, result: dict) -> None:
    ok = result.get("ok", False)
    status_fn = st.success if ok else st.error
    status_fn(f"{name}: {'OK' if ok else 'Issue detected'}")
    with st.expander(f"Details: {name}", expanded=not ok):
        st.json(result)


st.title("System Health")

tess_cmd, tessdata_dir, tess_lang = resolve_tesseract(cfg)

st.subheader("App config summary")
st.json(
    {
        "database_path": cfg.database.sqlite_path,
        "data_root": cfg.app.data_root,
        "qdrant": {"url": cfg.qdrant.url, "collection": cfg.qdrant.collection, "vector_size": cfg.qdrant.vector_size},
        "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model, "base_url": cfg.llm.base_url},
        "embeddings": {"provider": cfg.embeddings.provider, "model": cfg.embeddings.model, "vector_size": cfg.embeddings.vector_size},
        "ocr": {
            "engine": cfg.ingest.ocr_engine,
            "tesseract_cmd_resolved": tess_cmd,
            "tessdata_dir_resolved": tessdata_dir,
            "lang": tess_lang,
        },
    }
)

col1, col2 = st.columns(2)
with col1:
    refresh = st.button("Refresh checks")
with col2:
    run_ocr = st.button("Run OCR self-test")

if "health_checks" not in st.session_state or refresh:
    st.session_state["health_checks"] = health_service.run_all_checks(cfg, include_ocr=False)

if run_ocr:
    st.session_state["health_checks"]["ocr"] = health_service.run_ocr_check(cfg)

checks = st.session_state["health_checks"]

st.subheader("Checks")
render_check("SQLite", checks.get("db", {}))
render_check("Qdrant", checks.get("qdrant", {}))
render_check("Ollama", checks.get("ollama", {}))
if "ocr" in checks:
    render_check("OCR self-test", checks.get("ocr", {}))
else:
    st.info("OCR self-test not run yet. Click 'Run OCR self-test' to execute.")
