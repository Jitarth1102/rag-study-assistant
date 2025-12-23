import shutil
import sys

import streamlit as st

from rag_assistant.ingest.pipeline import process_subject_new_assets
from rag_assistant.ingest.ocr.selftest import run_ocr_selftest
from rag_assistant.config import load_config
from rag_assistant.services import asset_service
from rag_assistant.services import cleanup_service
from rag_assistant.ui import render_sidebar, session_state

st.set_page_config(page_title="Upload", layout="wide")
subject_id = render_sidebar() or session_state.get_selected_subject()
cfg = load_config()

st.title("Upload materials")
st.write("Select files to store under the chosen subject. Files are saved locally and indexed for notes-only chat.")

with st.expander("OCR debug info"):
    st.write(f"SQLite path: {cfg.database.sqlite_path}")
    st.write(f"OCR engine: {cfg.ingest.ocr_engine}")
    st.write(f"tesseract_cmd (config): {cfg.ingest.tesseract_cmd or '(auto)'}")
    st.write(f"which tesseract: {shutil.which('tesseract')}")
    from rag_assistant.ingest.ocr.resolve import resolve_tesseract

    cmd, tessdata_dir, lang = resolve_tesseract(cfg)
    st.write(f"resolved tesseract_cmd: {cmd}")
    st.write(f"resolved tessdata_dir: {tessdata_dir}")
    st.write(f"resolved lang: {lang}")
    st.write(f"python: {sys.executable}")
    if st.button("Run OCR self-test"):
        res = run_ocr_selftest(cfg)
        st.write(res)

if not subject_id:
    st.warning("Please create and select a subject on Home.")
else:
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if uploaded_files and st.button("Save files"):
        saved_assets = []
        for uploaded in uploaded_files:
            file_bytes = uploaded.getvalue()
            asset = asset_service.add_asset(subject_id, uploaded.name, file_bytes, uploaded.type)
            saved_assets.append(asset)
        st.success(f"Saved {len(saved_assets)} file(s).")

    if st.button("Index new uploads"):
        with st.spinner("Indexing..."):
            try:
                result = process_subject_new_assets(subject_id, None)
                st.success(
                    f"Indexed: {result.get('indexed', 0)} | Missing skipped: {result.get('skipped_missing', 0)} | Failed: {result.get('failed', 0)}"
                )
                if result.get("failed"):
                    st.warning("Some assets failed; see details below.")
            except Exception as exc:
                st.error(f"Indexing failed: {type(exc).__name__}: {exc}")

    assets = asset_service.list_assets(subject_id)
    st.subheader("Uploaded assets")
    if not assets:
        st.info("No assets uploaded yet.")
    else:
        for asset in assets:
            status = asset.get("status")
            status_row = asset_service.get_index_status(asset["asset_id"]) or {}
            stage = status_row.get("stage", status)
            badge = f"status: {stage}"
            ocr_engine_used = status_row.get("ocr_engine")
            if stage == "missing":
                st.warning(
                    f"⚠️ {asset['original_filename']} — file missing on disk (id: {asset['asset_id']}). {badge}",
                    icon="⚠️",
                )
                if status_row.get("error"):
                    st.caption(status_row["error"])
            else:
                st.write(
                    f"{asset['original_filename']} — {asset['size_bytes']} bytes — {badge} (id: {asset['asset_id']})"
                )
                if ocr_engine_used:
                    st.caption(f"OCR engine: {ocr_engine_used}")
                if status_row.get("error"):
                    msg = status_row["error"]
                    if "OCR engine used" in msg or "fallback OCR" in msg:
                        st.info(msg)
                    else:
                        st.error(f"Error: {msg}")

    missing_assets = cleanup_service.list_missing_assets(subject_id)
    if missing_assets:
        st.subheader("Missing assets")
        st.info("The following assets are missing on disk. Remove them to clean up the registry.")
        for m in missing_assets:
            st.write(f"{m['original_filename']} (id: {m['asset_id']}) — missing path: {m['stored_path']}")
        if st.button("Remove missing assets from registry"):
            removed = cleanup_service.remove_assets(subject_id, [m["asset_id"] for m in missing_assets], remove_vectors=False)
            st.success(f"Removed {len(removed.get('deleted', []))} assets.")
