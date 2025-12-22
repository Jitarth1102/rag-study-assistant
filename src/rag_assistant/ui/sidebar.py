"""Sidebar layout."""

from __future__ import annotations

import streamlit as st

from rag_assistant.services.subject_service import list_subjects
from rag_assistant.ui import session_state

# Navigation targets relative to the Streamlit entrypoint
NAV_LINKS = [
    ("Chat", "pages/1_Chat.py"),
    ("Upload", "pages/2_Upload.py"),
    ("Study Tools", "pages/3_Study_Tools.py"),
]


def _is_valid_page_path(path: str) -> bool:
    return path.startswith("pages/") and path.endswith(".py")


def _safe_page_link(path: str, label: str) -> None:
    if not _is_valid_page_path(path):
        st.sidebar.warning(f"Invalid page path for '{label}': {path}")
        return
    st.sidebar.page_link(path, label=label)


def render_sidebar():
    st.sidebar.title("RAG Study Assistant")
    subjects = list_subjects()
    if not subjects:
        st.sidebar.info("No subjects yet. Create one on Home.")
        for label, path in NAV_LINKS:
            _safe_page_link(path, label=label)
        return None

    options = {f"{s['name']} ({s['subject_id']})": s["subject_id"] for s in subjects}
    current = session_state.get_selected_subject()
    default_index = 0
    if current:
        keys = list(options.keys())
        try:
            default_index = keys.index(next(k for k, v in options.items() if v == current))
        except StopIteration:
            default_index = 0

    selected_label = st.sidebar.selectbox("Choose subject", list(options.keys()), index=default_index)
    subject_id = options[selected_label]
    session_state.set_selected_subject(subject_id)
    for label, path in NAV_LINKS:
        _safe_page_link(path, label=label)
    return subject_id
