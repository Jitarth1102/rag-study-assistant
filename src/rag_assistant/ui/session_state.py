"""Helpers for Streamlit session state."""

from __future__ import annotations

import streamlit as st


def get_messages_key() -> str:
    return "chat_messages"

SELECTED_SUBJECT_KEY = "selected_subject_id"


def ensure_state() -> None:
    key = get_messages_key()
    if key not in st.session_state:
        st.session_state[key] = []


def add_message(role: str, content: str) -> None:
    ensure_state()
    st.session_state[get_messages_key()].append({"role": role, "content": content})


def get_messages():
    ensure_state()
    return st.session_state[get_messages_key()]


def get_selected_subject() -> str | None:
    return st.session_state.get(SELECTED_SUBJECT_KEY)


def set_selected_subject(subject_id: str) -> None:
    st.session_state[SELECTED_SUBJECT_KEY] = subject_id
