"""Chat rendering helpers."""

from __future__ import annotations

import streamlit as st

import os

from rag_assistant.services import chat_service
from rag_assistant.ui import session_state
from rag_assistant.config import load_config


def render_chat(subject_id: str):
    session_state.ensure_state()
    messages = session_state.get_messages()

    cfg = load_config()

    st.sidebar.markdown("### Web fallback (optional)")
    web_enabled_default = cfg.web.enabled
    max_queries_default = max(1, min(3, getattr(cfg.web, "max_web_queries_per_question", 2)))
    force_default = getattr(cfg.web, "force_even_if_rag_strong", False)

    web_enabled = st.sidebar.checkbox("Enable web fallback", value=st.session_state.get("web_enabled_override", web_enabled_default))
    st.session_state["web_enabled_override"] = web_enabled

    max_queries = st.sidebar.slider("Max web queries per question", min_value=1, max_value=3, value=st.session_state.get("web_max_queries_override", max_queries_default))
    st.session_state["web_max_queries_override"] = max_queries

    only_if_insufficient = st.sidebar.checkbox(
        "Only use web if slides are insufficient",
        value=st.session_state.get("web_only_if_insufficient_override", not force_default),
    )
    st.session_state["web_only_if_insufficient_override"] = only_if_insufficient
    st.session_state["web_force_even_if_rag_strong_override"] = not only_if_insufficient

    allowed_text = st.sidebar.text_area("Allowed domains (comma-separated)", value=",".join(st.session_state.get("web_allowed_domains_override", [])))
    blocked_text = st.sidebar.text_area("Blocked domains (comma-separated)", value=",".join(st.session_state.get("web_blocked_domains_override", [])))

    def _parse_domains(text: str):
        return [d.strip().lower() for d in text.split(",") if d.strip()]

    allowed_list = _parse_domains(allowed_text)
    blocked_list = _parse_domains(blocked_text)
    st.session_state["web_allowed_domains_override"] = allowed_list
    st.session_state["web_blocked_domains_override"] = blocked_list

    serp_key_present = bool(os.getenv("WEB_API_KEY") or getattr(cfg.web, "api_key", ""))
    st.sidebar.caption(f"SerpAPI key: {'✅ present' if serp_key_present else '❌ missing'}")
    st.sidebar.caption(f"Web enabled: {'✅' if web_enabled else '❌'}")

    st.subheader("Chat")
    user_input = st.text_input("Ask a question")
    citations = []
    context_expanded = st.session_state.get("last_context_expanded", 0)
    last_debug = st.session_state.get("last_retrieval_debug")
    used_web = st.session_state.get("last_used_web", False)
    if st.button("Send") and user_input:
        session_state.add_message("user", user_input)
        overrides = {
            "web_enabled_override": web_enabled,
            "web_max_queries_override": max_queries,
            "web_allowed_domains_override": allowed_list,
            "web_blocked_domains_override": blocked_list,
            "web_force_even_if_rag_strong_override": not only_if_insufficient,
        }
        response = chat_service.ask(subject_id=subject_id, question=user_input, overrides=overrides)
        citations = response.get("citations", [])
        session_state.add_message("assistant", response["answer"])
        context_expanded = response.get("context_expanded", 0) or 0
        st.session_state["last_context_expanded"] = context_expanded
        st.session_state["last_retrieval_debug"] = response.get("debug")
        st.session_state["last_used_web"] = response.get("used_web", False)
        st.success("Response received")

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        st.markdown(f"**{role.capitalize()}:** {content}")

    if context_expanded:
        st.caption(f"Context expanded: +{context_expanded} neighbor chunks")
    if used_web:
        web_dbg = st.session_state.get("last_retrieval_debug") or {}
        queries_used = web_dbg.get("web_queries_used") or web_dbg.get("web_queries_attempted")
        results_used = web_dbg.get("web_results_used")
        reason = web_dbg.get("judge_reason")
        st.caption(f"🌐 Web used (queries: {queries_used}, results: {results_used}, reason: {reason})")

    if st.checkbox("Show retrieval debug", value=False):
        debug_data = st.session_state.get("last_retrieval_debug")
        if debug_data:
            with st.expander("Retrieval debug"):
                st.json(debug_data)
        else:
            st.info("No retrieval debug data available yet.")

    return citations
