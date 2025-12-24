"""Chat rendering helpers."""

from __future__ import annotations

import streamlit as st

from rag_assistant.services import chat_service
from rag_assistant.ui import session_state


def render_chat(subject_id: str):
    session_state.ensure_state()
    messages = session_state.get_messages()

    st.subheader("Chat")
    user_input = st.text_input("Ask a question")
    citations = []
    context_expanded = st.session_state.get("last_context_expanded", 0)
    last_debug = st.session_state.get("last_retrieval_debug")
    used_web = st.session_state.get("last_used_web", False)
    if st.button("Send") and user_input:
        session_state.add_message("user", user_input)
        response = chat_service.ask(subject_id=subject_id, question=user_input)
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
        st.caption("Used web sources to fill gaps.")

    if st.checkbox("Show retrieval debug", value=False):
        debug_data = st.session_state.get("last_retrieval_debug")
        if debug_data:
            with st.expander("Retrieval debug"):
                st.json(debug_data)
        else:
            st.info("No retrieval debug data available yet.")

    return citations
