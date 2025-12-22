import streamlit as st

from rag_assistant.ui import render_sidebar, render_chat, render_citations, session_state

st.set_page_config(page_title="Chat", layout="wide")

subject_id = render_sidebar() or session_state.get_selected_subject()

st.title("Chat")
if not subject_id:
    st.warning("Please create and select a subject on Home.")
else:
    citations = render_chat(subject_id)
    render_citations(citations)
