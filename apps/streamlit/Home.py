from pathlib import Path

import streamlit as st

from rag_assistant.config import load_config
from rag_assistant.db.sqlite import init_db
from rag_assistant.services import subject_service
from rag_assistant.ui import render_sidebar, session_state

st.set_page_config(page_title="RAG Study Assistant", layout="wide")
config = load_config()
init_db(Path(config.database.sqlite_path))

st.title("RAG Study Assistant")
st.write("Use subjects to organize study materials, uploads, and chat context.")

sidebar_subject = render_sidebar()

st.subheader("Create a new subject")
with st.form(key="create_subject_form"):
    new_subject_name = st.text_input("Subject name", placeholder="Enter Subject name here")
    submitted = st.form_submit_button("Create subject")
    if submitted:
        try:
            subject = subject_service.create_subject(new_subject_name)
            session_state.set_selected_subject(subject["subject_id"])
            st.success(f"Created subject '{subject['name']}' ({subject['subject_id']})")
        except ValueError as exc:
            st.error(str(exc))

st.subheader("Choose subject")
subjects = subject_service.list_subjects()
if subjects:
    options = {f"{s['name']} ({s['subject_id']})": s["subject_id"] for s in subjects}
    current = session_state.get_selected_subject() or sidebar_subject or list(options.values())[0]
    current_index = list(options.values()).index(current) if current in options.values() else 0
    selected_label = st.selectbox("Active subject", list(options.keys()), index=current_index)
    selected_subject = options[selected_label]
    session_state.set_selected_subject(selected_subject)
    st.info(f"Active subject: {selected_subject}")
else:
    st.info("No subjects yet. Create one to get started.")

st.markdown(
    """
### Pages
- Chat: ask questions
- Upload: add source files
- Study Tools: placeholder utilities
"""
)
