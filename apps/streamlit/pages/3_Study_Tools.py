import streamlit as st

from rag_assistant.ui import render_sidebar, session_state

st.set_page_config(page_title="Study Tools", layout="wide")
subject_id = render_sidebar() or session_state.get_selected_subject()

st.title("Study Tools")
if not subject_id:
    st.warning("Please create and select a subject on Home.")
else:
    st.write("Placeholder study tools for subject: " + subject_id)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate Summary"):
            st.info("Summary generation not implemented yet.")
    with col2:
        if st.button("Create Flashcards"):
            st.info("Flashcard creation not implemented yet.")
    with col3:
        if st.button("Schedule Quiz"):
            st.info("Quiz scheduling not implemented yet.")
