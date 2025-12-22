import streamlit as st

from rag_assistant.services import asset_service
from rag_assistant.ui import render_sidebar, session_state

st.set_page_config(page_title="Upload", layout="wide")
subject_id = render_sidebar() or session_state.get_selected_subject()

st.title("Upload materials")
st.write("Select files to store under the chosen subject. Files are saved only; no processing yet.")

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
    assets = asset_service.list_assets(subject_id)
    st.subheader("Uploaded assets")
    if not assets:
        st.info("No assets uploaded yet.")
    else:
        for asset in assets:
            st.write(
                f"{asset['original_filename']} — {asset['size_bytes']} bytes — status: {asset['status']} (id: {asset['asset_id']})"
            )
