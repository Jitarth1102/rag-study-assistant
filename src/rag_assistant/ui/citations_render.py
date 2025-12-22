"""Citations rendering helper."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_citations(citations):
    if not citations:
        return
    st.markdown("### Citations")
    for citation in citations:
        source = citation.get("filename") or citation.get("source") or "unknown"
        page = citation.get("page")
        quote = citation.get("quote", "")
        page_text = f" (page {page})" if page is not None else ""
        st.markdown(f"- **{source}**{page_text}: {quote}")
        image_path = citation.get("image_path")
        if image_path and Path(str(image_path)).exists():
            with st.expander("View page image"):
                st.image(str(image_path), use_column_width=True)
