"""Citations rendering helper."""

from __future__ import annotations

import streamlit as st


def render_citations(citations):
    if not citations:
        return
    st.markdown("### Citations")
    for citation in citations:
        source = citation.get("source", "unknown")
        page = citation.get("page")
        quote = citation.get("quote", "")
        page_text = f" (page {page})" if page is not None else ""
        st.markdown(f"- **{source}**{page_text}: {quote}")
