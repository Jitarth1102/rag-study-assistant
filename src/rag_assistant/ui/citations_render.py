"""Citations rendering helper."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_citations(citations):
    if not citations:
        return
    slides = [c for c in citations if c.get("type") == "slide" or "asset_id" in c]
    web = [c for c in citations if c.get("type") == "web"]

    if slides:
        st.markdown("### Citations (Slides)")
        for citation in slides:
            source = citation.get("filename") or citation.get("source") or "unknown"
            page = citation.get("page")
            quote = citation.get("quote", "")
            page_text = f" (page {page})" if page is not None else ""
            st.markdown(f"- **{source}**{page_text}: {quote}")
            image_path = citation.get("image_path")
            if image_path and Path(str(image_path)).exists():
                with st.expander("View page image"):
                    st.image(str(image_path), use_column_width=True)

    if web:
        st.markdown("### Citations (Web)")
        for citation in web:
            title = citation.get("title") or citation.get("url") or "web source"
            quote = citation.get("quote", "")
            url = citation.get("url")
            domain = citation.get("source") or ""
            link = f" [{domain}]({url})" if url else ""
            st.markdown(f"- **{title}**{link}: {quote}")
