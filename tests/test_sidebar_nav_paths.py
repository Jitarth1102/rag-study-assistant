from rag_assistant.ui import sidebar


def test_nav_links_paths_are_relative_pages():
    for label, path in sidebar.NAV_LINKS:
        assert path.startswith("pages/"), f"{label} path should start with pages/: {path}"
        assert path.endswith(".py"), f"{label} path should end with .py: {path}"
        assert "apps/streamlit" not in path, f"{label} path should not include apps/streamlit: {path}"
