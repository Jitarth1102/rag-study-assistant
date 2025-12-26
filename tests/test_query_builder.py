from rag_assistant.web.query_builder import build_web_queries, sanitize_text


def test_sanitize_text_strips_markdown_and_filenames():
    text = "**Weak** intro to models.pdf with $math$ and # headers"
    cleaned = sanitize_text(text)
    assert "Weak" not in cleaned
    assert "pdf" not in cleaned
    assert "$" not in cleaned
    assert "#" not in cleaned


def test_build_web_queries_limits_words_and_sanitizes():
    queries = build_web_queries(
        subject="Probabilistic Models",
        asset_title="lecture01.pdf",
        intents=["Generative vs discriminative models * definitions"],
        question=None,
        max_queries=3,
    )
    assert queries
    assert all(len(q.split()) <= 12 for q in queries)
    assert not any("pdf" in q for q in queries)
    assert not any("*" in q for q in queries)
