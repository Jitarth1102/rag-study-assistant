from types import SimpleNamespace

from rag_assistant.ui import citations_render


class DummyExpander:
    def __init__(self):
        self.images = []

    def image(self, *args, **kwargs):
        self.images.append(args)


class DummySt:
    def __init__(self):
        self.markdowns = []
        self.expanders = []

    def markdown(self, text):
        self.markdowns.append(text)

    def expander(self, *args, **kwargs):
        exp = DummyExpander()
        self.expanders.append(exp)
        return exp


def test_render_citations_splits_web_and_slides(monkeypatch):
    dummy = DummySt()
    monkeypatch.setattr(citations_render, "st", dummy)
    monkeypatch.setattr(citations_render, "Path", lambda p: SimpleNamespace(exists=lambda: False))

    citations = [
        {"type": "slide", "filename": "slide.pdf", "page": 1, "quote": "slide quote"},
        {"type": "web", "title": "Web", "url": "http://example.com", "quote": "web quote", "source": "example.com"},
    ]
    citations_render.render_citations(citations)
    assert any("Citations (Slides)" in m for m in dummy.markdowns)
    assert any("Citations (Web)" in m for m in dummy.markdowns)
