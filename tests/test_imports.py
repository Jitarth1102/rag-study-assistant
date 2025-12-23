def test_imports():
    import rag_assistant
    import rag_assistant.cli
    import rag_assistant.config
    import rag_assistant.logging
    import rag_assistant.ui
    import rag_assistant.services.chat_service
    import rag_assistant.services.subject_service
    import rag_assistant.services.asset_service
    import rag_assistant.db.sqlite
    import rag_assistant.retrieval.vector_store.qdrant
    import rag_assistant.ingest.pipeline
    import rag_assistant.ingest.render.pdf_to_images
    import rag_assistant.ingest.chunking.layout_chunker
    import rag_assistant.ingest.ocr.paddle
    import rag_assistant.rag.answerer
    import rag_assistant.services.cleanup_service

    assert rag_assistant is not None
