import math

from rag_assistant.rag import answerer


def test_validate_embedding_rejects_wrong_dim():
    try:
        answerer._validate_embedding([0.1, 0.2], expected_dim=3)
        assert False, "should raise"
    except ValueError as exc:
        assert "dim" in str(exc)


def test_validate_embedding_rejects_nan():
    try:
        answerer._validate_embedding([math.nan, 0.1], expected_dim=2)
        assert False, "should raise"
    except ValueError as exc:
        assert "NaN" in str(exc)
