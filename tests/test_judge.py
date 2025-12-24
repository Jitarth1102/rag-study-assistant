from rag_assistant.rag import judge


class DummyCfg:
    class Web:
        enabled = True
        min_rag_hits_to_skip_web = 3
        min_rag_score_to_skip_web = 0.65

    web = Web()


def test_judge_skips_when_confident():
    rag_hits = [{"score": 0.9}]
    debug = {"hit_count_after_filter": 3}
    decision = judge.should_search_web("What is ML?", rag_hits, debug, config=DummyCfg())
    assert decision.do_search is False
    assert decision.reason == "rag_confident"


def test_judge_uses_web_when_no_hits():
    decision = judge.should_search_web("Define gradient", [], {"hit_count_after_filter": 0}, config=DummyCfg())
    assert decision.do_search is True
    assert decision.reason in {"no_hits", "definition_with_weak_rag"}
