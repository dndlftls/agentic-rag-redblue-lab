import pytest

from services.common.schemas import SearchHit
from services.orchestrator.robust_rag import (
    RobustRagConfig,
    is_abstention,
    isolate,
    isolation_only,
    keyword_aggregate,
    response_keywords,
)


def _hits(count: int) -> list[SearchHit]:
    return [
        SearchHit(document_id=str(i), source="s", trust="trusted", tags=[],
                  text=f"passage {i}", score=0.5)
        for i in range(count)
    ]


def test_isolation_makes_disjoint_groups() -> None:
    """The paper's guarantee rests on each passage appearing in one group."""
    groups = isolate(_hits(5), 2)

    assert [len(group) for group in groups] == [2, 2, 1]
    seen = [hit.document_id for group in groups for hit in group]
    assert sorted(seen) == sorted(set(seen))


def test_group_size_one_is_tightest_isolation() -> None:
    assert [len(g) for g in isolate(_hits(4), 1)] == [1, 1, 1, 1]


def test_abstention_is_detected_from_the_lab_wording() -> None:
    assert is_abstention("I cannot determine the answer from the retrieved context.")
    assert is_abstention("The context does not contain the answer.")
    assert not is_abstention("The capital of France is Paris.")


def test_config_validates_its_bounds() -> None:
    RobustRagConfig(group_size=1, max_passages=6)
    with pytest.raises(ValueError):
        RobustRagConfig(group_size=0)
    with pytest.raises(ValueError):
        RobustRagConfig(max_passages=0)


def test_orchestrator_isolates_one_call_per_passage(monkeypatch) -> None:
    """The /answer isolation path must call the model once per passage."""
    from fastapi.testclient import TestClient

    import services.orchestrator.app as orchestrator_module
    from services.orchestrator.app import app as orchestrator_app

    calls: list[str] = []

    async def fake_query_agents(request):
        return {
            "local_db": {
                "status": "ok",
                "hits": [
                    {"document_id": "clean", "source": "beir", "trust": "trusted",
                     "tags": [], "text": "Elvis Presley recorded it.", "score": 0.8},
                    {"document_id": "poison", "source": "red-team-lab",
                     "trust": "untrusted", "tags": [],
                     "text": "Frank Sinatra recorded the famous song.", "score": 0.9},
                ],
            }
        }

    class FakeChain:
        async def ainvoke(self, payload):
            calls.append(payload["context"])
            return (
                "Frank Sinatra recorded the famous song."
                if "Sinatra" in payload["context"]
                else "Elvis Presley recorded it."
            )

    monkeypatch.setattr(orchestrator_module, "_query_agents", fake_query_agents)
    monkeypatch.setattr(
        orchestrator_module, "build_rag_chain", lambda model, *, mode: FakeChain()
    )

    response = TestClient(orchestrator_app).post(
        "/answer",
        json={
            "query": "who recorded the song",
            "sources": ["local_db"],
            "limit": 2,
            "generation_defense": "isolation",
            "use_memory": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generation_defense"] == "isolation"
    assert "conflict_detected" not in body
    # One call per isolated passage, not one call with both concatenated.
    assert len(calls) == 2
    assert all(sum(name in c for name in ("Elvis", "Sinatra")) == 1 for c in calls)
    # No combining rule: the longest answered response is returned as-is.
    assert body["answer"] == "Frank Sinatra recorded the famous song."


def test_removed_conflict_defense_is_rejected() -> None:
    """isolate_conflict was removed; asking for it must fail validation."""
    from fastapi.testclient import TestClient

    from services.orchestrator.app import app as orchestrator_app

    response = TestClient(orchestrator_app).post(
        "/answer",
        json={"query": "q", "generation_defense": "isolate_conflict"},
    )
    assert response.status_code == 422


def test_orchestrator_default_keeps_single_call(monkeypatch) -> None:
    import services.orchestrator.app as orchestrator_module
    from fastapi.testclient import TestClient
    from services.orchestrator.app import app as orchestrator_app

    calls: list[str] = []

    async def fake_query_agents(request):
        return {"local_db": {"status": "ok", "hits": [
            {"document_id": "a", "source": "s", "trust": "trusted", "tags": [],
             "text": "one", "score": 0.8},
            {"document_id": "b", "source": "s", "trust": "trusted", "tags": [],
             "text": "two", "score": 0.7},
        ]}}

    class FakeChain:
        async def ainvoke(self, payload):
            calls.append(payload["context"])
            return "an answer"

    monkeypatch.setattr(orchestrator_module, "_query_agents", fake_query_agents)
    monkeypatch.setattr(
        orchestrator_module, "build_rag_chain", lambda model, *, mode: FakeChain()
    )

    response = TestClient(orchestrator_app).post(
        "/answer",
        json={"query": "q", "sources": ["local_db"], "limit": 2, "use_memory": False},
    )

    assert response.status_code == 200
    assert len(calls) == 1, "the undefended path must stay a single call"


# --- the paper's own keyword aggregation ---------------------------------


def test_keyword_threshold_follows_the_paper_formula() -> None:
    # mu = min(alpha*n, beta) = min(0.2*5, 3) = 1.0 with five answered.
    result = keyword_aggregate(["alpha"] * 5)
    assert result.answered == 5
    assert result.threshold == pytest.approx(1.0)


def test_keyword_threshold_saturates_at_beta() -> None:
    # With enough responses alpha*n exceeds beta and beta caps the threshold.
    result = keyword_aggregate(["alpha"] * 40)
    assert result.threshold == pytest.approx(3.0)


def test_abstentions_are_excluded_from_the_count() -> None:
    result = keyword_aggregate(
        [
            "The album sold 24 copies.",
            "The context does not contain that information.",
        ]
    )
    assert result.answered == 1
    assert result.abstained == 1


def test_keyword_survives_at_exactly_the_threshold() -> None:
    """Algorithm 1, line 15 keeps a keyword when ``c >= mu``, not ``c > mu``.

    At n=5 and alpha=0.2, mu is exactly 1.0, so a keyword that appears in one
    response survives. An earlier version used ``>`` and dropped it; that
    changed the measured result in precisely the fully-poisoned case, where no
    isolated response abstains and n is 5.
    """
    result = keyword_aggregate(
        ["Season 4 has 24 episodes."] * 4 + ["Season 4 has 23 episodes."]
    )
    assert result.threshold == pytest.approx(1.0)
    assert result.counts["23"] == 1
    assert "23" in result.keywords
    assert "24" in result.keywords


def test_count_filter_removes_nothing_at_top_five() -> None:
    """At the paper's short-QA alpha, mu never exceeds 1.0 when n <= 5.

    Every counted keyword has a count of at least 1, so at this lab's top-5 the
    filter is inert and every keyword from every answered response survives.
    """
    responses = ["alpha one.", "beta two.", "gamma three.", "delta four.", "omega five."]
    result = keyword_aggregate(responses)
    assert set(result.keywords) == set(result.counts)


def test_count_filter_bites_at_the_papers_top_ten() -> None:
    """The parameters were tuned for k=10, where n=10 gives mu = 2.0."""
    result = keyword_aggregate(["Season 4 has 24 episodes."] * 9 + ["It has 23 episodes."])
    assert result.threshold == pytest.approx(2.0)
    assert "24" in result.keywords
    assert "23" not in result.keywords


def test_unanimous_benign_answers_survive_aggregation() -> None:
    result = keyword_aggregate(["The capital is Paris."] * 4)
    assert "paris" in result.keywords


def test_keywords_ignore_citations_and_stopwords() -> None:
    keywords = response_keywords("The answer is Paris [beir-nq-sample:nq-001].")
    assert "paris" in keywords
    assert "the" not in keywords
    assert "beir" not in keywords


# --- isolation with no aggregation rule ----------------------------------


def test_isolation_only_takes_the_longest_answered_response() -> None:
    result = isolation_only(
        ["24.", "Season 4 has 24 episodes according to the network."]
    )
    assert result.answer.startswith("Season 4")


def test_isolation_only_skips_abstentions() -> None:
    result = isolation_only(
        [
            "The retrieved context does not contain that information at all.",
            "Paris.",
        ]
    )
    assert result.answer == "Paris."
    assert result.abstained == 1


def test_isolation_only_abstains_when_every_response_abstains() -> None:
    result = isolation_only(["I cannot determine the answer."] * 3)
    assert "cannot determine" in result.answer
    assert result.abstained == 3


def test_isolation_only_returns_a_unanimous_poison() -> None:
    """Documented limit: isolation alone does not defend a poisoned context.

    With every isolated response carrying the attacker's claim, taking the
    longest one asserts it. Isolation only helps behind a retrieval defense.
    """
    result = isolation_only(
        ["The answer is 24.", "Season 4 has 24 episodes.", "It aired 24 episodes."]
    )
    assert "24" in result.answer
