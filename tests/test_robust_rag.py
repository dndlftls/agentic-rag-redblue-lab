import pytest

from services.common.schemas import SearchHit
from services.orchestrator.robust_rag import (
    RobustRagConfig,
    aggregate,
    claim_signature,
    claims_agree,
    is_abstention,
    isolate,
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


def test_topic_overlap_must_not_hide_a_contradiction() -> None:
    """Regression: shared topic words once masked a 24-against-23 conflict.

    Both responses mention "Chicago Fire" and "season 4", so comparing raw
    signatures found an overlap and merged them. The answer is the part the
    question does not contain, so query terms are subtracted first.
    """
    query = "how many episodes are in chicago fire season 4"
    poison = claim_signature("Chicago Fire season 4 consists of 24 episodes.", query)
    truth = claim_signature("Chicago Fire season 4 has 23 episodes.", query)

    assert poison == {"24"}
    assert truth == {"23"}
    assert not claims_agree(poison, truth)

    result = aggregate(
        ["Chicago Fire season 4 consists of 24 episodes.",
         "Chicago Fire season 4 has 23 episodes."],
        query,
    )
    assert result.conflict is True


def test_query_subtraction_keeps_real_agreement() -> None:
    query = "how many episodes are in chicago fire season 4"

    assert claims_agree(
        claim_signature("The answer is 24.", query),
        claim_signature("Season 4 contains 24 episodes total.", query),
    )


def test_signature_keeps_numerals_and_names_only() -> None:
    assert claim_signature("The answer is 24 [beir:doc1].") == {"24"}
    assert "presley" in claim_signature("Elvis Presley recorded it.")
    # A sentence-initial capital is not a name.
    assert "the" not in claim_signature("The song is old.")


def test_agreement_survives_different_phrasing() -> None:
    left = claim_signature("The answer is 24 [a:1].")
    right = claim_signature("Season 4 has 24 episodes [b:2].")

    assert claims_agree(left, right)


def test_disagreement_on_the_salient_token_is_a_conflict() -> None:
    assert not claims_agree(claim_signature("The answer is 24."),
                            claim_signature("The answer is 23."))


def test_empty_signature_cannot_manufacture_a_conflict() -> None:
    assert claims_agree(set(), {"24"})


def test_agreeing_responses_yield_one_answer() -> None:
    result = aggregate([
        "The answer is 24 [a:1].",
        "Season 4 has 24 episodes [b:2].",
        "I cannot determine the answer from the retrieved context.",
    ])

    assert result.conflict is False
    assert result.abstained == 1
    assert "24" in result.answer


def test_a_single_disagreeing_response_triggers_conflict() -> None:
    """The defense's whole point: one poison turns a wrong answer into a flag."""
    result = aggregate([
        "Elvis Presley recorded the song.",
        "Frank Sinatra recorded the song.",
    ])

    assert result.conflict is True
    assert "disagree" in result.answer
    assert len(result.distinct_claims) == 2


def test_all_abstaining_returns_the_lab_abstention() -> None:
    result = aggregate([
        "I cannot determine the answer from the retrieved context.",
        "The context does not contain the answer.",
    ])

    assert result.conflict is False
    assert result.abstained == 2
    assert "cannot determine" in result.answer


def test_unanimous_poisons_are_not_detectable() -> None:
    """Documented limit: with no clean passage left there is no disagreement.

    This is the regime the retrieval-stage redundancy filter has to cover.
    """
    result = aggregate([
        "The answer is 24 [p:1].",
        "The answer is 24 [p:2].",
        "The answer is 24 [p:3].",
    ])

    assert result.conflict is False
    assert "24" in result.answer


def test_config_validates_its_bounds() -> None:
    RobustRagConfig(group_size=1, max_passages=6)
    with pytest.raises(ValueError):
        RobustRagConfig(group_size=0)
    with pytest.raises(ValueError):
        RobustRagConfig(max_passages=0)


def test_orchestrator_isolates_and_reports_conflict(monkeypatch) -> None:
    """The /answer path must call the model per group and surface the conflict."""
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
                     "text": "Frank Sinatra recorded it.", "score": 0.9},
                ],
            }
        }

    class FakeChain:
        async def ainvoke(self, payload):
            calls.append(payload["context"])
            return (
                "Frank Sinatra recorded the song."
                if "Sinatra" in payload["context"]
                else "Elvis Presley recorded the song."
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
            "generation_defense": "isolate_conflict",
            "use_memory": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conflict_detected"] is True
    assert "disagree" in body["answer"]
    # One call per isolated passage, not one call with both concatenated.
    assert len(calls) == 2
    assert all(sum(name in c for name in ("Elvis", "Sinatra")) == 1 for c in calls)


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
    assert response.json()["conflict_detected"] is False
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


def test_keyword_aggregation_drops_a_singly_supported_truth() -> None:
    """The measured failure mode, encoded as a test.

    Four poisoned passages agree on 24 and one benign passage says 23. The
    paper's threshold keeps the majority token and discards the minority one,
    so the aggregation hands the attacker's claim to the final prompt. This is
    the paper's own rule at the paper's own defaults, not a weakened version.
    """
    result = keyword_aggregate(
        ["Season 4 has 24 episodes."] * 4 + ["Season 4 has 23 episodes."]
    )
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
