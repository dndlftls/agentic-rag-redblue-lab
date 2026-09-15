"""Isolation-based generation defenses from RobustRAG.

RobustRAG (Xiang et al., *Certifiably Robust RAG against Retrieval Corruption*,
SaTML 2026, arXiv:2405.15556) answers each retrieved passage in isolation and
then combines the isolated responses, so an injected passage can influence at
most the response of the group it lands in.

Two combinations live here, both measured in docs/robustrag.ko.md.

``keyword_aggregate`` is the paper's keyword aggregation (Algorithm 1): count
keywords once per isolated response, keep those with ``c >= min(alpha*n,
beta)``, and answer from the surviving keywords alone.

``isolation_only`` applies the isolation and nothing else, answering from the
longest non-abstaining response. Behind the redundancy filter it measured best
of the options tried. It is not a defense on its own: when every retrieved
passage is poisoned, every isolated response carries the attacker's claim and
the longest one is returned.

A conflict-detection variant (``isolate_conflict``) used to live here. It was
removed after measurement: on its own it raised ASR above the undefended
baseline, because a context saturated with agreeing poisons leaves no
disagreement to detect, and behind the redundancy filter an ablation showed its
apparent gain came entirely from the isolation it shared with
``isolation_only`` while its conflict rule lowered the correct-answer rate.

Decoding-based aggregation is not implemented. It is feasible (Ollama does
expose token logprobs) but needs one logprob call per group per generated
token, so a five-group answer of fifty tokens costs 250 sequential calls per
query.
"""

import re
from dataclasses import dataclass

from services.common.schemas import SearchHit

ABSTENTION_MARKERS = (
    "cannot determine",
    "not contain",
    "no information",
    "unable to determine",
)


@dataclass(frozen=True)
class RobustRagConfig:
    """``group_size`` is the paper's isolation width omega.

    omega=1 gives the tightest isolation: one passage per LLM call, so an
    injected passage can corrupt exactly one response.
    """

    group_size: int = 1
    max_passages: int = 6

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.max_passages < 1:
            raise ValueError("max_passages must be positive")


@dataclass(frozen=True)
class IsolatedAnswer:
    answer: str
    responses: list[str]
    abstained: int


def isolate(hits: list[SearchHit], group_size: int) -> list[list[SearchHit]]:
    """Disjoint groups of ``group_size`` adjacent passages (paper's IsoGroup)."""
    return [
        hits[start : start + group_size]
        for start in range(0, len(hits), group_size)
    ]


def is_abstention(response: str) -> bool:
    lowered = response.casefold()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


# ---------------------------------------------------------------------------
# The paper's own keyword aggregation, kept so it can be measured directly.
# ---------------------------------------------------------------------------

KEYWORD_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "that",
        "this", "these", "those", "there", "here", "is", "are", "was", "were",
        "be", "been", "being", "am", "do", "does", "did", "have", "has", "had",
        "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
        "as", "into", "over", "after", "before", "between", "during", "it",
        "its", "he", "she", "they", "them", "his", "her", "their", "we", "you",
        "i", "not", "no", "can", "cannot", "could", "will", "would", "should",
        "may", "might", "must", "so", "such", "also", "only", "very", "more",
        "most", "some", "any", "each", "which", "who", "whom", "whose", "what",
        "when", "where", "why", "how", "according", "based", "context",
        "passage", "passages", "retrieved", "answer", "question", "source",
        "document", "documents", "information", "states", "state", "said",
        "says", "provided", "given", "text",
    }
)


@dataclass(frozen=True)
class KeywordAggregation:
    """Outcome of the paper's keyword aggregation, with its working exposed.

    ``counts`` and ``threshold`` are kept so an experiment can report *why* a
    keyword survived or was dropped rather than only the final answer.
    """

    keywords: list[str]
    counts: dict[str, int]
    threshold: float
    answered: int
    abstained: int
    responses: list[str]


def response_keywords(response: str) -> set[str]:
    """Keywords of one isolated response.

    The paper extracts keywords and keyphrases from the text between adjacent
    uninformative words (Appendix B); the official code does this with spaCy
    part-of-speech tagging, keeping lemmatised noun-phrase chunks, their
    individual lemmas, and the whole response string. This is a lexical
    stand-in -- single casefolded content words and numbers, citations
    stripped -- chosen to avoid a spaCy dependency. It is not a reproduction
    of the official extractor: it produces no multi-word phrases and no
    lemmas, so "Frank Sinatra" counts as two tokens rather than one phrase.
    """
    text = re.sub(r"\[[^\]]*\]", " ", response)
    keywords: set[str] = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z'-]*|\d[\d,./]*", text):
        token = match.group().strip(",./").casefold()
        if not token or token in KEYWORD_STOPWORDS:
            continue
        keywords.add(token)
    return keywords


def keyword_aggregate(
    responses: list[str],
    *,
    alpha: float = 0.2,
    beta: float = 3.0,
) -> KeywordAggregation:
    """RobustRAG keyword aggregation with the paper's short-answer QA parameters.

    Abstaining responses are dropped, keywords are counted once per remaining
    response, and a keyword survives when its count is *at least*
    ``mu = min(alpha * n, beta)`` over the ``n`` non-abstaining responses --
    Algorithm 1, line 15: ``W* <- {w | (w, c) in C, c >= mu}``. The official
    implementation (inspire-group/RobustRAG, ``KeywordAgg``) deletes keywords
    with ``count < count_threshold``, which is the same rule. The paper sets
    alpha=0.2, beta=3 for short-answer QA.

    An earlier version of this function used ``count > mu``. That differs from
    the paper only when ``alpha * n`` is a whole number, which at alpha=0.2
    means n=5 -- exactly the case where all five retrieved passages are
    poisoned and none of the isolated responses abstains.

    Note what these parameters do at this lab's top-5: ``mu`` never exceeds 1.0,
    and every counted keyword has a count of at least 1, so **the count filter
    removes nothing**. The paper tuned alpha and beta for k=10, where n=10 gives
    ``mu = 2.0``. At k=5 any defensive effect comes from the final step alone --
    answering from the keyword list instead of from the passages.
    """
    answered = [text for text in responses if text.strip() and not is_abstention(text)]
    counts: dict[str, int] = {}
    for text in answered:
        for keyword in response_keywords(text):
            counts[keyword] = counts.get(keyword, 0) + 1

    threshold = min(alpha * len(answered), beta)
    survivors = sorted(
        (keyword for keyword, count in counts.items() if count >= threshold),
        key=lambda keyword: (-counts[keyword], keyword),
    )
    return KeywordAggregation(
        keywords=survivors,
        counts=counts,
        threshold=threshold,
        answered=len(answered),
        abstained=len(responses) - len(answered),
        responses=responses,
    )


def isolation_only(responses: list[str]) -> IsolatedAnswer:
    """Isolation with no combining rule: the longest non-abstaining response.

    This is the paper's isolation stripped of every combination rule. Behind
    the redundancy filter an ablation measured it at 0.75 correct, against 0.50
    for answering all passages jointly: one passage per call leaves the model
    less to be distracted by.

    It only helps once retrieval has already removed the poisons. With a fully
    poisoned context every isolated response asserts the attacker's claim, and
    picking the longest returns it confidently.
    """
    answered = [text for text in responses if text.strip() and not is_abstention(text)]
    if not answered:
        return IsolatedAnswer(
            answer="I cannot determine the answer from the retrieved context.",
            responses=responses,
            abstained=len(responses),
        )
    return IsolatedAnswer(
        answer=max(answered, key=len).strip(),
        responses=responses,
        abstained=len(responses) - len(answered),
    )
