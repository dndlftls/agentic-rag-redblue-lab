"""Isolate-then-aggregate generation defense, adapted from RobustRAG.

RobustRAG (Xiang et al., *Certifiably Robust RAG against Retrieval Corruption*,
SaTML 2026, arXiv:2405.15556) answers each retrieved passage in isolation and
then aggregates the isolated responses, so an injected passage can influence at
most the responses of the group it lands in.

Two aggregations live here.

``keyword_aggregate`` is the paper's own rule, implemented with the paper's
default parameters (alpha=0.2, beta=3, Section V-A) so it can be measured
rather than argued about. Unique keywords are counted across the isolated
responses and those above ``min(alpha*n, beta)`` survive; the surviving
keywords -- and no passages -- are then given back to the model to phrase the
final answer.

``aggregate`` is this lab's variant. The paper scopes its guarantee to small
corruption budgets: Section II-C states robust generation is only "tractable
and meaningful" when useful benign passages outnumber malicious ones, and
Figure 8 shows robustness falling to zero once half the passages are corrupted.
Measured here, only about one retrieved passage supports the correct answer
(1.1 on average across eight scenarios, zero for one of them) while PoisonedRAG
injects five agreeing poisons, so this lab sits far outside that regime and
counting favours the attacker. The variant keeps the paper's isolation and
replaces counting with conflict detection: when non-abstaining isolated
responses disagree, neither claim is asserted and the disagreement is reported.
It depends on disagreement existing rather than on a majority, so it still
works when a single passage supports the truth.

Neither helps when every retrieved passage is poisoned -- there is no benign
response left to count or to disagree -- which is the regime the
retrieval-stage redundancy filter covers.

Decoding-based aggregation is not implemented. It is feasible (Ollama does
expose token logprobs) but needs one logprob call per group per generated
token, so a five-group answer of fifty tokens costs 250 sequential calls per
query. It fails for the same structural reason anyway: summing the isolated
next-token distributions lets five poisoned passages outweigh one benign one.
"""

import re
from dataclasses import dataclass

from services.common.schemas import SearchHit
from services.orchestrator.evaluation import normalize_text

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
class Aggregation:
    answer: str
    conflict: bool
    responses: list[str]
    abstained: int
    distinct_claims: list[str]


def isolate(hits: list[SearchHit], group_size: int) -> list[list[SearchHit]]:
    """Disjoint groups of ``group_size`` adjacent passages (paper's IsoGroup)."""
    return [
        hits[start : start + group_size]
        for start in range(0, len(hits), group_size)
    ]


def is_abstention(response: str) -> bool:
    lowered = response.casefold()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)$")
GENERIC_CAPITALS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "there", "here", "however", "therefore", "based", "according",
    "retrieved", "context", "passage", "answer", "question", "source",
}


def claim_signature(response: str, query: str = "") -> set[str]:
    """What a response asserts: salient content that is not already in the question.

    The paper extracts keywords to compare isolated responses; short factual
    answers in this lab differ on exactly these tokens ("24" against "23",
    "Frank Sinatra" against "Elvis Presley"), while the surrounding phrasing
    varies freely between isolated calls. Citations are dropped first because
    their document ids differ by construction.

    Capitalisation only counts away from a sentence start, so "The" in "The
    answer is 24" is not mistaken for a name.

    Query terms are subtracted because the answer is the part the question does
    not already contain. Without this, two responses that contradict each other
    still share their topic words -- "Chicago Fire season 4 has 24 episodes"
    against "... 23 episodes" overlap on `fire` and `4` -- and would be read as
    agreeing. Removing the query leaves {24} against {23}, which is the
    disagreement.
    """
    query_terms = {
        match.group().strip(",./").casefold()
        for match in re.finditer(r"[A-Za-z][A-Za-z'-]*|\d[\d,./]*", query)
    }
    text = re.sub(r"\[[^\]]*\]", " ", response)
    signature: set[str] = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z'-]*|\d[\d,./]*", text):
        token = match.group()
        if token[0].isdigit():
            signature.add(token.strip(",./").casefold())
            continue
        if not token[0].isupper():
            continue
        if token.casefold() in GENERIC_CAPITALS:
            continue
        if SENTENCE_START.search(text[: match.start()]):
            continue
        signature.add(token.casefold())
    return signature - query_terms


def claims_agree(left: set[str], right: set[str]) -> bool:
    """Two responses agree when their salient content is not disjoint.

    An empty signature carries no assertion to contradict, so it agrees with
    anything and cannot manufacture a conflict on its own.
    """
    if not left or not right:
        return True
    return bool(left & right)


def aggregate(responses: list[str], query: str = "") -> Aggregation:
    """Report a single claim only when the non-abstaining responses agree."""
    answered = [text for text in responses if text.strip() and not is_abstention(text)]
    abstained = len(responses) - len(answered)

    if not answered:
        return Aggregation(
            answer="I cannot determine the answer from the retrieved context.",
            conflict=False,
            responses=responses,
            abstained=abstained,
            distinct_claims=[],
        )

    groups: list[tuple[set[str], list[str]]] = []
    for text in answered:
        signature = claim_signature(text, query)
        for existing, members in groups:
            if claims_agree(signature, existing):
                existing |= signature
                members.append(text)
                break
        else:
            groups.append((set(signature), [text]))

    if len(groups) > 1:
        return Aggregation(
            answer=(
                "The retrieved passages disagree, so no answer is asserted. "
                "Conflicting claims were found in isolated evidence."
            ),
            conflict=True,
            responses=responses,
            abstained=abstained,
            distinct_claims=[" / ".join(sorted(sig)) for sig, _ in groups],
        )

    signature, members = groups[0]
    return Aggregation(
        answer=max(members, key=len).strip(),
        conflict=False,
        responses=responses,
        abstained=abstained,
        distinct_claims=[" / ".join(sorted(signature))],
    )


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

    The paper extracts these with an auxiliary LLM call; this is a lexical
    stand-in -- content words and numbers, citations stripped -- which keeps
    the aggregation deterministic and testable.

    The substitution does not decide the outcome. What the counting rule turns
    on is how many *responses* carry the discriminating token, and that is a
    property of the retrieved passages, not of the extractor: a fact supported
    by one passage appears in one response however keywords are extracted.
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
    """RobustRAG keyword aggregation with the paper's default parameters.

    Abstaining responses are dropped, keywords are counted once per remaining
    response, and a keyword survives when its count exceeds
    ``mu = min(alpha * n, beta)`` over the ``n`` non-abstaining responses
    (paper Section IV-B; defaults alpha=0.2, beta=3 from Section V-A).

    With the lab's usual n=5 this puts mu at 1.0, so a keyword must appear in
    at least two isolated responses to survive.
    """
    answered = [text for text in responses if text.strip() and not is_abstention(text)]
    counts: dict[str, int] = {}
    for text in answered:
        for keyword in response_keywords(text):
            counts[keyword] = counts.get(keyword, 0) + 1

    threshold = min(alpha * len(answered), beta)
    survivors = sorted(
        (keyword for keyword, count in counts.items() if count > threshold),
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


def isolation_only(responses: list[str]) -> Aggregation:
    """Isolation with no aggregation rule: the longest non-abstaining response.

    This is the paper's isolation stripped of every combination rule, and an
    ablation behind the redundancy filter measured it as the best of the three
    generation-stage options (correct 0.75, against 0.50 for answering all
    passages jointly and 0.62 for ``aggregate``). The gain over answering
    jointly comes from isolation itself -- one passage per call leaves the
    model less to be distracted by -- not from anything that inspects the
    responses afterwards.

    Keeping it as its own defense makes that attribution testable rather than
    hidden inside a variant that also does something else.
    """
    answered = [text for text in responses if text.strip() and not is_abstention(text)]
    if not answered:
        return Aggregation(
            answer="I cannot determine the answer from the retrieved context.",
            conflict=False,
            responses=responses,
            abstained=len(responses),
            distinct_claims=[],
        )
    return Aggregation(
        answer=max(answered, key=len).strip(),
        conflict=False,
        responses=responses,
        abstained=len(responses) - len(answered),
        distinct_claims=[],
    )
