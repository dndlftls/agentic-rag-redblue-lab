"""Isolate-then-aggregate generation defense, adapted from RobustRAG.

RobustRAG (Xiang et al., *Certifiably Robust RAG against Retrieval Corruption*,
SaTML 2026, arXiv:2405.15556) answers each retrieved passage in isolation and
then aggregates the isolated responses, so an injected passage can influence at
most the responses of the group it lands in.

**Why the aggregation here is not the paper's.** The paper aggregates by
counting keywords across isolated responses and keeping those above
``min(alpha*n, beta)``. That needs benign passages to outnumber corrupted ones.
Measured on this lab's corpus, only 1.4 passages on average support the correct
answer (minimum 1), while PoisonedRAG injects three or more agreeing poisons. A
count threshold of 2 therefore discards the correct answer in 3 of 5 scenarios,
and a threshold of 1 admits a lone poison. Majority aggregation favours the
attacker here, so this module keeps the paper's isolation and replaces the
majority rule with conflict detection: when non-abstaining isolated responses
disagree, neither claim is asserted and the disagreement is reported.

That converts a confidently wrong answer into a detected conflict. It cannot
help when every retrieved passage is poisoned -- there is no disagreement left
to detect -- which is the regime the retrieval-stage redundancy filter covers.
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


def claim_signature(response: str) -> set[str]:
    """Salient content of a response: numerals and proper nouns.

    The paper extracts keywords to compare isolated responses; short factual
    answers in this lab differ on exactly these tokens ("24" against "23",
    "Frank Sinatra" against "Elvis Presley"), while the surrounding phrasing
    varies freely between isolated calls. Citations are dropped first because
    their document ids differ by construction.

    Capitalisation only counts away from a sentence start, so "The" in "The
    answer is 24" is not mistaken for a name.
    """
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
    return signature


def claims_agree(left: set[str], right: set[str]) -> bool:
    """Two responses agree when their salient content is not disjoint.

    An empty signature carries no assertion to contradict, so it agrees with
    anything and cannot manufacture a conflict on its own.
    """
    if not left or not right:
        return True
    return bool(left & right)


def aggregate(responses: list[str]) -> Aggregation:
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
        signature = claim_signature(text)
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
