"""RAGMask: retrieval-stage sanitisation defense from arXiv:2512.24268.

Where RAGPart changes how documents are indexed, RAGMask leaves the index
alone and cleans the candidates a query already retrieved. The top ``alpha * p``
documents are each split into fixed-length segments; every segment is masked in
turn and the query similarity recomputed. A segment whose removal makes the
document markedly *less* similar to the query was doing the retrieval work, so
it is dropped. The surviving text is re-embedded and re-ranked, and the top
``p`` sanitised documents are returned.

The paper's rule (Section 3.2) is stated in terms of keeping: for original
similarity ``v`` and masked similarity ``v'``, keep the masked tokens when
``v' + delta > v``. Equivalently a segment is discarded when masking it costs
at least ``delta`` of similarity, which is what a retrieval-boosting poison
does.

This module holds only the embedding-free logic so it can be tested directly;
`ChromaDocumentStore.search_ragmask` supplies the vectors.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RagMaskConfig:
    """Paper hyperparameters (Appendix Table 11/12, FiQA).

    ``delta`` dominates the trade-off: at ``delta=0.01`` the paper drives ASR
    from 95% to 5-11% but loses 8-9 points of SR, while ``delta=0.5`` preserves
    utility completely and defends nothing. Larger ``mask_length`` is better at
    small ``delta``. The defaults here sit in the middle of that table.
    """

    mask_length: int = 15
    delta: float = 0.05
    overfetch: float = 2.0

    def __post_init__(self) -> None:
        if self.mask_length < 1:
            raise ValueError("mask_length must be positive")
        if self.delta < 0:
            raise ValueError("delta must not be negative")
        if self.overfetch < 1:
            raise ValueError("overfetch must be at least 1")


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def segment(text: str, mask_length: int) -> list[str]:
    """Split text into consecutive segments of ``mask_length`` tokens."""
    tokens = text.split()
    if not tokens:
        return []
    return [
        " ".join(tokens[start : start + mask_length])
        for start in range(0, len(tokens), mask_length)
    ]


def masked_variants(segments: list[str]) -> list[str]:
    """One variant per segment, with that segment occluded."""
    return [
        " ".join(other for index, other in enumerate(segments) if index != position)
        for position in range(len(segments))
    ]


def keep_segments(
    original_score: float,
    masked_scores: list[float],
    delta: float,
) -> list[bool]:
    """Paper rule: keep segment i when ``v' + delta > v``.

    A poison segment carries the query similarity, so removing it drops the
    score by more than ``delta`` and the segment is discarded.
    """
    return [masked + delta > original_score for masked in masked_scores]


def assemble(segments: list[str], keep: list[bool]) -> str:
    kept = [text for text, keeping in zip(segments, keep, strict=True) if keeping]
    return " ".join(kept)


def candidate_count(limit: int, overfetch: float) -> int:
    """``alpha * p`` documents are sanitised before re-ranking down to ``p``."""
    return max(limit, math.ceil(limit * overfetch))
