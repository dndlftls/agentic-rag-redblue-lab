import math

import pytest

from services.common.ragmask import (
    RagMaskConfig,
    assemble,
    candidate_count,
    cosine,
    keep_segments,
    masked_variants,
    segment,
)


def test_segment_splits_into_fixed_length_pieces() -> None:
    text = " ".join(f"w{index}" for index in range(7))

    pieces = segment(text, 3)

    assert [len(piece.split()) for piece in pieces] == [3, 3, 1]
    assert " ".join(pieces) == text


def test_segment_handles_empty_text() -> None:
    assert segment("", 3) == []
    assert segment("   ", 3) == []


def test_masked_variants_occlude_one_segment_each() -> None:
    pieces = ["a b", "c d", "e f"]

    variants = masked_variants(pieces)

    assert variants == ["c d e f", "a b e f", "a b c d"]


def test_keep_rule_follows_the_paper() -> None:
    """Paper 3.2: keep the masked tokens when v' + delta > v."""
    original = 0.90

    # Masking segment 0 costs 0.50 -> it carried the similarity -> discard.
    # Masking segment 1 costs 0.02 -> below delta -> keep.
    keep = keep_segments(original, [0.40, 0.88], delta=0.05)

    assert keep == [False, True]


def test_keep_rule_boundary_is_strict() -> None:
    # v' + delta == v is not "> v", so the segment is discarded.
    assert keep_segments(0.9, [0.85], delta=0.05) == [False]
    assert keep_segments(0.9, [0.851], delta=0.05) == [True]


def test_delta_controls_the_defense_utility_trade_off() -> None:
    """Paper Table 11/12: delta dominates. Same drops, opposite decisions.

    A poison segment costing 0.30 of similarity is caught at delta=0.01 but
    ignored at delta=0.5, which is why the paper reports ASR 5-11% at the
    former and 57% at the latter.
    """
    masked = [0.60, 0.88]  # drops of 0.30 and 0.02 from an original of 0.90

    assert keep_segments(0.90, masked, delta=0.01) == [False, False]
    assert keep_segments(0.90, masked, delta=0.05) == [False, True]
    assert keep_segments(0.90, masked, delta=0.50) == [True, True]


def test_assemble_drops_the_discarded_segments() -> None:
    pieces = ["poison text", "real content", "more content"]

    assert assemble(pieces, [False, True, True]) == "real content more content"


def test_candidate_count_applies_overfetch() -> None:
    assert candidate_count(5, 2.0) == 10
    assert candidate_count(5, 1.0) == 5
    assert candidate_count(3, 1.5) == 5  # ceil, never below limit


def test_cosine_matches_manual_computation() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_config_validates_hyperparameters() -> None:
    RagMaskConfig(mask_length=15, delta=0.05, overfetch=2.0)
    with pytest.raises(ValueError):
        RagMaskConfig(mask_length=0)
    with pytest.raises(ValueError):
        RagMaskConfig(delta=-0.1)
    with pytest.raises(ValueError):
        RagMaskConfig(overfetch=0.5)
