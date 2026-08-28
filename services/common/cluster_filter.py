"""Redundancy filter: drop suspiciously tight clusters of retrieved documents.

RAGPart and RAGMask judge each document on its own against the query, which is
why they fail here: after sanitising, a poison scores about as well as a real
golden passage (measured 0.75 vs 0.74-0.84). PoisonedRAG's weakness is not in
any single document but *between* documents -- the attack needs several poisons
per query, all asserting the same wrong answer, so they end up near-duplicates.

Measured on this lab's corpus with `nomic-embed-text`:

    poison-to-poison cosine   0.9745  (minimum observed 0.9409)
    clean-to-clean cosine     0.4546  (maximum observed 0.8705)

Clean documents never formed a mutually-similar group larger than one at a
threshold of 0.88 or above, while poisons cluster far past it. The default
threshold sits between those two measurements.

Prior work: TrustRAG (arXiv:2501.00879, AAAI 2026 TrustAgent workshop) filters
retrieved documents with k-means before generation. This module uses average
linkage instead of k-means because the number of clusters is unknown and the
decision here is "is any group abnormally tight", not "partition into k".
"""

from dataclasses import dataclass

from services.common.ragmask import cosine


@dataclass(frozen=True)
class ClusterFilterConfig:
    """Defaults are measured, not guessed -- see the module docstring.

    ``similarity_threshold`` sits between the highest clean-document similarity
    observed (0.8705) and the lowest poison-to-poison similarity (0.9409).
    ``min_cluster_size`` of 2 is the smallest group that can be called
    redundant; the attack is inert against this filter if it injects only one
    document, which is the defense's main limitation.
    """

    similarity_threshold: float = 0.90
    min_cluster_size: int = 2
    overfetch: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within 0..1")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.overfetch < 1:
            raise ValueError("overfetch must be at least 1")


def _average_similarity(
    vectors: list[list[float]],
    left: list[int],
    right: list[int],
) -> float:
    return sum(
        cosine(vectors[i], vectors[j]) for i in left for j in right
    ) / (len(left) * len(right))


def cluster(
    vectors: list[list[float]],
    similarity_threshold: float,
) -> list[list[int]]:
    """Average-linkage agglomerative clustering, stopped at the threshold.

    Returns index groups. Candidate counts here are small (``overfetch`` times
    the requested top-k), so the naive O(n^3) merge loop is not a concern.
    """
    clusters = [[index] for index in range(len(vectors))]
    while len(clusters) > 1:
        best_score = -1.0
        best_pair = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                score = _average_similarity(vectors, clusters[a], clusters[b])
                if score > best_score:
                    best_score, best_pair = score, (a, b)
        if best_score < similarity_threshold:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    return clusters


def redundant_indices(
    clusters: list[list[int]],
    min_cluster_size: int,
) -> set[int]:
    """Indices belonging to a group large enough to look like injected redundancy."""
    return {
        index
        for group in clusters
        if len(group) >= min_cluster_size
        for index in group
    }
