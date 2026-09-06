import logging
import os
import math
from hashlib import blake2b

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from services.common.search import tokenize


class DeterministicHashEmbeddings(Embeddings):
    """Small offline embedding used for tests and the initial lab baseline."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


LOGGER = logging.getLogger(__name__)


def create_embeddings() -> Embeddings:
    """Build the embedding named by EMBEDDING_BACKEND.

    The chosen backend is logged because the default is the hash embedding,
    which is right for tests and wrong for any measurement: a harness that
    forgets to set EMBEDDING_BACKEND=ollama otherwise produces plausible
    numbers for a retriever nobody deploys. The log line is the record of
    which retriever a given run actually used.
    """
    backend = os.getenv("EMBEDDING_BACKEND", "deterministic").strip().lower()
    LOGGER.info("embedding backend: %s", backend)
    if backend == "deterministic":
        return DeterministicHashEmbeddings(
            dimensions=int(os.getenv("HASH_EMBEDDING_DIMENSIONS", "256"))
        )
    if backend == "ollama":
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv(
                "OLLAMA_EMBEDDING_BASE_URL",
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ).rstrip("/"),
        )
    raise ValueError(f"Unsupported EMBEDDING_BACKEND: {backend}")
