import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from services.common.embeddings import DeterministicHashEmbeddings
from services.common.cluster_filter import (
    ClusterFilterConfig,
    cluster,
    redundant_indices,
)
from services.common.ragmask import (
    RagMaskConfig,
    assemble,
    candidate_count,
    cosine,
    keep_segments,
    masked_variants,
    segment,
)
from services.common.ragpart import (
    RagPartConfig,
    combination_count,
    combination_vectors,
    majority_vote,
    partition_text,
)
from services.common.schemas import SearchHit
from services.common.search import load_json_documents


class ChromaDocumentStore:
    def __init__(
        self,
        data_file: Path,
        *,
        collection_name: str,
        persist_directory: Path | None = None,
        embedding: Embeddings | None = None,
        ragpart: RagPartConfig | None = None,
        ragmask: RagMaskConfig | None = None,
        cluster_filter: ClusterFilterConfig | None = None,
    ) -> None:
        self.data_file = data_file
        self.embedding = embedding or DeterministicHashEmbeddings()
        self.ragpart = ragpart or RagPartConfig()
        # RAGMask needs no index of its own -- it sanitises whatever the
        # ordinary search returns -- so it costs query time instead of
        # startup time and works on an existing collection.
        self.ragmask = ragmask or RagMaskConfig()
        # The redundancy filter also needs no index; it compares candidates
        # against each other rather than against the query.
        self.cluster_filter = cluster_filter or ClusterFilterConfig()
        persist = str(persist_directory) if persist_directory is not None else None
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=persist,
        )
        # RAGPart keeps C(N, k) mean-pooled fragment vectors per document in a
        # side collection. Only ids and metadata live here; hit text is read
        # back from the primary collection so passages are not duplicated.
        self._ragpart_store = Chroma(
            collection_name=f"{collection_name}-ragpart",
            embedding_function=self.embedding,
            persist_directory=persist,
        )
        self._index_documents()

    def _index_documents(self) -> None:
        records = load_json_documents(self.data_file)
        existing_ids_list: list[str] = []
        existing_metadatas: list[dict[str, object] | None] = []
        page_size = 5_000
        for offset in range(0, self.count(), page_size):
            page = self.vector_store.get(
                include=["metadatas"], limit=page_size, offset=offset
            )
            existing_ids_list.extend(str(item) for item in page.get("ids", []))
            existing_metadatas.extend(page.get("metadatas", []))
        existing_ids = set(existing_ids_list)
        sync_trusted = os.getenv("CHROMA_SYNC_TRUSTED_CORPUS", "false").lower() in {
            "1", "true", "yes"
        }
        if sync_trusted:
            desired_ids = {str(record["id"]) for record in records}
            stale_trusted_ids = [
                str(document_id)
                for document_id, metadata in zip(
                    existing_ids_list,
                    existing_metadatas,
                )
                if (metadata or {}).get("trust") == "trusted"
                and str(document_id) not in desired_ids
            ]
            if stale_trusted_ids:
                self.vector_store.delete(ids=stale_trusted_ids)
                existing_ids.difference_update(stale_trusted_ids)

        pending_records = [
            record for record in records if str(record["id"]) not in existing_ids
        ]

        documents = [
            Document(
                page_content=record["text"],
                metadata={
                    "document_id": record["id"],
                    "source": record["source"],
                    "trust": record.get("trust", "unknown"),
                    "tags": json.dumps(record.get("tags", [])),
                },
            )
            for record in pending_records
        ]
        batch_size = max(1, int(os.getenv("CHROMA_INDEX_BATCH_SIZE", "1000")))
        for start in range(0, len(pending_records), batch_size):
            end = start + batch_size
            self.vector_store.add_documents(
                documents=documents[start:end],
                ids=[record["id"] for record in pending_records[start:end]],
            )
        for record in pending_records:
            self._index_ragpart(record["id"], record["text"])

    def _index_ragpart(self, document_id: str, text: str) -> None:
        # Building the side index costs `fragments` embedding calls per
        # document, so it stays opt-in: on the full NQ corpus that is hundreds
        # of thousands of calls against the embedding server at startup.
        if not self.ragpart.enabled:
            return
        fragments = partition_text(text, self.ragpart.fragments)
        vectors = combination_vectors(
            self.embedding.embed_documents(fragments),
            self.ragpart.combination_size,
        )
        self._ragpart_store._collection.upsert(
            ids=[f"{document_id}#c{index}" for index in range(len(vectors))],
            embeddings=vectors,
            metadatas=[
                {"document_id": document_id, "combo_index": index}
                for index in range(len(vectors))
            ],
        )

    def _delete_ragpart(self, document_ids: list[str]) -> None:
        if document_ids:
            self._ragpart_store._collection.delete(
                where={"document_id": {"$in": document_ids}}
            )

    def search_ragpart(self, query: str, limit: int) -> list[SearchHit]:
        """RAGPart retrieval: per-combination top-p, then majority vote.

        Each combination index is searched as its own database, exactly as in
        the paper. Ranking on the mean combination score instead would remove
        the defense, because mean pooling does not lower a poisoned document's
        score under this embedding -- only its combination coverage drops.
        """
        if not self.ragpart.enabled:
            raise RuntimeError(
                "RAGPart index is not built. Set RAGPART_ENABLED=true and "
                "reindex the collection before requesting this defense."
            )
        query_vector = self.embedding.embed_query(query)
        ranked_sets: list[list[tuple[str, float]]] = []
        for combo_index in range(
            combination_count(
                self.ragpart.fragments,
                self.ragpart.combination_size,
            )
        ):
            response = self._ragpart_store._collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where={"combo_index": combo_index},
                include=["metadatas", "distances"],
            )
            metadatas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]
            ranked_sets.append(
                [
                    (str(metadata["document_id"]), self._to_score(distance))
                    for metadata, distance in zip(
                        metadatas, distances, strict=True
                    )
                ]
            )

        voted = majority_vote(ranked_sets, limit)
        return self._hits_by_id(
            {document_id: score for document_id, score, _ in voted}
        )

    def _hits_by_id(self, scores: dict[str, float]) -> list[SearchHit]:
        if not scores:
            return []
        records = self.vector_store.get(
            ids=list(scores),
            include=["metadatas", "documents"],
        )
        by_id = {
            str(metadata["document_id"]): (metadata, text)
            for metadata, text in zip(
                records.get("metadatas", []),
                records.get("documents", []),
                strict=True,
            )
        }
        hits: list[SearchHit] = []
        for document_id, score in scores.items():
            if document_id not in by_id:
                continue
            metadata, text = by_id[document_id]
            hits.append(
                SearchHit(
                    document_id=document_id,
                    source=str(metadata["source"]),
                    trust=str(metadata.get("trust", "unknown")),
                    tags=self._decode_tags(metadata.get("tags", "[]")),
                    text=text,
                    score=score,
                )
            )
        return hits

    @staticmethod
    def _to_score(distance: float) -> float:
        # Chroma returns a distance where a lower value is a closer match.
        return round(1.0 / (1.0 + max(float(distance), 0.0)), 6)

    def search_ragmask(self, query: str, limit: int) -> list[SearchHit]:
        """RAGMask retrieval: sanitise the top alpha*p candidates, then re-rank.

        Each candidate is split into `mask_length`-token segments. A segment is
        dropped when masking it costs at least `delta` of query similarity,
        which is what a retrieval-boosting poison does. The kept text is
        re-embedded and the candidates re-ranked, and the top `limit` are
        returned with their sanitised score.

        Embedding calls are batched: one request covers every masked variant of
        every candidate, and a second covers the sanitised texts.
        """
        candidates = self.search(
            query, candidate_count(limit, self.ragmask.overfetch)
        )
        if not candidates:
            return []

        query_vector = self.embedding.embed_query(query)
        segments = [segment(hit.text, self.ragmask.mask_length) for hit in candidates]

        # One batch: each candidate's original text followed by its masked
        # variants. Offsets let us slice the scores back out per candidate.
        batch: list[str] = []
        offsets: list[tuple[int, int]] = []
        for hit, pieces in zip(candidates, segments, strict=True):
            variants = masked_variants(pieces) if len(pieces) > 1 else []
            offsets.append((len(batch), len(variants)))
            batch.append(hit.text)
            batch.extend(variants)
        scores = [
            cosine(query_vector, vector)
            for vector in self.embedding.embed_documents(batch)
        ]

        sanitised: list[str] = []
        for (start, count), pieces in zip(offsets, segments, strict=True):
            if count == 0:
                sanitised.append(" ".join(pieces))
                continue
            keep = keep_segments(
                scores[start],
                scores[start + 1 : start + 1 + count],
                self.ragmask.delta,
            )
            # Never sanitise a document down to nothing; if every segment looks
            # suspicious the document is left intact for the re-ranking to judge.
            text = assemble(pieces, keep)
            sanitised.append(text if text.strip() else " ".join(pieces))

        final = [
            cosine(query_vector, vector)
            for vector in self.embedding.embed_documents(sanitised)
        ]
        ranked = sorted(
            zip(candidates, sanitised, final, strict=True),
            key=lambda item: (-item[2], item[0].document_id),
        )
        return [
            hit.model_copy(update={"text": text, "score": round(score, 6)})
            for hit, text, score in ranked[:limit]
        ]

    def search_cluster(self, query: str, limit: int) -> list[SearchHit]:
        """Drop candidates that form a suspiciously tight cluster.

        PoisonedRAG needs several near-duplicate poisons per query, so the
        attack shows up as redundancy between candidates even when each one
        looks individually plausible against the query. Clean documents in this
        corpus never form such a group -- see services/common/cluster_filter.py
        for the measurements behind the default threshold.

        Falls back to the undefended ranking if every candidate is dropped, so
        the filter can never empty a result set.
        """
        candidates = self.search(
            query, candidate_count(limit, self.cluster_filter.overfetch)
        )
        if len(candidates) < self.cluster_filter.min_cluster_size:
            return candidates[:limit]

        vectors = self.embedding.embed_documents([hit.text for hit in candidates])
        groups = cluster(vectors, self.cluster_filter.similarity_threshold)
        dropped = redundant_indices(groups, self.cluster_filter.min_cluster_size)
        kept = [
            hit for index, hit in enumerate(candidates) if index not in dropped
        ]
        return (kept or candidates)[:limit]

    def search(self, query: str, limit: int) -> list[SearchHit]:
        results = self.vector_store.similarity_search_with_score(query, k=limit)
        hits: list[SearchHit] = []
        for document, distance in results:
            metadata = document.metadata
            # Chroma returns a distance where a lower value is a closer match.
            score = 1.0 / (1.0 + max(float(distance), 0.0))
            hits.append(
                SearchHit(
                    document_id=str(metadata["document_id"]),
                    source=str(metadata["source"]),
                    trust=str(metadata.get("trust", "unknown")),
                    tags=self._decode_tags(metadata.get("tags", "[]")),
                    text=document.page_content,
                    score=round(score, 6),
                )
            )
        return hits

    def add_document(
        self,
        *,
        document_id: str,
        source: str,
        trust: str,
        tags: list[str],
        text: str,
    ) -> None:
        if self.contains(document_id):
            raise ValueError(f"Document already exists: {document_id}")

        self.vector_store.add_documents(
            documents=[
                Document(
                    page_content=text,
                    metadata={
                        "document_id": document_id,
                        "source": source,
                        "trust": trust,
                        "tags": json.dumps(tags),
                    },
                )
            ],
            ids=[document_id],
        )
        self._index_ragpart(document_id, text)

    def contains(self, document_id: str) -> bool:
        return bool(self.vector_store.get(ids=[document_id]).get("ids", []))

    def count(self) -> int:
        return self.vector_store._collection.count()

    def document_counts(self) -> dict[str, int]:
        untrusted = self.vector_store.get(where={"trust": "untrusted"}, include=[])
        untrusted_count = len(untrusted.get("ids", []))
        total_count = self.count()
        return {
            "trusted": total_count - untrusted_count,
            "untrusted": untrusted_count,
            "total": total_count,
        }

    def delete_untrusted_documents(self) -> int:
        records = self.vector_store.get(
            where={"trust": "untrusted"},
            include=[],
        )
        document_ids = [str(item) for item in records.get("ids", [])]
        if document_ids:
            self.vector_store.delete(ids=document_ids)
            self._delete_ragpart(document_ids)
        return len(document_ids)

    def delete_untrusted_document(self, document_id: str) -> bool:
        records = self.vector_store.get(
            ids=[document_id],
            include=["metadatas"],
        )
        ids = records.get("ids", [])
        if not ids:
            return False
        metadatas = records.get("metadatas", [])
        metadata = metadatas[0] if metadatas else {}
        if metadata.get("trust") != "untrusted":
            raise ValueError("Only untrusted experiment documents can be deleted")
        self.vector_store.delete(ids=[document_id])
        self._delete_ragpart([document_id])
        return True

    @staticmethod
    def _decode_tags(value: object) -> list[str]:
        if not isinstance(value, str):
            return []
        try:
            tags = json.loads(value)
        except json.JSONDecodeError:
            return []
        if not isinstance(tags, list):
            return []
        return [str(tag) for tag in tags]
