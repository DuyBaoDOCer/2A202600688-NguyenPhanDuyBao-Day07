from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        embedding = self._embedding_fn(doc.content)
        # Merge doc_id into metadata so we can filter/delete by it later
        metadata = dict(doc.metadata)
        metadata["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search (dot product) over provided records."""
        query_embedding = self._embedding_fn(query)

        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({
                "content": record["content"],
                "score": score,
                "metadata": record["metadata"],
            })

        # Sort descending by score and return top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                record = self._make_record(doc)
                # ChromaDB requires unique IDs; use a counter suffix to allow
                # re-adding the same doc_id (e.g. chunked documents)
                chroma_id = f"{doc.id}__{self._next_index}"
                self._next_index += 1
                ids.append(chroma_id)
                documents.append(doc.content)
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for content, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB returns L2 distances; convert to a similarity-like score
                score = 1.0 / (1.0 + dist)
                output.append({"content": content, "score": score, "metadata": meta})
            return output
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            # Build ChromaDB where clause from metadata_filter dict
            where = {k: {"$eq": v} for k, v in metadata_filter.items()}
            if len(where) == 1:
                where = {list(where.keys())[0]: list(where.values())[0]["$eq"]}
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for content, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1.0 / (1.0 + dist)
                output.append({"content": content, "score": score, "metadata": meta})
            return output
        else:
            # Pre-filter in-memory records by metadata
            filtered = [
                record for record in self._store
                if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            # Query all items with matching doc_id in metadata
            results = self._collection.get(where={"doc_id": doc_id})
            ids_to_delete = results.get("ids", [])
            if not ids_to_delete:
                return False
            self._collection.delete(ids=ids_to_delete)
            return True
        else:
            original_len = len(self._store)
            self._store = [
                record for record in self._store
                if record["metadata"].get("doc_id") != doc_id
            ]
            return len(self._store) < original_len
