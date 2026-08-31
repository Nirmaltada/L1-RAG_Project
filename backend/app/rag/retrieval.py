import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.models import RetrievedNode
from app.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)
COLLECTION_NAME = "rag_chunks"


class ChromaStores:
    """Persistent vectors plus a hydrated RAM mirror."""

    def __init__(self, persist_directory: Path) -> None:
        self.persistent_client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.ram_client = chromadb.EphemeralClient(
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.persistent_collection = self.persistent_client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        # EphemeralClient instances share an in-process system. Recreate the
        # mirror so repeated app lifespans cannot retain stale vectors.
        try:
            self.ram_client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass
        self.ram_collection = self.ram_client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.persistent_store = ChromaVectorStore(chroma_collection=self.persistent_collection)
        self.ram_store = ChromaVectorStore(chroma_collection=self.ram_collection)

    def hydrate_ram(self, batch_size: int = 500) -> int:
        """Copy saved vectors verbatim; no embedding provider is involved."""
        copied = 0
        total = self.persistent_collection.count()
        for offset in range(0, total, batch_size):
            batch = self.persistent_collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            ids = batch.get("ids") or []
            if not ids:
                continue
            embeddings = batch.get("embeddings")
            documents = batch.get("documents")
            metadatas = batch.get("metadatas")
            if embeddings is None or documents is None or metadatas is None:
                raise RuntimeError("Persistent Chroma data is missing fields required for hydration")
            if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
                raise RuntimeError("Persistent Chroma data is inconsistent and cannot be hydrated")
            self.ram_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            copied += len(ids)
        logger.info("Hydrated %s Chroma vectors into RAM", copied)
        return copied

    def embedding_dimension(self) -> int | None:
        for collection in (self.ram_collection, self.persistent_collection):
            try:
                result = collection.get(limit=1, include=["embeddings"])
            except Exception:
                logger.exception("Failed to inspect Chroma embedding dimension")
                continue
            embeddings = result.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                continue
            try:
                return len(embeddings[0])
            except TypeError:
                return None
        return None

    def ensure_embedding_mode(self, requested_mode: str) -> None:
        """Prevent API and local vector spaces from being mixed in one collection."""
        metadata = dict(self.persistent_collection.metadata or {})
        stored_mode = metadata.get("embedding_mode")
        count = self.persistent_collection.count()
        if stored_mode is None and count:
            # Collections created before this marker existed used OpenRouter.
            stored_mode = "api"
        if stored_mode is not None and stored_mode != requested_mode and count:
            raise RuntimeError(
                f"Chroma contains {stored_mode} embeddings, but {requested_mode} mode is "
                "configured. Delete existing chats/documents before switching embedding mode."
            )
        # Chroma 0.4 rejects hnsw:space in modify(), even when its value is
        # unchanged. The distance function remains attached internally.
        updated = {
            key: value for key, value in metadata.items() if key != "hnsw:space"
        }
        updated["embedding_mode"] = requested_mode
        self.persistent_collection.modify(metadata=updated)

    def add_nodes(self, nodes: list[BaseNode]) -> None:
        """Add already-embedded LlamaIndex nodes to both stores."""
        # LlamaIndex reserves the Chroma `document_id` field and derives it
        # from the node's SOURCE relationship. Keep that relationship aligned
        # with our application document ID so metadata deletion works.
        for node in nodes:
            document_id = str(node.metadata.get("document_id", ""))
            if document_id:
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=document_id
                )
        self.persistent_store.add(nodes)
        try:
            self.ram_store.add(nodes)
        except Exception:
            document_ids = {str(node.metadata.get("document_id", "")) for node in nodes}
            for document_id in document_ids:
                if document_id:
                    try:
                        self._delete_where({"document_id": document_id})
                    except Exception:
                        logger.exception("Failed to roll back vectors for document %s", document_id)
            raise

    def _delete_where(self, where: dict[str, str]) -> None:
        errors: list[Exception] = []
        for collection in (self.persistent_collection, self.ram_collection):
            try:
                collection.delete(where=where)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Chroma cleanup failed: {errors[0]}") from errors[0]

    def delete_document(self, document_id: str) -> None:
        self._delete_where({"document_id": document_id})

    def delete_chat(self, chat_id: str) -> None:
        self._delete_where({"chat_id": chat_id})


def _decode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(metadata)
    for key in ("topics", "keywords"):
        value = decoded.get(key, "[]")
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                decoded[key] = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                decoded[key] = []
    page = decoded.get("page_number")
    if page in ("", 0, "0"):
        decoded["page_number"] = None
    return decoded


def metadata_relevance(metadata: dict[str, Any], intent: dict[str, Any]) -> float:
    hints = {
        str(item).casefold()
        for item in [*(intent.get("likely_categories") or []), *(intent.get("topics") or [])]
        if item
    }
    if not hints:
        return 0.0
    values = {
        str(item).casefold()
        for item in [
            metadata.get("category", ""),
            metadata.get("document_type", ""),
            *(metadata.get("topics") or []),
            *(metadata.get("keywords") or []),
        ]
        if item
    }
    exact = len(hints & values)
    partial = sum(1 for hint in hints for value in values if hint in value or value in hint)
    return min(1.0, exact * 0.25 + partial * 0.08)


class Retriever:
    def __init__(
        self,
        stores: ChromaStores,
        embeddings: EmbeddingService,
    ) -> None:
        self.stores = stores
        self.embeddings = embeddings

    async def retrieve(
        self,
        chat_id: str,
        query: str,
        intent: dict[str, Any],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedNode]:
        embeddings, model = await self.embeddings.embed(
            [query],
            expected_dimension=self.stores.embedding_dimension(),
        )
        logger.info("Retrieval embedding model=%s", model)
        query_embedding = embeddings[0]
        retrieved: list[RetrievedNode] = []

        # A comparison needs evidence from every requested document. Running a
        # small filtered query per document prevents one large/semantically
        # repetitive file from occupying all global top-k slots.
        scopes: list[str | None] = list(dict.fromkeys(document_ids or [])) or [None]
        per_scope_k = max(1, (top_k + len(scopes) - 1) // len(scopes))

        def query_scope(document_id: str | None):
            filters = [ExactMatchFilter(key="chat_id", value=chat_id)]
            if document_id:
                filters.append(ExactMatchFilter(key="document_id", value=document_id))
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=per_scope_k if document_id else top_k,
                filters=MetadataFilters(filters=filters, condition=FilterCondition.AND),
            )
            return self.stores.ram_store.query(vector_query)

        results = await asyncio.gather(
            *(asyncio.to_thread(query_scope, document_id) for document_id in scopes)
        )
        for result in results:
            nodes = result.nodes or []
            similarities = result.similarities or [0.0] * len(nodes)
            for node, score in zip(nodes, similarities, strict=False):
                metadata = _decode_metadata(node.metadata)
                retrieved.append(
                    RetrievedNode(
                        node_id=node.node_id,
                        text=node.get_content(),
                        metadata=metadata,
                        vector_score=float(score or 0.0),
                        metadata_score=metadata_relevance(metadata, intent),
                    )
                )

        retrieved.sort(
            key=lambda item: item.vector_score + (0.08 * item.metadata_score), reverse=True
        )
        # Scoped searches return at least one candidate per requested document,
        # even when a chat contains more documents than the usual global top-k.
        return retrieved if document_ids else retrieved[:top_k]
