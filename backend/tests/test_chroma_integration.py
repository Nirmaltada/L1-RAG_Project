import json
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode

from app.rag.retrieval import ChromaStores, Retriever


def _node(node_id: str, chat_id: str, document_id: str, embedding: list[float]) -> TextNode:
    return TextNode(
        id_=node_id,
        text=f"content for {document_id}",
        embedding=embedding,
        metadata={
            "chat_id": chat_id,
            "document_id": document_id,
            "filename": f"{document_id}.txt",
            "file_type": "txt",
            "category": "general",
            "document_type": "document",
            "topics": json.dumps([]),
            "keywords": json.dumps([]),
            "chunk_id": node_id,
            "page_number": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )


class QueryEmbeddingProvider:
    async def embed(self, texts, expected_dimension=None):
        vector = [0.0] * 384
        vector[0] = 1.0
        return [vector.copy() for _ in texts], "fake-embedding-model"


@pytest.mark.asyncio
async def test_real_chroma_hydration_filtering_and_cleanup(tmp_path: Path):
    persist_directory = tmp_path / "chroma"
    stores = ChromaStores(persist_directory)
    matching = [0.0] * 384
    matching[0] = 1.0
    secondary = [0.0] * 384
    secondary[1] = 1.0
    stores.add_nodes(
        [
            _node("node-a", "chat-a", "doc-a", matching),
            _node("node-b", "chat-b", "doc-b", secondary),
        ]
    )
    assert stores.persistent_collection.count() == 2
    assert stores.ram_collection.count() == 2

    restarted = ChromaStores(persist_directory)
    assert restarted.ram_collection.count() == 0
    assert restarted.hydrate_ram() == 2

    results = await Retriever(restarted, QueryEmbeddingProvider()).retrieve(
        "chat-a", "content", {}, 40
    )
    assert [result.metadata["chat_id"] for result in results] == ["chat-a"]
    assert [result.metadata["document_id"] for result in results] == ["doc-a"]

    restarted.delete_document("doc-a")
    assert restarted.persistent_collection.count() == 1
    assert restarted.ram_collection.count() == 1
    restarted.delete_chat("chat-b")
    assert restarted.persistent_collection.count() == 0
    assert restarted.ram_collection.count() == 0


def test_chroma_rejects_switching_embedding_mode_with_existing_vectors(tmp_path: Path):
    stores = ChromaStores(tmp_path / "mode-chroma")
    vector = [0.0] * 384
    stores.ensure_embedding_mode("api")
    stores.add_nodes([_node("node", "chat", "doc", vector)])

    with pytest.raises(RuntimeError, match="api embeddings.*local mode"):
        stores.ensure_embedding_mode("local")
