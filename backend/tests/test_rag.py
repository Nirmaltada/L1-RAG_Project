import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from app.config import Settings
from app.models import RetrievedNode
from app.rag.ingestion import IngestionService
from app.rag.embeddings import EmbeddingService
from app.rag.pipeline import RagPipeline
from app.rag.reranker import Reranker
from app.rag.retrieval import ChromaStores, Retriever


class FakeOpenRouter:
    def __init__(self) -> None:
        self.embedding_calls = 0
        self.rerank_calls = 0

    async def embed(self, texts, expected_dimension=None):
        self.embedding_calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(word in lowered for word in ("api", "authentication", "technical")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(word in lowered for word in ("biography", "university", "person")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors, "fake-embedding-model"

    async def rerank(self, query, documents, model, top_n):
        self.rerank_calls += 1
        return [
            {"index": index, "relevance_score": 0.9 - index * 0.01}
            for index in range(min(top_n, len(documents)))
        ]

    async def stream_chat(self, model, messages, temperature=0.2):
        yield "Generated answer"


class FakeGroq:
    async def json_completion(self, model, system_prompt, user_prompt):
        return {"query": "API authentication", "likely_categories": ["technology"], "topics": ["API"]}

    async def stream_chat(self, model, messages, temperature=0.2):
        yield "Groq answer"


class ClassificationGroq:
    async def json_completion(self, model, system_prompt, user_prompt):
        return {
            "category": "technology",
            "document_type": "technical_documentation",
            "topics": ["FastAPI", "authentication"],
            "keywords": ["JWT", "middleware"],
        }


@pytest.mark.asyncio
async def test_embedding_mode_true_uses_only_local_provider():
    class ApiProvider:
        async def embed_with_fallbacks(self, *args, **kwargs):
            raise AssertionError("API embeddings must not run in local mode")

    class LocalProvider:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    service = EmbeddingService(ApiProvider(), ["api"], True, "local-model", 4)
    service.local = LocalProvider()
    vectors, model = await service.embed(["hello"])
    assert vectors == [[1.0, 0.0]]
    assert model == "local:local-model"


@pytest.mark.asyncio
async def test_embedding_mode_false_uses_only_api_providers():
    class ApiProvider:
        async def embed_with_fallbacks(self, texts, models, expected_dimension=None):
            return [[0.0, 1.0] for _ in texts], models[0]

    class LocalProvider:
        async def embed(self, texts):
            raise AssertionError("Local embeddings must not run in API mode")

    service = EmbeddingService(ApiProvider(), ["api-primary"], False, "local-model", 4)
    service.local = LocalProvider()
    vectors, model = await service.embed(["hello"])
    assert vectors == [[0.0, 1.0]]
    assert model == "api-primary"


def make_node(node_id, chat_id, document_id, text, vector, category):
    return TextNode(
        id_=node_id,
        text=text,
        embedding=vector,
        metadata={
            "chat_id": chat_id,
            "document_id": document_id,
            "filename": f"{document_id}.txt",
            "file_type": "txt",
            "category": category,
            "document_type": "document",
            "topics": json.dumps([category]),
            "keywords": json.dumps([]),
            "chunk_id": node_id,
            "page_number": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )


@pytest.mark.asyncio
async def test_chat_isolation_and_metadata_survive_retrieval(tmp_path: Path):
    all_nodes = [
        make_node("tech", "chat-a", "technical", "API authentication middleware", [1.0, 0.0, 0.0], "technology"),
        make_node("bio", "chat-b", "biography", "The person attended a university", [0.0, 1.0, 0.0], "biography"),
    ]

    class FilteringVectorStore:
        def query(self, query):
            chat_id = query.filters.filters[0].value
            selected = [node for node in all_nodes if node.metadata["chat_id"] == chat_id]
            return VectorStoreQueryResult(nodes=selected, similarities=[0.95] * len(selected), ids=[n.node_id for n in selected])

    stores = SimpleNamespace(ram_store=FilteringVectorStore(), embedding_dimension=lambda: None)
    provider = FakeOpenRouter()
    retriever = Retriever(stores, provider)
    results = await retriever.retrieve(
        "chat-a", "API authentication", {"likely_categories": ["technology"], "topics": ["API"]}, 40
    )

    assert results
    assert {result.metadata["chat_id"] for result in results} == {"chat-a"}
    assert results[0].metadata["category"] == "technology"
    assert results[0].metadata["topics"] == ["technology"]


@pytest.mark.asyncio
async def test_metadata_survives_ingestion_retrieval_and_reranking(tmp_path: Path):
    class CapturingStores:
        def __init__(self):
            self.nodes = []
            self.ram_store = SimpleNamespace(query=self.query)

        def add_nodes(self, nodes):
            self.nodes = nodes

        def query(self, query):
            return VectorStoreQueryResult(
                nodes=self.nodes,
                similarities=[0.91] * len(self.nodes),
                ids=[node.node_id for node in self.nodes],
            )

        def embedding_dimension(self):
            return None

    path = tmp_path / "backend_notes.txt"
    path.write_text(
        "The FastAPI backend validates JWT tokens in authentication middleware before protected endpoints.",
        encoding="utf-8",
    )
    stores, provider = CapturingStores(), FakeOpenRouter()
    ingestion = IngestionService(
        stores, provider, ClassificationGroq(), "classify", 300, 40
    )
    record = await ingestion.ingest(
        "chat-a", "doc-a", path.name, "stored.txt", "hash", path
    )
    assert record.category == "technology"
    assert json.loads(stores.nodes[0].metadata["topics"]) == ["FastAPI", "authentication"]

    retrieved = await Retriever(stores, provider).retrieve(
        "chat-a", "FastAPI authentication", {"topics": ["authentication"]}, 40
    )
    reranked = await Reranker(provider, "rerank").rerank(
        "FastAPI authentication", retrieved, 8
    )
    assert reranked[0].metadata["document_id"] == "doc-a"
    assert reranked[0].metadata["topics"] == ["FastAPI", "authentication"]
    assert reranked[0].reranker_score == pytest.approx(0.9)


def test_hydration_reuses_stored_embeddings_without_provider(tmp_path: Path):
    node = make_node("saved", "chat-a", "doc", "saved vector", [0.2, 0.3, 0.5], "general")

    class PersistentCollection:
        def count(self):
            return 1

        def get(self, **kwargs):
            return {
                "ids": [node.node_id],
                "embeddings": [node.embedding],
                "documents": [node.text],
                "metadatas": [node.metadata],
            }

    class RamCollection:
        def __init__(self):
            self.saved = None

        def upsert(self, **kwargs):
            self.saved = kwargs

    provider = FakeOpenRouter()
    restarted = ChromaStores.__new__(ChromaStores)
    restarted.persistent_collection = PersistentCollection()
    restarted.ram_collection = RamCollection()
    assert restarted.hydrate_ram(batch_size=1) == 1
    assert restarted.ram_collection.saved["embeddings"] == [[0.2, 0.3, 0.5]]
    assert provider.embedding_calls == 0


@pytest.mark.asyncio
async def test_retrieval_respects_requested_top_40(tmp_path: Path):
    captured = {}

    def fake_query(query):
        captured["top_k"] = query.similarity_top_k
        captured["filter"] = query.filters.filters[0].value
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

    stores = SimpleNamespace(
        ram_store=SimpleNamespace(query=fake_query),
        embedding_dimension=lambda: None,
    )
    retriever = Retriever(stores, FakeOpenRouter())
    await retriever.retrieve("only-this-chat", "technical query", {}, 40)
    assert captured == {"top_k": 40, "filter": "only-this-chat"}


@pytest.mark.asyncio
async def test_retrieval_balances_scoped_documents_with_one_query_embedding():
    calls = []
    nodes = {
        "doc-a": make_node("a", "chat", "doc-a", "biology text", [0.0, 0.0, 1.0], "biology"),
        "doc-b": make_node("b", "chat", "doc-b", "physics text", [0.0, 0.0, 1.0], "physics"),
    }

    def fake_query(query):
        filters = {item.key: item.value for item in query.filters.filters}
        calls.append((filters, query.similarity_top_k, query.filters.condition.value))
        node = nodes[filters["document_id"]]
        return VectorStoreQueryResult(nodes=[node], similarities=[0.9], ids=[node.node_id])

    provider = FakeOpenRouter()
    stores = SimpleNamespace(
        ram_store=SimpleNamespace(query=fake_query),
        embedding_dimension=lambda: None,
    )
    results = await Retriever(stores, provider).retrieve(
        "chat", "categorize both documents", {}, 40, ["doc-a", "doc-b"]
    )

    assert provider.embedding_calls == 1
    assert {call[0]["document_id"] for call in calls} == {"doc-a", "doc-b"}
    assert all(call[0]["chat_id"] == "chat" and call[2] == "and" for call in calls)
    assert {result.metadata["document_id"] for result in results} == {"doc-a", "doc-b"}


@pytest.mark.asyncio
async def test_reranker_keeps_top_8_and_metadata():
    provider = FakeOpenRouter()
    candidates = [
        RetrievedNode(
            node_id=str(index),
            text=f"candidate {index}",
            metadata={"filename": "report.pdf", "category": "finance", "topics": ["revenue"]},
            vector_score=0.8 - index * 0.01,
        )
        for index in range(40)
    ]
    results = await Reranker(provider, "rerank-model").rerank("revenue", candidates, 8)
    assert len(results) == 8
    assert all(result.metadata["filename"] == "report.pdf" for result in results)
    assert all(result.reranker_score is not None for result in results)


class FixedRetriever:
    def __init__(self, nodes):
        self.nodes = nodes
        self.top_k = None

        self.document_ids = None

    async def retrieve(self, chat_id, query, intent, top_k, document_ids=None):
        self.top_k = top_k
        self.document_ids = document_ids
        return self.nodes


class FixedReranker:
    def __init__(self):
        self.top_n = None

    async def rerank(self, query, candidates, top_n):
        self.top_n = top_n
        return candidates[:top_n]


def pipeline_settings(tmp_path: Path) -> Settings:
    return Settings(
        GROQ_API_KEY="test",
        OPENROUTER_API_KEY="test",
        SQLITE_DATABASE_PATH=tmp_path / "app.db",
        CHROMA_PERSIST_DIRECTORY=tmp_path / "chroma",
        UPLOAD_DIRECTORY=tmp_path / "uploads",
    )


async def collect_events(pipeline, documents):
    return [event async for event in pipeline.answer("chat", "What is it?", [], documents)]


def document(document_id: str, filename: str, category: str = "general"):
    return {
        "id": document_id,
        "filename": filename,
        "category": category,
        "document_type": "document",
        "topics": [],
        "keywords": [],
    }


@pytest.mark.asyncio
async def test_general_chat_when_no_documents_has_no_citations(tmp_path: Path):
    retriever, reranker = FixedRetriever([]), FixedReranker()
    pipeline = RagPipeline(pipeline_settings(tmp_path), retriever, reranker, FakeOpenRouter(), FakeGroq())
    events = await collect_events(pipeline, [])
    done = events[-1]
    assert done["rag_used"] is False
    assert done["sources"] == []
    assert retriever.top_k is None


@pytest.mark.asyncio
async def test_irrelevant_documents_fall_back_without_fake_citations(tmp_path: Path):
    node = RetrievedNode("n", "unrelated", {"document_id": "d", "filename": "x.txt"}, 0.01, 0.01)
    retriever, reranker = FixedRetriever([node]), FixedReranker()
    pipeline = RagPipeline(pipeline_settings(tmp_path), retriever, reranker, FakeOpenRouter(), FakeGroq())
    events = await collect_events(pipeline, [document("d", "x.txt")])
    done = events[-1]
    assert done["rag_used"] is False
    assert done["sources"] == []


@pytest.mark.asyncio
async def test_relevant_document_produces_sources_and_pipeline_limits(tmp_path: Path):
    node = RetrievedNode(
        "n", "JWT is checked in middleware.",
        {"document_id": "d", "filename": "architecture.pdf", "page_number": 14, "category": "technology"},
        0.8, 0.92,
    )
    retriever, reranker = FixedRetriever([node]), FixedReranker()
    pipeline = RagPipeline(pipeline_settings(tmp_path), retriever, reranker, FakeOpenRouter(), FakeGroq())
    events = await collect_events(pipeline, [document("d", "architecture.pdf")])
    done = events[-1]
    assert retriever.top_k == 40
    assert reranker.top_n == 8
    assert done["rag_used"] is True
    assert done["sources"][0]["filename"] == "architecture.pdf"
    assert done["sources"][0]["page"] == 14


def test_document_scope_uses_inventory_instead_of_chat_hallucinations():
    documents = [
        document("bio", "gkae346.pdf", "biology"),
        document("physics", "2125.pdf", "physics"),
    ]
    assert [item["id"] for item in RagPipeline._document_scope("compare both", documents)] == [
        "bio",
        "physics",
    ]
    assert [item["id"] for item in RagPipeline._document_scope("explain 2125.pdf", documents)] == [
        "physics"
    ]
    assert [item["id"] for item in RagPipeline._document_scope("brief this document", documents)] == [
        "bio"
    ]


def test_document_scope_resolves_ordinals_and_metadata_descriptions():
    documents = [
        document("quality", "cnotes8.pdf", "Software Quality Assurance"),
        document("math", "eigenvalues.pdf", "Linear Algebra"),
    ]

    assert [item["id"] for item in RagPipeline._document_scope("overview of the 2nd doc", documents)] == [
        "math"
    ]
    assert [item["id"] for item in RagPipeline._document_scope("topics in the linear math document", documents)] == [
        "math"
    ]


@pytest.mark.asyncio
async def test_resolved_document_uses_its_chunks_even_when_relevance_score_is_low(tmp_path: Path):
    node = RetrievedNode(
        "math-node",
        "For Ax = lambda x, lambda is an eigenvalue and x is an eigenvector.",
        {"document_id": "math", "filename": "eigenvalues.pdf", "page_number": 1},
        0.01,
        0.01,
    )
    retriever, reranker = FixedRetriever([node]), FixedReranker()
    pipeline = RagPipeline(
        pipeline_settings(tmp_path), retriever, reranker, FakeOpenRouter(), FakeGroq()
    )
    documents = [
        document("quality", "cnotes8.pdf", "Software Quality Assurance"),
        document("math", "eigenvalues.pdf", "Linear Algebra"),
    ]

    events = [
        event
        async for event in pipeline.answer(
            "chat", "Give an overview of the 2nd doc and its main equation", [], documents
        )
    ]

    assert retriever.document_ids == ["math"]
    assert events[-1]["rag_used"] is True
    assert events[-1]["sources"][0]["filename"] == "eigenvalues.pdf"


@pytest.mark.asyncio
async def test_multi_document_question_keeps_evidence_from_every_document(tmp_path: Path):
    nodes = [
        RetrievedNode(
            "bio-node",
            "GPS-SUMO predicts SUMOylation sites.",
            {"document_id": "bio", "filename": "gkae346.pdf", "page_number": 1},
            0.91,
            0.90,
        ),
        RetrievedNode(
            "physics-node",
            "This paper discusses legal requirements for time and frequency.",
            {"document_id": "physics", "filename": "2125.pdf", "page_number": 1},
            0.89,
            0.88,
        ),
    ]
    retriever, reranker = FixedRetriever(nodes), FixedReranker()
    pipeline = RagPipeline(
        pipeline_settings(tmp_path), retriever, reranker, FakeOpenRouter(), FakeGroq()
    )
    documents = [
        document("bio", "gkae346.pdf", "biology"),
        document("physics", "2125.pdf", "physics"),
    ]

    events = [
        event
        async for event in pipeline.answer(
            "chat", "Which categories do both belong to?", [], documents
        )
    ]

    assert retriever.document_ids == ["bio", "physics"]
    assert {source["document_id"] for source in events[-1]["sources"]} == {
        "bio",
        "physics",
    }
