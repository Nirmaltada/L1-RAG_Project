import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chats import router as chats_router
from app.api.documents import router as documents_router
from app.config import BACKEND_DIR, get_settings
from app.database import Database
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider
from app.rag.ingestion import IngestionService
from app.rag.embeddings import EmbeddingService
from app.rag.pipeline import RagPipeline
from app.rag.reranker import Reranker
from app.rag.retrieval import ChromaStores, Retriever

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database = Database(settings.database_path)
    await asyncio.to_thread(database.initialize)
    stores = ChromaStores(settings.chroma_path)
    await asyncio.to_thread(stores.ensure_embedding_mode, settings.embedding_mode)
    copied = await asyncio.to_thread(stores.hydrate_ram)
    openrouter = OpenRouterProvider(
        settings.openrouter_api_key,
        settings.request_timeout_seconds,
        settings.max_retries,
        settings.embedding_batch_size,
    )
    groq = GroqProvider(
        settings.groq_api_key,
        settings.request_timeout_seconds,
        settings.max_retries,
    )
    embeddings = EmbeddingService(
        openrouter=openrouter,
        api_models=settings.embedding_models,
        use_local=settings.use_local_embeddings,
        local_model=settings.local_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    retriever = Retriever(stores, embeddings)
    reranker = Reranker(openrouter, settings.rerank_model)
    ingestion = IngestionService(
        stores=stores,
        embeddings=embeddings,
        groq=groq,
        classification_model=settings.query_rewrite_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    pipeline = RagPipeline(settings, retriever, reranker, openrouter, groq)
    app.state.settings = settings
    app.state.db = database
    app.state.stores = stores
    app.state.openrouter = openrouter
    app.state.groq = groq
    app.state.embeddings = embeddings
    app.state.ingestion = ingestion
    app.state.ingestion_lock = asyncio.Lock()
    app.state.pipeline = pipeline
    logger.info(
        "Application started; hydrated_vectors=%s embedding_mode=%s",
        copied,
        settings.embedding_mode,
    )
    yield
    await openrouter.close()
    await groq.close()


app = FastAPI(title="Local RAG Chat", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chats_router)
app.include_router(documents_router)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "groq_configured": bool(settings.groq_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
        "embedding_mode": settings.embedding_mode,
    }


frontend_dir = BACKEND_DIR.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
