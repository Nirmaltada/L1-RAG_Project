from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from backend/.env."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    query_rewrite_model: str = Field("openai/gpt-oss-20b", alias="QUERY_REWRITE_MODEL")
    embedding_model: str = Field("nvidia/nemotron-3-embed-1b:free", alias="EMBEDDING_MODEL")
    embedding_fallback_models: str = Field(
        "nvidia/llama-nemotron-embed-vl-1b-v2:free,liquid/lfm-2.5-embedding-350m:free",
        alias="EMBEDDING_FALLBACK_MODELS",
    )
    use_local_embeddings: bool = Field(False, alias="USE_LOCAL_EMBEDDINGS")
    local_embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="LOCAL_EMBEDDING_MODEL"
    )
    rerank_model: str = Field(
        "nvidia/llama-nemotron-rerank-vl-1b-v2:free", alias="RERANK_MODEL"
    )
    generation_model: str = Field("z-ai/glm-5.2:free", alias="GENERATION_MODEL")
    generation_fallback_model: str = Field(
        "minimax/minimax-m2.7:free", alias="GENERATION_FALLBACK_MODEL"
    )
    groq_fallback_model: str = Field("openai/gpt-oss-120b", alias="GROQ_FALLBACK_MODEL")

    vector_top_k: int = Field(40, alias="VECTOR_TOP_K")
    rerank_top_n: int = Field(8, alias="RERANK_TOP_N")
    embedding_batch_size: int = Field(16, alias="EMBEDDING_BATCH_SIZE")
    chunk_size: int = Field(300, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(40, alias="CHUNK_OVERLAP")
    chat_history_messages: int = Field(12, alias="CHAT_HISTORY_MESSAGES")
    context_relevance_threshold: float = Field(0.18, alias="CONTEXT_RELEVANCE_THRESHOLD")

    chroma_persist_directory: Path = Field(Path("./data/chroma"), alias="CHROMA_PERSIST_DIRECTORY")
    sqlite_database_path: Path = Field(Path("./data/app.db"), alias="SQLITE_DATABASE_PATH")
    upload_directory: Path = Field(Path("./data/uploads"), alias="UPLOAD_DIRECTORY")

    request_timeout_seconds: float = Field(60, alias="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(3, alias="MAX_RETRIES")
    debug_rag: bool = Field(False, alias="DEBUG_RAG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    def resolved_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    @property
    def database_path(self) -> Path:
        return self.resolved_path(self.sqlite_database_path)

    @property
    def uploads_path(self) -> Path:
        return self.resolved_path(self.upload_directory)

    @property
    def chroma_path(self) -> Path:
        return self.resolved_path(self.chroma_persist_directory)

    @property
    def embedding_models(self) -> list[str]:
        models = [self.embedding_model, *self.embedding_fallback_models.split(",")]
        cleaned = [model.strip() for model in models if model.strip()]
        return list(dict.fromkeys(cleaned))

    @property
    def embedding_mode(self) -> str:
        return "local" if self.use_local_embeddings else "api"

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
