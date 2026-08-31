import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LocalEmbeddingError(RuntimeError):
    pass


class LocalEmbeddingProvider:
    """Lazily load a CPU SentenceTransformer only when local mode is used."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            self._model = await asyncio.to_thread(self._load_model)

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LocalEmbeddingError(
                "Local embeddings are enabled, but sentence-transformers is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        try:
            logger.info("Loading local embedding model=%s on CPU", self.model_name)
            return SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:
            raise LocalEmbeddingError(
                f"Could not load local embedding model {self.model_name}: {exc}"
            ) from exc

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vectors.tolist()
        except Exception as exc:
            raise LocalEmbeddingError(f"Local embedding failed: {exc}") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_loaded()
        return await asyncio.to_thread(self._encode, texts)
