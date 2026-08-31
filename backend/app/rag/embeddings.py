from app.providers.local_embeddings import LocalEmbeddingProvider
from app.providers.openrouter import OpenRouterProvider


class EmbeddingDimensionError(RuntimeError):
    pass


class EmbeddingService:
    """Select exactly one embedding mode; this is not a fallback router."""

    def __init__(
        self,
        openrouter: OpenRouterProvider,
        api_models: list[str],
        use_local: bool,
        local_model: str,
        batch_size: int,
    ) -> None:
        self.openrouter = openrouter
        self.api_models = api_models
        self.use_local = use_local
        self.local_model = local_model
        self.local = LocalEmbeddingProvider(local_model, batch_size)

    @property
    def mode(self) -> str:
        return "local" if self.use_local else "api"

    async def embed(
        self, texts: list[str], expected_dimension: int | None = None
    ) -> tuple[list[list[float]], str]:
        if self.use_local:
            vectors = await self.local.embed(texts)
            model = f"local:{self.local_model}"
        else:
            vectors, model = await self.openrouter.embed_with_fallbacks(
                texts,
                self.api_models,
                expected_dimension=expected_dimension,
            )

        if expected_dimension is not None and any(
            len(vector) != expected_dimension for vector in vectors
        ):
            actual = len(vectors[0]) if vectors else 0
            raise EmbeddingDimensionError(
                f"The selected {self.mode} embedding model returns {actual}-dimensional "
                f"vectors, but existing Chroma data uses {expected_dimension}. "
                "Delete the existing chats/documents and re-upload them after changing "
                "embedding mode."
            )
        return vectors, model
