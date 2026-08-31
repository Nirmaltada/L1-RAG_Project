import logging

from app.models import RetrievedNode
from app.providers.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, provider: OpenRouterProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def rerank(
        self, query: str, candidates: list[RetrievedNode], top_n: int
    ) -> list[RetrievedNode]:
        if not candidates:
            return []
        try:
            results = await self.provider.rerank(
                query, [candidate.text for candidate in candidates], self.model, top_n
            )
            selected: list[RetrievedNode] = []
            for result in results:
                index = int(result["index"])
                if index < 0 or index >= len(candidates):
                    continue
                candidate = candidates[index]
                candidate.reranker_score = float(result.get("relevance_score", 0.0))
                selected.append(candidate)
            selected.sort(
                key=lambda item: item.best_score + (0.05 * item.metadata_score), reverse=True
            )
            return selected[:top_n]
        except Exception:
            logger.exception("Reranker unavailable; using vector-ranked candidates")
            return candidates[:top_n]
