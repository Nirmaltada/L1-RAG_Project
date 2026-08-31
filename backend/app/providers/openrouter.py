import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


class OpenRouterProvider:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        timeout: float,
        max_retries: int,
        embedding_batch_size: int = 16,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.embedding_batch_size = max(1, min(64, embedding_batch_size))
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5500",
                "X-Title": "Local RAG Chat",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
        retry_after = response.headers.get("retry-after") if response else None
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    return min(30.0, max(0.0, (retry_at - datetime.now(UTC)).total_seconds()))
                except (TypeError, ValueError):
                    pass
        return min(2**attempt, 8)

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY is not configured")
        last_error: Exception | None = None
        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            try:
                response = await self.client.post(endpoint, json=payload)
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"temporary OpenRouter error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise json.JSONDecodeError("response was not a JSON object", response.text, 0)
                return data
            except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in {408, 409, 429} or exc.response.status_code >= 500
                )
                if not retryable or attempt + 1 >= attempts:
                    break
                response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
                await asyncio.sleep(self._retry_delay(attempt, response))
        raise ProviderError(f"OpenRouter request failed: {last_error}") from last_error

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            data = await self._post("/embeddings", {"model": model, "input": batch})
            for item in data.get("data", []):
                try:
                    index = int(item["index"])
                    embedding = item["embedding"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ProviderError("OpenRouter returned an invalid embedding item") from exc
                if not 0 <= index < len(batch) or not isinstance(embedding, list):
                    raise ProviderError("OpenRouter returned an invalid embedding item")
                embeddings[start + index] = embedding
        if any(item is None for item in embeddings):
            raise ProviderError("OpenRouter returned an incomplete embedding batch")
        return [item for item in embeddings if item is not None]

    async def embed_with_fallbacks(
        self,
        texts: list[str],
        models: list[str],
        expected_dimension: int | None = None,
    ) -> tuple[list[list[float]], str]:
        if not texts:
            return [], models[0] if models else ""
        if not models:
            raise ProviderError("No OpenRouter embedding models are configured")
        last_error: Exception | None = None
        for model in models:
            try:
                embeddings = await self.embed(texts, model)
            except ProviderError as exc:
                last_error = exc
                logger.warning("OpenRouter embedding model %s failed: %s", model, exc)
                continue
            if expected_dimension and any(
                len(embedding) != expected_dimension for embedding in embeddings
            ):
                last_error = ProviderError(
                    f"embedding model {model} returned vectors that do not match "
                    f"the existing Chroma dimension {expected_dimension}"
                )
                logger.warning("%s", last_error)
                continue
            return embeddings, model
        raise ProviderError(f"All OpenRouter embedding models failed: {last_error}") from last_error

    async def rerank(
        self, query: str, documents: list[str], model: str, top_n: int
    ) -> list[dict[str, Any]]:
        data = await self._post(
            "/rerank",
            {"model": model, "query": query, "documents": documents, "top_n": top_n},
        )
        results = data.get("results")
        if not isinstance(results, list):
            raise ProviderError("OpenRouter returned an invalid rerank response")
        return results

    async def stream_chat(
        self, model: str, messages: list[dict[str, str]], temperature: float = 0.2
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY is not configured")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        last_error: Exception | None = None
        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            emitted = False
            done_received = False
            try:
                async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                    if response.status_code in {408, 409, 429} or response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"temporary OpenRouter error {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            done_received = True
                            return
                        try:
                            data = json.loads(raw)
                            if not isinstance(data, dict):
                                continue
                            if data.get("error"):
                                raise ProviderError(f"OpenRouter stream error: {data['error']}")
                            delta = data["choices"][0]["delta"].get("content")
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue
                        if delta:
                            emitted = True
                            yield delta
                    if not done_received:
                        raise ProviderError("OpenRouter stream ended before the completion marker")
            except (httpx.HTTPError, ProviderError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in {408, 409, 429} or exc.response.status_code >= 500
                )
                # Retrying after output would duplicate the beginning of the answer.
                if emitted or not retryable or attempt + 1 >= attempts:
                    break
                response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
                await asyncio.sleep(self._retry_delay(attempt, response))
        raise ProviderError(f"OpenRouter streaming request failed: {last_error}") from last_error
