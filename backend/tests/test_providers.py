from types import SimpleNamespace

import httpx
import pytest

from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider, ProviderError


async def _provider(handler, retries: int = 1) -> OpenRouterProvider:
    provider = OpenRouterProvider("key", 5, retries)
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url=provider.BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    return provider


@pytest.mark.asyncio
async def test_openrouter_embedding_response_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    provider = await _provider(handler)
    try:
        assert await provider.embed(["first", "second"], "embed") == [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openrouter_embedding_batch_size_is_configurable():
    batch_sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        input_count = payload.count(b"item-")
        batch_sizes.append(input_count)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(index)]}
                    for index in range(input_count)
                ]
            },
        )

    provider = OpenRouterProvider("key", 5, 1, embedding_batch_size=2)
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url=provider.BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await provider.embed(["item-1", "item-2", "item-3"], "embed") == [
            [0.0],
            [1.0],
            [0.0],
        ]
        assert batch_sizes == [2, 1]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openrouter_uses_retry_after_for_retryable_status(monkeypatch):
    calls = 0
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("app.providers.openrouter.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    provider = await _provider(handler, retries=2)
    try:
        assert await provider.embed(["text"], "embed") == [[1.0]]
        assert sleeps == [7.0]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openrouter_embedding_fallback_models_are_tried_in_order():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        calls.append(payload)
        if b"primary" in payload:
            return httpx.Response(429)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    provider = await _provider(handler)
    try:
        embeddings, model = await provider.embed_with_fallbacks(
            ["text"],
            ["primary", "fallback"],
        )
        assert embeddings == [[1.0, 2.0]]
        assert model == "fallback"
        assert len(calls) == 2
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openrouter_embedding_fallback_skips_wrong_dimension():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        embedding = [1.0] if b"wrong" in payload else [1.0, 2.0]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": embedding}]})

    provider = await _provider(handler)
    try:
        embeddings, model = await provider.embed_with_fallbacks(
            ["text"],
            ["wrong", "right"],
            expected_dimension=2,
        )
        assert embeddings == [[1.0, 2.0]]
        assert model == "right"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openrouter_sse_accepts_optional_space_and_requires_done_marker():
    completed = await _provider(
        lambda request: httpx.Response(
            200,
            text='data:{"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )
    try:
        assert [token async for token in completed.stream_chat("model", [])] == ["hello"]
    finally:
        await completed.close()

    truncated = await _provider(
        lambda request: httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )
    try:
        stream = truncated.stream_chat("model", [])
        assert await anext(stream) == "partial"
        with pytest.raises(ProviderError, match="before the completion marker"):
            await anext(stream)
    finally:
        await truncated.close()


@pytest.mark.asyncio
async def test_openrouter_does_not_retry_non_retryable_status():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = await _provider(handler, retries=3)
    try:
        with pytest.raises(ProviderError):
            await provider._post("/embeddings", {"input": ["text"], "model": "embed"})
        assert calls == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_openrouter_retries_transport_errors_and_zero_means_one_attempt():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    provider = await _provider(handler, retries=2)
    try:
        with pytest.raises(ProviderError):
            await provider._post("/embeddings", {"input": ["text"], "model": "embed"})
        assert calls == 2
    finally:
        await provider.close()

    calls = 0
    provider = await _provider(handler, retries=0)
    try:
        with pytest.raises(ProviderError):
            await provider._post("/embeddings", {"input": ["text"], "model": "embed"})
        assert calls == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_groq_stream_ignores_chunks_without_choices():
    async def chunks():
        yield SimpleNamespace(choices=[])
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))]
        )

    class Completions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return chunks()

    provider = GroqProvider.__new__(GroqProvider)
    provider.api_key = "key"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    assert [token async for token in provider.stream_chat("model", [])] == ["answer"]
