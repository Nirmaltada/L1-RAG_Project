import json
from collections.abc import AsyncIterator
from typing import Any

from groq import AsyncGroq

from app.providers.openrouter import ProviderError


class GroqProvider:
    def __init__(self, api_key: str, timeout: float, max_retries: int) -> None:
        self.api_key = api_key
        self.client = AsyncGroq(api_key=api_key or "missing", timeout=timeout, max_retries=max_retries)

    async def close(self) -> None:
        await self.client.close()

    async def json_completion(
        self, model: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError("GROQ_API_KEY is not configured")
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("JSON response was not an object")
            return parsed
        except Exception as exc:
            raise ProviderError(f"Groq structured request failed: {exc}") from exc

    async def stream_chat(
        self, model: str, messages: list[dict[str, str]], temperature: float = 0.2
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise ProviderError("GROQ_API_KEY is not configured")
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            raise ProviderError(f"Groq streaming request failed: {exc}") from exc
