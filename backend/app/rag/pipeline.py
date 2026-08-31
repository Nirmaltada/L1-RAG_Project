import asyncio
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.models import Message, RetrievedNode
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider, ProviderError
from app.rag.prompts import (
    REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    context_prompt,
    rewrite_user_prompt,
)
from app.rag.reranker import Reranker
from app.rag.retrieval import Retriever

logger = logging.getLogger(__name__)
PRIVATE_DOCUMENT_PATTERN = re.compile(
    r"\b(my|our|company(?:'s)?|uploaded|attached)\b.*\b(document|report|contract|notes|manual|dataset|file|policy)\b",
    re.IGNORECASE,
)
CURRENT_DOCUMENT_PATTERN = re.compile(
    r"\b(this|the|that|uploaded|attached)\s+(document|file|pdf|report|manual|guide|doc)\b",
    re.IGNORECASE,
)
MULTI_DOCUMENT_PATTERN = re.compile(
    r"\b(both|all|each|every|two|2)\b(?:\s+\w+){0,4}\s+\b(documents?|files?|pdfs?|attachments?)\b"
    r"|\b(both|each)\b"
    r"|\b(compare|comparison|contrast|differences?|similarities?)\b",
    re.IGNORECASE,
)
ORDINAL_DOCUMENT_PATTERN = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b"
    r"(?:\s+(?:uploaded|attached))?\s+(?:documents?|docs?|files?|pdfs?|attachments?|ones?)\b",
    re.IGNORECASE,
)
ORDINAL_VALUES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
DOCUMENT_MATCH_STOPWORDS = {
    "about", "attached", "document", "documents", "file", "files", "give", "main",
    "overview", "please", "talk", "talks", "this", "topics", "uploaded", "which", "write",
}


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        reranker: Reranker,
        openrouter: OpenRouterProvider,
        groq: GroqProvider,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.openrouter = openrouter
        self.groq = groq

    @staticmethod
    def _history(history: list[Message]) -> list[dict[str, str]]:
        return [{"role": item.role, "content": item.content} for item in history]

    async def rewrite_query(
        self, question: str, history: list[dict[str, str]], documents: list[dict] | None = None
    ) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        try:
            result = await self.groq.json_completion(
                self.settings.query_rewrite_model,
                REWRITE_SYSTEM_PROMPT,
                rewrite_user_prompt(question, history, documents),
            )
            query = str(result.get("query") or question).strip()
            categories = result.get("likely_categories")
            topics = result.get("topics")
            intent = {
                "likely_categories": categories if isinstance(categories, list) else [],
                "topics": topics if isinstance(topics, list) else [],
            }
            return query, intent
        except Exception:
            logger.exception("Query rewriting failed; using original question")
            return question, {"likely_categories": [], "topics": []}
        finally:
            logger.info("Query rewrite latency_ms=%.1f", (time.perf_counter() - started) * 1000)

    @staticmethod
    def _document_scope(question: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        folded = question.casefold()
        explicit: list[dict[str, Any]] = []
        for document in documents:
            filename = str(document.get("filename", ""))
            stem = filename.rsplit(".", 1)[0]
            if filename.casefold() in folded or (len(stem) >= 3 and stem.casefold() in folded):
                explicit.append(document)
        if explicit:
            return explicit
        if MULTI_DOCUMENT_PATTERN.search(question):
            return documents

        ordinal = ORDINAL_DOCUMENT_PATTERN.search(question)
        if ordinal:
            value = ordinal.group(1).casefold()
            position = ORDINAL_VALUES.get(value)
            if position is None:
                position = int(re.match(r"\d+", value).group())
            if 1 <= position <= len(documents):
                return [documents[position - 1]]

        # Resolve descriptions such as "the linear algebra notes" from the
        # stored metadata. Only select a document when it is the unique best
        # lexical match; otherwise leave semantic retrieval chat-wide.
        question_terms = {
            term for term in re.findall(r"[a-z0-9]+", folded)
            if len(term) >= 3 and term not in DOCUMENT_MATCH_STOPWORDS
        }
        scored: list[tuple[int, dict[str, Any]]] = []
        for document in documents:
            searchable = " ".join(
                [
                    str(document.get("filename", "")),
                    str(document.get("category", "")),
                    str(document.get("document_type", "")),
                    *(str(item) for item in document.get("topics") or []),
                    *(str(item) for item in document.get("keywords") or []),
                ]
            ).casefold()
            metadata_terms = set(re.findall(r"[a-z0-9]+", searchable))
            score = len(question_terms & metadata_terms)
            if score:
                scored.append((score, document))
        if scored:
            best_score = max(score for score, _ in scored)
            best = [document for score, document in scored if score == best_score]
            if len(best) == 1:
                return best

        # list_documents returns newest first. A singular deictic reference
        # ("this document") therefore means the most recently attached file.
        if CURRENT_DOCUMENT_PATTERN.search(question) and documents:
            return documents[:1]
        return []

    @staticmethod
    def _document_inventory(documents: list[dict[str, Any]]) -> str:
        if not documents:
            return "No documents are attached to this chat."
        lines = ["Authoritative uploaded-document inventory (one line equals one document):"]
        for document in documents:
            details = [str(document.get("filename", "document"))]
            category = str(document.get("category", "")).strip()
            document_type = str(document.get("document_type", "")).strip()
            if category:
                details.append(f"category={category}")
            if document_type:
                details.append(f"type={document_type}")
            lines.append("- " + " | ".join(details))
        lines.append(
            "Use only these entries when counting or naming uploaded documents. "
            "Pages, chunks, citations, titles mentioned in chat history, and source numbers are not separate documents."
        )
        return "\n".join(lines)

    def context_is_relevant(self, nodes: list[RetrievedNode]) -> bool:
        if not nodes:
            return False
        threshold = self.settings.context_relevance_threshold
        for node in nodes:
            if node.reranker_score is not None and node.reranker_score >= threshold:
                return True
            if node.reranker_score is None and (
                node.vector_score >= threshold or (
                    node.vector_score >= threshold * 0.7 and node.metadata_score >= 0.25
                )
            ):
                return True
        return False

    @staticmethod
    def _sources_and_context(
        nodes: list[RetrievedNode],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        sources: list[dict[str, Any]] = []
        blocks: list[str] = []
        source_indexes: dict[tuple[str, int | None], int] = {}
        for node in nodes:
            metadata = node.metadata
            document_id = str(metadata.get("document_id", ""))
            raw_page = metadata.get("page_number")
            page = int(raw_page) if raw_page not in (None, "", 0, "0") else None
            key = (document_id, page)
            if key not in source_indexes:
                index = len(sources) + 1
                source_indexes[key] = index
                sources.append(
                    {
                        "index": index,
                        "document_id": document_id,
                        "filename": str(metadata.get("filename", "document")),
                        "page": page,
                        "category": str(metadata.get("category", "general")),
                        "vector_score": round(node.vector_score, 6),
                        "reranker_score": (
                            round(node.reranker_score, 6)
                            if node.reranker_score is not None
                            else None
                        ),
                    }
                )
            index = source_indexes[key]
            page_label = f", page {page}" if page else ""
            blocks.append(
                f"[Source {index}: {metadata.get('filename', 'document')}{page_label}]\n{node.text}"
            )
        return sources, blocks

    async def _stream_with_fallbacks(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[tuple[str, str]]:
        providers = [
            (
                self.settings.generation_model,
                lambda: self.openrouter.stream_chat(self.settings.generation_model, messages),
            ),
            (
                self.settings.generation_fallback_model,
                lambda: self.openrouter.stream_chat(self.settings.generation_fallback_model, messages),
            ),
            (
                self.settings.groq_fallback_model,
                lambda: self.groq.stream_chat(self.settings.groq_fallback_model, messages),
            ),
        ]
        errors: list[str] = []
        for model, factory in providers:
            stream = factory()
            try:
                first = await anext(stream)
            except (Exception, StopAsyncIteration) as exc:
                logger.warning("Generation model %s failed before output: %s", model, exc)
                errors.append(f"{model}: {exc}")
                continue
            logger.info("Generation model=%s", model)
            yield model, first
            try:
                async for token in stream:
                    yield model, token
                return
            except Exception as exc:
                # Switching models after partial output would produce a duplicated answer.
                raise ProviderError(f"{model} failed after streaming began: {exc}") from exc
        raise ProviderError("All answer models failed. " + " | ".join(errors))

    async def answer(
        self,
        chat_id: str,
        question: str,
        history: list[Message],
        documents: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        request_id = str(uuid.uuid4())
        recent = self._history(history[-self.settings.chat_history_messages :])
        document_catalog = documents or []
        documents_exist = bool(document_catalog)
        selected: list[RetrievedNode] = []
        sources: list[dict[str, Any]] = []
        rag_used = False
        asks_current_document = bool(CURRENT_DOCUMENT_PATTERN.search(question))
        scoped_documents = self._document_scope(question, document_catalog)
        logger.info(
            "Document routing chat_id=%s request_id=%s inventory=%s scope=%s",
            chat_id,
            request_id,
            len(document_catalog),
            [str(document.get("filename", "document")) for document in scoped_documents]
            or "chat-wide",
        )

        if documents_exist:
            yield {"type": "status", "message": "Searching documents…"}
            rewritten, intent = await self.rewrite_query(question, recent, document_catalog)
            started = time.perf_counter()
            try:
                query = rewritten
                if scoped_documents:
                    query += "\n\nTarget uploaded documents:\n" + "\n".join(
                        str(document.get("filename", "document")) for document in scoped_documents
                    )
                candidates = await self.retriever.retrieve(
                    chat_id,
                    query,
                    intent,
                    self.settings.vector_top_k,
                    [str(document["id"]) for document in scoped_documents] or None,
                )
            except Exception:
                logger.exception("Retrieval failed chat_id=%s request_id=%s", chat_id, request_id)
                candidates = []
            logger.info(
                "Retrieval chat_id=%s request_id=%s candidates=%s latency_ms=%.1f",
                chat_id,
                request_id,
                len(candidates),
                (time.perf_counter() - started) * 1000,
            )
            started = time.perf_counter()
            if len(scoped_documents) > 1:
                per_document_top_n = max(
                    1, math.ceil(self.settings.rerank_top_n / len(scoped_documents))
                )
                groups = [
                    [
                        candidate for candidate in candidates
                        if candidate.metadata.get("document_id") == document.get("id")
                    ]
                    for document in scoped_documents
                ]
                reranked_groups = await asyncio.gather(
                    *(
                        self.reranker.rerank(rewritten, group, per_document_top_n)
                        for group in groups
                    )
                )
                selected = [node for group in reranked_groups for node in group]
                selected.sort(key=lambda node: node.best_score, reverse=True)
            else:
                selected = await self.reranker.rerank(
                    rewritten, candidates, self.settings.rerank_top_n
                )
            logger.info(
                "Reranking request_id=%s selected=%s latency_ms=%.1f",
                request_id,
                len(selected),
                (time.perf_counter() - started) * 1000,
            )
            rag_used = self.context_is_relevant(selected) or (
                bool(scoped_documents) and bool(selected)
            )

        system = SYSTEM_PROMPT + "\n\n" + self._document_inventory(document_catalog)
        if documents_exist and not rag_used:
            system += (
                "\nDocuments exist in this chat, but no sufficiently relevant passage was found. "
                "Do not say no document is attached. If the user asks about 'this document' or the uploaded file, "
                "say you can see the uploaded document but could not find enough relevant text to answer confidently."
            )
        elif not documents_exist and PRIVATE_DOCUMENT_PATTERN.search(question):
            system += "\nThe requested private/specific material is not available in this chat; mention uploading it."

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if rag_used:
            sources, blocks = self._sources_and_context(selected)
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant retrieved context follows. Cite it using [1], [2], etc.\n\n"
                    + context_prompt(blocks),
                }
            )
        messages.extend(recent)
        messages.append({"role": "user", "content": question})
        yield {"type": "status", "message": "Writing answer…"}

        model_used = ""
        async for model, token in self._stream_with_fallbacks(messages):
            model_used = model
            yield {"type": "delta", "content": token}
        yield {
            "type": "done",
            "sources": sources if rag_used else [],
            "rag_used": rag_used,
            "model": model_used,
            "request_id": request_id,
        }
