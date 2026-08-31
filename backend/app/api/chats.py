import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.database import Database
from app.rag.pipeline import RagPipeline
from app.schemas import ChatCreate, ChatDetail, ChatSummary, ChatUpdate, MessageCreate

router = APIRouter(prefix="/api/chats", tags=["chats"])
logger = logging.getLogger(__name__)


def _db(request: Request) -> Database:
    return request.app.state.db


def _pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


def _chat_summary(chat: object) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
    }


def make_title(message: str) -> str:
    cleaned = re.sub(r"\s+", " ", message).strip()
    cleaned = re.sub(r"^(can you|could you|please|tell me|explain)\s+", "", cleaned, flags=re.I)
    cleaned = cleaned.rstrip("?.!")
    if len(cleaned) > 48:
        cleaned = cleaned[:48].rsplit(" ", 1)[0] + "…"
    return (cleaned or "New chat").capitalize()


@router.get("", response_model=list[ChatSummary])
async def list_chats(request: Request):
    return [_chat_summary(chat) for chat in await asyncio.to_thread(_db(request).list_chats)]


@router.post("", response_model=ChatSummary, status_code=201)
async def create_chat(payload: ChatCreate, request: Request):
    chat = await asyncio.to_thread(_db(request).create_chat, payload.title)
    return _chat_summary(chat)


@router.get("/{chat_id}", response_model=ChatDetail)
async def get_chat(chat_id: str, request: Request):
    try:
        chat = await asyncio.to_thread(_db(request).get_chat, chat_id)
        messages = await asyncio.to_thread(_db(request).list_messages, chat_id)
    except KeyError:
        raise HTTPException(404, "Chat not found") from None
    return {
        **_chat_summary(chat),
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
                "metadata": message.metadata,
            }
            for message in messages
        ],
    }


@router.patch("/{chat_id}", response_model=ChatSummary)
async def rename_chat(chat_id: str, payload: ChatUpdate, request: Request):
    try:
        chat = await asyncio.to_thread(_db(request).rename_chat, chat_id, payload.title)
    except KeyError:
        raise HTTPException(404, "Chat not found") from None
    return _chat_summary(chat)


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, request: Request):
    database, stores = _db(request), request.app.state.stores
    try:
        documents = await asyncio.to_thread(database.list_documents, chat_id)
        await asyncio.to_thread(database.get_chat, chat_id)
    except KeyError:
        raise HTTPException(404, "Chat not found") from None
    # Keep the SQLite record until all external artifacts are gone, so a
    # partially failed cleanup remains visible and can be retried safely.
    cleanup_errors: list[Exception] = []
    try:
        await asyncio.to_thread(stores.delete_chat, chat_id)
    except Exception as exc:
        logger.exception("Failed to delete vectors for chat %s", chat_id)
        cleanup_errors.append(exc)
    for document in documents:
        path = request.app.state.settings.uploads_path / document.stored_filename
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except Exception as exc:
            logger.exception("Failed to delete uploaded file %s", path)
            cleanup_errors.append(exc)
    if cleanup_errors:
        raise HTTPException(500, "Chat cleanup failed; the deletion can be retried")
    await asyncio.to_thread(database.delete_chat, chat_id)


@router.post("/{chat_id}/messages")
async def send_message(chat_id: str, payload: MessageCreate, request: Request):
    database = _db(request)
    try:
        chat = await asyncio.to_thread(database.get_chat, chat_id)
        history = await asyncio.to_thread(
            database.list_messages,
            chat_id,
            request.app.state.settings.chat_history_messages,
        )
    except KeyError:
        raise HTTPException(404, "Chat not found") from None

    if chat.title == "New chat" and not history:
        await asyncio.to_thread(database.rename_chat, chat_id, make_title(payload.content))
    await asyncio.to_thread(database.create_message, chat_id, "user", payload.content)
    documents = await asyncio.to_thread(database.list_documents, chat_id)
    document_catalog = [
        {
            "id": document.id,
            "filename": document.filename,
            "category": document.category,
            "document_type": document.document_type,
            "topics": document.topics,
            "keywords": document.keywords,
        }
        for document in documents
    ]

    async def stream() -> AsyncIterator[str]:
        complete: list[str] = []
        final_metadata: dict = {}
        try:
            async for event in _pipeline(request).answer(
                chat_id, payload.content, history, document_catalog
            ):
                if event["type"] == "delta":
                    complete.append(event["content"])
                elif event["type"] == "done":
                    final_metadata = {
                        "sources": event["sources"],
                        "rag_used": event["rag_used"],
                        "model": event["model"],
                        "request_id": event["request_id"],
                    }
                yield json.dumps(event, ensure_ascii=False) + "\n"
            if complete:
                await asyncio.to_thread(
                    database.create_message,
                    chat_id,
                    "assistant",
                    "".join(complete),
                    final_metadata,
                )
        except Exception as exc:
            error = {"type": "error", "message": f"Unable to generate an answer: {exc}"}
            yield json.dumps(error, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
