import asyncio
import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.database import Database
from app.providers.openrouter import ProviderError
from app.rag.ingestion import IngestionError, SUPPORTED_EXTENSIONS
from app.schemas import DocumentResponse

router = APIRouter(prefix="/api/chats/{chat_id}/documents", tags=["documents"])
logger = logging.getLogger(__name__)


async def _remove_artifacts(request: Request, document_id: str, path: Path) -> list[Exception]:
    errors: list[Exception] = []
    try:
        await asyncio.to_thread(request.app.state.stores.delete_document, document_id)
    except Exception as exc:
        logger.exception("Failed to delete vectors for document %s", document_id)
        errors.append(exc)
    try:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except Exception as exc:
        logger.exception("Failed to delete uploaded file %s", path)
        errors.append(exc)
    return errors


def _response(document: object) -> dict:
    return {
        "id": document.id,
        "chat_id": document.chat_id,
        "filename": document.filename,
        "file_type": document.file_type,
        "category": document.category,
        "document_type": document.document_type,
        "topics": document.topics,
        "keywords": document.keywords,
        "created_at": document.created_at,
        "chunk_count": document.chunk_count,
    }


@router.get("", response_model=list[DocumentResponse])
async def list_documents(chat_id: str, request: Request):
    database: Database = request.app.state.db
    try:
        await asyncio.to_thread(database.get_chat, chat_id)
    except KeyError:
        raise HTTPException(404, "Chat not found") from None
    documents = await asyncio.to_thread(database.list_documents, chat_id)
    return [_response(document) for document in documents]


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    chat_id: str, request: Request, file: UploadFile = File(...)
):
    database: Database = request.app.state.db
    try:
        await asyncio.to_thread(database.get_chat, chat_id)
    except KeyError:
        raise HTTPException(404, "Chat not found") from None

    original_filename = Path(file.filename or "document").name
    suffix = Path(original_filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, "Supported file types are PDF, DOCX, TXT, and Markdown")
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "The uploaded file is empty")
    file_hash = hashlib.sha256(contents).hexdigest()
    async with request.app.state.ingestion_lock:
        duplicate = await asyncio.to_thread(database.find_document_by_hash, chat_id, file_hash)
        if duplicate:
            raise HTTPException(409, f"{duplicate.filename} is already uploaded to this chat")

        document_id = str(uuid.uuid4())
        stored_filename = f"{document_id}{suffix}"
        destination = request.app.state.settings.uploads_path / stored_filename
        await asyncio.to_thread(destination.write_bytes, contents)
        try:
            record = await request.app.state.ingestion.ingest(
                chat_id=chat_id,
                document_id=document_id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                file_hash=file_hash,
                path=destination,
            )
            await asyncio.to_thread(database.add_document, record)
            return _response(record)
        except IngestionError as exc:
            await _remove_artifacts(request, document_id, destination)
            raise HTTPException(422, str(exc)) from exc
        except ProviderError as exc:
            await _remove_artifacts(request, document_id, destination)
            message = str(exc)
            if "429" in message or "Too Many Requests" in message:
                raise HTTPException(
                    503,
                    "OpenRouter is rate limiting embedding requests right now. "
                    "Wait a minute and upload the document again.",
                ) from exc
            raise HTTPException(502, f"Document ingestion failed: {exc}") from exc
        except Exception as exc:
            await _remove_artifacts(request, document_id, destination)
            raise HTTPException(502, f"Document ingestion failed: {exc}") from exc


@router.delete("/{document_id}", status_code=204)
async def delete_document(chat_id: str, document_id: str, request: Request):
    database: Database = request.app.state.db
    try:
        document = await asyncio.to_thread(database.get_document, chat_id, document_id)
    except KeyError:
        raise HTTPException(404, "Document not found") from None
    path = request.app.state.settings.uploads_path / document.stored_filename
    cleanup_errors = await _remove_artifacts(request, document_id, path)
    if cleanup_errors:
        raise HTTPException(500, "Document cleanup failed; the deletion can be retried")
    await asyncio.to_thread(database.delete_document, chat_id, document_id)
