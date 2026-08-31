import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from app.models import Conversation, DocumentRecord, Message


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    topics_json TEXT NOT NULL DEFAULT '[]',
                    keywords_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (chat_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    UNIQUE(chat_id, file_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_documents_chat ON documents(chat_id);
                """
            )

    def create_chat(self, title: str = "New chat") -> Conversation:
        chat_id, now = str(uuid.uuid4()), utc_now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, title.strip() or "New chat", now, now),
            )
        return self.get_chat(chat_id)

    def list_chats(self) -> list[Conversation]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [self._conversation(row) for row in rows]

    def get_chat(self, chat_id: str) -> Conversation:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (chat_id,)
            ).fetchone()
        if row is None:
            raise KeyError(chat_id)
        return self._conversation(row)

    def rename_chat(self, chat_id: str, title: str) -> Conversation:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip(), utc_now(), chat_id),
            )
            if not cursor.rowcount:
                raise KeyError(chat_id)
        return self.get_chat(chat_id)

    def delete_chat(self, chat_id: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (chat_id,))
            if not cursor.rowcount:
                raise KeyError(chat_id)

    def create_message(
        self, chat_id: str, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> Message:
        message_id, now = str(uuid.uuid4()), utc_now()
        with self.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (chat_id,)
            ).fetchone() is None:
                raise KeyError(chat_id)
            connection.execute(
                """INSERT INTO messages
                   (id, conversation_id, role, content, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (message_id, chat_id, role, content, now, json.dumps(metadata or {})),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, chat_id)
            )
        return Message(message_id, chat_id, role, content, datetime.fromisoformat(now), metadata or {})

    def list_messages(self, chat_id: str, limit: int | None = None) -> list[Message]:
        sql = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at DESC"
        parameters: list[Any] = [chat_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._message(row) for row in reversed(rows)]

    def add_document(self, record: DocumentRecord) -> DocumentRecord:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO documents
                   (id, chat_id, filename, stored_filename, file_hash, file_type, category,
                    document_type, topics_json, keywords_json, created_at, chunk_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id, record.chat_id, record.filename, record.stored_filename,
                    record.file_hash, record.file_type, record.category, record.document_type,
                    json.dumps(record.topics), json.dumps(record.keywords),
                    record.created_at.isoformat(), record.chunk_count,
                ),
            )
        return record

    def list_documents(self, chat_id: str) -> list[DocumentRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE chat_id = ? ORDER BY created_at DESC", (chat_id,)
            ).fetchall()
        return [self._document(row) for row in rows]

    def get_document(self, chat_id: str, document_id: str) -> DocumentRecord:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ? AND chat_id = ?", (document_id, chat_id)
            ).fetchone()
        if row is None:
            raise KeyError(document_id)
        return self._document(row)

    def find_document_by_hash(self, chat_id: str, file_hash: str) -> DocumentRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE chat_id = ? AND file_hash = ?", (chat_id, file_hash)
            ).fetchone()
        return self._document(row) if row else None

    def delete_document(self, chat_id: str, document_id: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ? AND chat_id = ?", (document_id, chat_id)
            )
            if not cursor.rowcount:
                raise KeyError(document_id)

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(
            row["id"], row["title"], datetime.fromisoformat(row["created_at"]),
            datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(
            row["id"], row["conversation_id"], row["role"], row["content"],
            datetime.fromisoformat(row["created_at"]), json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"], chat_id=row["chat_id"], filename=row["filename"],
            stored_filename=row["stored_filename"], file_hash=row["file_hash"],
            file_type=row["file_type"], category=row["category"],
            document_type=row["document_type"], topics=json.loads(row["topics_json"]),
            keywords=json.loads(row["keywords_json"]), created_at=datetime.fromisoformat(row["created_at"]),
            chunk_count=row["chunk_count"],
        )
