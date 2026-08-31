from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Conversation:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentRecord:
    id: str
    chat_id: str
    filename: str
    stored_filename: str
    file_hash: str
    file_type: str
    category: str
    document_type: str
    topics: list[str]
    keywords: list[str]
    created_at: datetime
    chunk_count: int


@dataclass(slots=True)
class RetrievedNode:
    node_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float = 0.0
    reranker_score: float | None = None
    metadata_score: float = 0.0

    @property
    def best_score(self) -> float:
        return self.reranker_score if self.reranker_score is not None else self.vector_score
