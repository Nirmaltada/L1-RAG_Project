from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=120)


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=50_000)


class MessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatDetail(ChatSummary):
    messages: list[MessageResponse]


class DocumentResponse(BaseModel):
    id: str
    chat_id: str
    filename: str
    file_type: str
    category: str
    document_type: str
    topics: list[str]
    keywords: list[str]
    created_at: datetime
    chunk_count: int


class SourceResponse(BaseModel):
    index: int
    document_id: str
    filename: str
    page: int | None = None
    category: str
    vector_score: float
    reranker_score: float | None = None
