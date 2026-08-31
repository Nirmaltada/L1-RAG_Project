# Local RAG Chat

Local RAG Chat is a single-user document chat application. It lets you create isolated chats, upload documents to each chat, and ask questions that are answered with retrieval-augmented generation when the uploaded files are relevant.

The project combines a plain HTML/CSS/JavaScript frontend with a FastAPI backend, SQLite persistence, and Chroma vector storage. It supports general chat, document-aware answers, per-chat document isolation, source metadata, and streaming responses.

## Features

- Chat interface with persistent conversations
- Per-chat document uploads and document isolation
- PDF, DOCX, TXT, and Markdown ingestion
- Retrieval-augmented answers when uploaded documents are relevant
- General model answers when documents are not relevant
- Source metadata for grounded document answers
- SQLite storage for chats, messages, and document records
- Persistent Chroma storage with an in-memory runtime mirror
- Metadata-aware retrieval using document categories, topics, and keywords
- Query rewriting, reranking, and answer fallback support
- Backend-served frontend at the same local server URL
- Test coverage for chat, retrieval, providers, API behavior, and document flows

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: FastAPI
- Database: SQLite
- Vector store: Chroma
- RAG framework: LlamaIndex
- LLM providers: Groq and OpenRouter
- Embeddings: OpenRouter embeddings or local MiniLM embeddings
- Testing: Pytest

## Project Layout

```text
project/
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── api.js
│       ├── chat.js
│       ├── markdown.js
│       └── app.js
└── backend/
    ├── app/
    │   ├── api/              # HTTP routes
    │   ├── providers/        # Groq, OpenRouter, and embedding clients
    │   ├── rag/              # Ingestion, retrieval, reranking, prompts, pipeline
    │   ├── config.py
    │   ├── database.py
    │   ├── models.py
    │   ├── schemas.py
    │   └── main.py
    ├── data/                 # Local runtime data
    ├── tests/
    ├── .env.example
    └── requirements.txt
```

## How It Works

Each chat has its own document namespace. When a file is uploaded, the backend parses the document, creates token-based chunks, classifies document metadata, embeds the chunks, and stores them in Chroma with the active chat ID.

When the user asks a question, the backend rewrites the query using recent chat context, retrieves matching chunks from the active chat only, reranks the best candidates, and decides whether the document context is relevant enough to use. Relevant document questions receive grounded answers with source metadata. Unrelated questions are answered normally without forcing document context into the prompt.

## RAG Pipeline

```text
question + recent chat history
        ↓
standalone query rewrite
        ↓
selected embedding mode
        ↓
Chroma retrieval filtered by chat_id
        ↓
metadata-aware candidate ranking
        ↓
reranking
        ↓
relevance decision
        ↓
grounded answer or general answer
```

## Embedding Modes

The backend supports two embedding modes:

- API embeddings through the configured OpenRouter embedding model chain
- Local CPU embeddings through `sentence-transformers/all-MiniLM-L6-v2`

The active mode is controlled through environment configuration.

## Models

The app is configured to use:

- Groq for query rewriting and document classification
- OpenRouter for embeddings, reranking, and answer generation
- Groq as a final answer fallback

The provider layer is separated into `backend/app/providers`, so model routing and fallback behavior stay outside the API route code.

## Running Locally

Install backend dependencies:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Add your API keys in `backend\.env`:

```env
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

Start the backend:

```powershell
uvicorn app.main:app --reload
```

Open the app:

```text
http://127.0.0.1:8000
```

## Tests

Run the test suite from the backend folder:

```powershell
pytest -q
```

The tests cover chat isolation, metadata handling, retrieval behavior, reranking behavior, general chat, grounded answers, API persistence, provider fallback paths, and unsupported upload handling.

## Local Data

Runtime data is stored under `backend/data`:

- `app.db` stores chats, messages, and document records
- `uploads/` stores uploaded files
- `chroma/` stores persistent vector data

These files are local runtime artifacts and are ignored by Git.
