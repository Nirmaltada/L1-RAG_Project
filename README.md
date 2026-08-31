# Local RAG Chat

A Windows-first, single-user chat application that answers normal questions and automatically uses documents uploaded to the active chat when they are relevant. The UI is plain HTML/CSS/JavaScript; the API is FastAPI; chat state lives in SQLite; and LlamaIndex connects token-based chunking and filtered retrieval to persistent and in-memory Chroma collections.

## Project layout

```text
project/
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── api.js
│       ├── chat.js
│       └── app.js
└── backend/
    ├── app/
    │   ├── api/              # Thin HTTP routes
    │   ├── providers/        # Groq and OpenRouter clients
    │   ├── rag/              # Ingestion, retrieval, reranking, prompts, orchestration
    │   ├── config.py
    │   ├── database.py
    │   ├── models.py
    │   ├── schemas.py
    │   └── main.py
    ├── data/                 # Runtime database, uploads, and persistent Chroma
    ├── tests/
    ├── .env.example
    └── requirements.txt
```

## Windows installation

Use 64-bit **Python 3.11**. The pinned Chroma HNSW package provides a prebuilt Windows wheel for Python 3.11; using Python 3.12 would require local Microsoft C++ build tools.

Open PowerShell in the project folder:

```powershell
cd backend
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

The temporary `Set-ExecutionPolicy` line is needed only when PowerShell blocks virtual-environment activation. It affects the current PowerShell process only.

## API keys and `.env`

Create a Groq API key in the Groq console and an OpenRouter API key in OpenRouter. Open `backend\.env` and set only these required secrets:

```env
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

Keep `.env` private. It is ignored by Git and is read only by the backend; no key is included in frontend JavaScript.

Embedding mode is a strict switch:

```env
# false: use the OpenRouter embedding model chain only
# true: use the local CPU model only
USE_LOCAL_EMBEDDINGS=false
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Local mode lazily loads MiniLM on the first document upload or document query;
it does not call any OpenRouter embedding endpoint. API mode never loads MiniLM.
The first local use downloads the model from Hugging Face and caches it on the
computer. OpenRouter may still be used for reranking and answer generation.

The configured models are:

- Query rewriting and document classification: Groq `openai/gpt-oss-20b`
- Embeddings: either local `all-MiniLM-L6-v2`, or the configured OpenRouter
  primary and fallback models, selected by `USE_LOCAL_EMBEDDINGS`
- Reranking: OpenRouter `nvidia/llama-nemotron-rerank-vl-1b-v2:free`
- Answer generation: OpenRouter `z-ai/glm-5.2:free`
- First answer fallback: OpenRouter `minimax/minimax-m2.7:free`
- Final answer fallback: Groq `openai/gpt-oss-120b`

Free model availability and provider limits can change. The configured fallback chain handles temporary errors, unavailable models, timeouts, and rate limits before any response text has been emitted.

## Start the backend

From `backend` with the virtual environment active:

```powershell
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. Check `http://localhost:8000/api/health` or interactive API documentation at `http://localhost:8000/docs`.

## Start the frontend

The backend serves the frontend directly. Open `http://127.0.0.1:8000` after
Uvicorn starts.

You can still use a separate static file server during development, but serving
the UI from FastAPI avoids Live Server reloads when SQLite or Chroma files
change during a streamed answer.

## First use

1. The browser creates an initial chat automatically.
2. Ask an ordinary question to use general model knowledge.
3. Click **Upload** and choose a PDF, DOCX, TXT, or Markdown file.
4. Wait until the sidebar says the file is ready. Ingestion includes parsing, one metadata-classification request, token chunking, embedding, and dual Chroma writes.
5. Ask a question about the file. Relevant answers include source metadata; PDFs include real extracted page numbers when available.
6. Create another chat to get a completely isolated document namespace.

Chat titles are derived locally from the first user message. Chats, messages, and document records survive browser/backend restarts.

## RAG pipeline

For a chat with documents, the backend performs:

```text
question + 12 recent messages
        ↓
Groq standalone-query rewrite + intent hints
        ↓
Selected embedding mode (local MiniLM or OpenRouter)
        ↓
LlamaIndex Chroma query with hard chat_id filter (top 40)
        ↓
soft category/topic preference
        ↓
OpenRouter Nemotron reranker (top 8)
        ↓
relevance decision
   relevant ├─ yes → grounded prompt + citations
            └─ no  → general answer without citations
```

If query rewriting fails, the original question is used. If reranking fails, the best vector candidates are used. Documents are never blindly injected into every prompt.

## Metadata-aware retrieval

Each document is classified once from its filename, title, and a limited representative excerpt. The resulting `category`, `document_type`, `topics`, and `keywords` are attached to every LlamaIndex node alongside `chat_id`, `document_id`, filename, file type, chunk ID, page number, and creation time.

The query-rewrite call also returns likely categories and topics. These hints provide a small ranking boost; they are not hard filters. The only strict retrieval constraint is `chat_id == current_chat_id`, applied by LlamaIndex/Chroma before candidates are returned. This prevents cross-chat leakage without allowing an imperfect classifier to hide relevant text.

Chroma supports scalar metadata values, so topics and keywords are stored as JSON strings and decoded back to lists at retrieval/API boundaries.

## Persistent and RAM Chroma

`chromadb.PersistentClient` is the durable source of truth under `backend\data\chroma`. `chromadb.EphemeralClient` is a fast runtime mirror.

On startup, the backend reads IDs, text, metadata, and existing embedding vectors from persistent Chroma in batches and upserts them into RAM. It never calls the embedding API during hydration. New chunks are embedded once; the same LlamaIndex nodes and vectors are added to both stores.

Deleting a document removes its file, SQLite record, and vectors from both stores. Deleting a chat removes its messages, documents, uploaded files, and all persistent/RAM vectors carrying that chat ID.

## Tests

From `backend` with the environment active:

```powershell
pytest -q
```

The tests do not require API keys. They cover chat isolation, metadata round-tripping, direct-vector RAM hydration, top-40 retrieval, top-8 reranking, general chat, irrelevant-document fallback without citations, grounded sources, API persistence, and unsupported uploads.

For a syntax/import-only check:

```powershell
python -m compileall app tests
python -c "from app.main import app; print(app.title)"
```

## Troubleshooting

### Backend unavailable in the browser

Confirm Uvicorn is running on port 8000. If you use VS Code Live Server instead
of the built-in FastAPI frontend route, restart Live Server after changing
`.vscode/settings.json` so it ignores backend runtime files.

### PowerShell will not activate `.venv`

Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Alternatively, run tools explicitly as `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\uvicorn.exe`.

### `chroma-hnswlib` tries to compile or requests Visual C++

The virtual environment was created with the wrong Python version. Delete only `backend\.venv`, install 64-bit Python 3.11, and recreate it with `py -3.11 -m venv .venv`. Python 3.11 uses the pinned prebuilt Windows wheel and does not need a compiler.

### API key or 401 error

Ensure the key has no quotes or trailing spaces, `.env` is inside `backend`, and Uvicorn was restarted after editing it. `/api/health` reports whether both values loaded, but never returns the keys.

### 429, timeout, or unavailable free model

The answer path tries GLM, then MiniMax, then Groq GPT-OSS-120B. In API
embedding mode, the configured OpenRouter embedding chain is tried in order.
Set `USE_LOCAL_EMBEDDINGS=true` and restart the backend to use MiniLM instead.
Reranking and answer generation can still be affected by OpenRouter limits.

### Changing embedding mode

Do not mix local and API vectors in one Chroma collection. The backend records
the active mode and refuses to start if existing vectors belong to the other
mode. Delete the existing chats/documents before changing the switch, then
restart and re-upload them so all chunks are embedded in the same vector space.

### PDF has no text

The included parser extracts text layers; it does not perform OCR. Scanned image-only PDFs will report that no extractable text was found. OCR the PDF first, then upload it again.

### Duplicate file

SHA-256 duplicate detection is scoped to one chat. Delete the existing copy first or upload the same file to a different chat.

### Reset local data

Stop the backend first. To start fresh, delete `backend\data\app.db`, the contents of `backend\data\chroma`, and the contents of `backend\data\uploads`, while keeping the directories themselves. This permanently removes local chats and documents.
