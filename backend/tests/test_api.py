from pathlib import Path
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.main import app, settings
from app.models import DocumentRecord


def test_chat_crud_and_persistence(tmp_path: Path):
    settings.sqlite_database_path = tmp_path / "app.db"
    settings.chroma_persist_directory = tmp_path / "chroma"
    settings.upload_directory = tmp_path / "uploads"
    settings.ensure_directories()

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200

        created = client.post("/api/chats", json={"title": "Test chat"})
        assert created.status_code == 201
        chat_id = created.json()["id"]

        renamed = client.patch(f"/api/chats/{chat_id}", json={"title": "Renamed"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Renamed"

        reopened = client.get(f"/api/chats/{chat_id}")
        assert reopened.status_code == 200
        assert reopened.json()["messages"] == []

        deleted = client.delete(f"/api/chats/{chat_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/chats/{chat_id}").status_code == 404


def test_upload_rejects_unsupported_file(tmp_path: Path):
    settings.sqlite_database_path = tmp_path / "unsupported.db"
    settings.chroma_persist_directory = tmp_path / "unsupported-chroma"
    settings.upload_directory = tmp_path / "unsupported-uploads"
    settings.ensure_directories()

    with TestClient(app) as client:
        chat_id = client.post("/api/chats", json={"title": "Files"}).json()["id"]
        response = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("malware.exe", b"not really", "application/octet-stream")},
        )
        assert response.status_code == 415


def test_duplicate_upload_is_not_embedded_twice_and_delete_cleans_up(tmp_path: Path):
    settings.sqlite_database_path = tmp_path / "documents.db"
    settings.chroma_persist_directory = tmp_path / "documents-chroma"
    settings.upload_directory = tmp_path / "documents-uploads"
    settings.ensure_directories()

    class FakeIngestion:
        def __init__(self):
            self.calls = 0

        async def ingest(self, **kwargs):
            self.calls += 1
            return DocumentRecord(
                id=kwargs["document_id"],
                chat_id=kwargs["chat_id"],
                filename=kwargs["original_filename"],
                stored_filename=kwargs["stored_filename"],
                file_hash=kwargs["file_hash"],
                file_type="txt",
                category="general",
                document_type="document",
                topics=[],
                keywords=[],
                created_at=datetime.now(UTC),
                chunk_count=1,
            )

    with TestClient(app) as client:
        fake_ingestion = FakeIngestion()
        app.state.ingestion = fake_ingestion
        chat_id = client.post("/api/chats", json={"title": "Files"}).json()["id"]
        first = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("notes.txt", b"same bytes", "text/plain")},
        )
        duplicate = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("copy.txt", b"same bytes", "text/plain")},
        )
        assert first.status_code == 201
        assert duplicate.status_code == 409
        assert fake_ingestion.calls == 1

        document_id = first.json()["id"]
        stored_files = list(settings.uploads_path.iterdir())
        assert len(stored_files) == 1
        assert client.delete(f"/api/chats/{chat_id}/documents/{document_id}").status_code == 204
        assert not stored_files[0].exists()
        assert client.get(f"/api/chats/{chat_id}/documents").json() == []
