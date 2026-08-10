from fastapi.testclient import TestClient

import ops_rag.web as web
from ops_rag.web import app
from ops_rag.models import Manifest
from ops_rag.settings import Settings


def test_health_reports_ok():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_index_page_served():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "运维知识助手" in response.text


def test_static_avatar_served():
    with TestClient(app) as client:
        response = client.get("/static/robot.svg")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/svg+xml")


def test_chat_rejects_blank_message():
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "   "})
        assert response.status_code == 422


def test_upload_rejects_unsupported_extension():
    with TestClient(app) as client:
        response = client.post(
            "/api/upload", files={"file": ("a.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 422


def test_upload_accepts_pdf_and_creates_task(monkeypatch):
    async def fake_run(task, dest):
        pass

    monkeypatch.setattr(web.service, "run_upload_task", fake_run)
    name = "web_test_upload.pdf"
    dest = web.service.settings.source_dir / name
    dest.unlink(missing_ok=True)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/upload",
                files={"file": (name, b"%PDF-1.4 fake", "application/pdf")},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["task_id"]
            assert data["file_name"] == name
            assert dest.exists()
    finally:
        dest.unlink(missing_ok=True)


def test_upload_sanitizes_path_traversal(monkeypatch):
    async def fake_run(task, dest):
        pass

    monkeypatch.setattr(web.service, "run_upload_task", fake_run)
    dest = web.service.settings.source_dir / "evil.pdf"
    dest.unlink(missing_ok=True)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/upload",
                files={"file": ("..\\evil.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
            assert response.status_code == 200
            assert dest.exists()
            assert not (web.service.settings.source_dir.parent / "evil.pdf").exists()
    finally:
        dest.unlink(missing_ok=True)


def test_task_not_found():
    with TestClient(app) as client:
        assert client.get("/api/tasks/not-exist").status_code == 404


def _test_settings(tmp_path) -> Settings:
    data = tmp_path / "data"
    data.mkdir()
    return Settings(
        source_dir=data,
        manifest_path=tmp_path / "manifest.yaml",
        working_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        audit_log=tmp_path / "audit.jsonl",
        parser="mineru",
        parse_method="auto",
        max_workers=1,
        llm_model="test",
        vision_model="test",
        embedding_model="test",
        embedding_dim=64,
        crewai_model="test",
        api_key="test",
        base_url=None,
    )


def test_rebuild_manifest_only_inspects_new_file(monkeypatch, tmp_path):
    settings = _test_settings(tmp_path)
    old_pdf = settings.source_dir / "a.pdf"
    new_pdf = settings.source_dir / "b.pdf"
    old_pdf.write_bytes(b"%PDF-1.4 fake a")
    new_pdf.write_bytes(b"%PDF-1.4 fake b")

    from ops_rag.manifest import inspect_document, save_manifest

    old_entry = inspect_document(old_pdf)
    save_manifest(
        Manifest(source_root=str(settings.source_dir.resolve()), documents=[old_entry]),
        settings.manifest_path,
    )

    inspected = []
    original = web.inspect_document

    def counting_inspect(path):
        inspected.append(path.name)
        return original(path)

    monkeypatch.setattr(web, "inspect_document", counting_inspect)
    documents = web._rebuild_manifest(settings)

    assert len(documents) == 2
    assert documents[0] == old_entry  # 旧文件条目被复用（等值，非重新解析）
    assert documents[1].file_name == "b.pdf"
    assert inspected == ["b.pdf"]  # 只解析了新文件
