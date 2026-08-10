from pathlib import Path

from ops_rag.manifest import filter_documents, load_manifest
from ops_rag.models import SearchFilters
from ops_rag.settings import Settings


ROOT = Path(__file__).resolve().parents[1]


def test_example_manifest_is_valid() -> None:
    manifest = load_manifest(ROOT / "config" / "manifest.example.yaml")
    assert manifest.source_root == "./data"
    assert len(manifest.documents) == 2
    assert len({doc.document_id for doc in manifest.documents}) == 2
    assert all(len(doc.sha256) == 64 for doc in manifest.documents)


def test_same_code_different_problem_ids_are_preserved() -> None:
    manifest = load_manifest(ROOT / "config" / "manifest.example.yaml")
    records = [doc for doc in manifest.documents if doc.document_code == "OPS-100"]
    assert {doc.problem_id for doc in records} == {"example-final", "example-draft"}


def test_metadata_filters_apply_before_retrieval() -> None:
    manifest = load_manifest(ROOT / "config" / "manifest.example.yaml")
    records = filter_documents(
        manifest.documents,
        SearchFilters(
            status=["Final"],
            system=["ExampleSystem"],
            software=["1.0"],
            document_code=["OPS-100"],
        ),
    )
    assert {doc.problem_id for doc in records} == {"example-final"}


def test_settings_load_workspace_csv(tmp_path: Path, monkeypatch) -> None:
    credential_file = tmp_path / "workspace.csv"
    credential_file.write_text(
        "\ufeffid,123\napiKey,secret-key\n"
        "openAiCompatible,https://example.test/compatible-mode/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPS_RAG_CREDENTIAL_FILE", str(credential_file))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    settings = Settings.from_env()
    assert settings.api_key == "secret-key"
    assert settings.base_url == "https://example.test/compatible-mode/v1"
