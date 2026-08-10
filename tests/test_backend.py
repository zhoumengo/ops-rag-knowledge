from pathlib import Path

import pytest

from ops_rag.backend import RAGService, _structured_evidence
from ops_rag.manifest import load_manifest
from ops_rag.models import SearchFilters, SearchRequest
from ops_rag.settings import Settings


ROOT = Path(__file__).resolve().parents[1]


class FakeRAG:
    def __init__(self) -> None:
        self.calls = []

    async def aquery(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if kwargs.get("only_need_context"):
            return (
                'File Path: OPS-100 (Final).pdf, "page_idx": 0, text: old result\n'
                'File Path: OPS-100 (Draft).pdf, "page_idx": 1, text: newer result'
            )
        return "两个版本的文档给出不同记录，应分别核实。"

    async def finalize_storages(self):
        return None


class FakeChunks:
    async def get_by_ids(self, ids):
        chunks = {
            "chunk-final": {
                "content": "检查连接器 WWC1-X2 是否松动，并检查 X-Encoder。",
                "full_doc_id": "OPS-100-example-final",
                "file_path": "OPS-100 (Final).pdf",
                "page_idx": 2,
                "original_type": "text",
            }
        }
        return [chunks.get(chunk_id) for chunk_id in ids]


def settings(tmp_path: Path) -> Settings:
    return Settings(
        source_dir=ROOT / "data",
        manifest_path=ROOT / "config" / "manifest.example.yaml",
        working_dir=tmp_path / "storage",
        output_dir=tmp_path / "output",
        audit_log=tmp_path / "audit.jsonl",
        parser="mineru",
        parse_method="auto",
        max_workers=1,
        llm_model="test",
        vision_model="test",
        embedding_model="test",
        embedding_dim=3,
        crewai_model="test",
        api_key="test",
        base_url=None,
    )


@pytest.mark.asyncio
async def test_search_returns_evidence_without_inventing_conflict(tmp_path: Path) -> None:
    rag = FakeRAG()
    service = RAGService(rag, settings(tmp_path))
    response = await service.search(
        SearchRequest(
            query="OPS-100 有什么差异？",
            filters=SearchFilters(document_code=["OPS-100"]),
        )
    )
    assert len(rag.calls) == 2
    assert {item.page for item in response.evidence} == {1, 2}
    assert response.conflicts == []
    assert response.warning is None
    assert service.settings.audit_log.exists()


@pytest.mark.asyncio
async def test_search_rejects_when_filters_match_nothing(tmp_path: Path) -> None:
    rag = FakeRAG()
    service = RAGService(rag, settings(tmp_path))
    response = await service.search(
        SearchRequest(
            query="不存在",
            filters=SearchFilters(system=["NOT_A_REAL_SYSTEM"]),
        )
    )
    assert not rag.calls
    assert "无法从已入库资料确认" in response.answer


@pytest.mark.asyncio
async def test_structured_evidence_resolves_entity_source_chunk() -> None:
    documents = load_manifest(ROOT / "config" / "manifest.example.yaml").documents
    query_data = {
        "data": {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "WWC1-X2",
                    "source_id": "chunk-final",
                    "file_path": "OPS-100 (Final).pdf",
                }
            ],
            "relationships": [],
        }
    }

    evidence = await _structured_evidence(query_data, documents, FakeChunks())

    assert len(evidence) == 1
    assert evidence[0].document_id == "OPS-100-example-final"
    assert evidence[0].file_name == "OPS-100 (Final).pdf"
    assert evidence[0].page == 3
    assert "WWC1-X2" in evidence[0].excerpt
