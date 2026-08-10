from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    document_id: str
    file_name: str
    source_path: str
    sha256: str
    size_bytes: int
    pages: int | None = None
    document_code: str | None = None
    problem_id: str | None = None
    status: str | None = None
    version: str | None = None
    updated_at: str | None = None
    systems: list[str] = Field(default_factory=list)
    software: list[str] = Field(default_factory=list)
    destinations: list[str] = Field(default_factory=list)
    content_profile: Literal[
        "operations_record", "training_module", "presentation", "terminology", "other"
    ] = "other"
    parser: Literal["mineru", "docling", "paddleocr"] = "mineru"
    parse_method: Literal["auto", "txt", "ocr"] = "auto"


class Manifest(BaseModel):
    schema_version: int = 1
    source_root: str
    documents: list[DocumentRecord]


class SearchFilters(BaseModel):
    status: list[str] = Field(default_factory=list)
    system: list[str] = Field(default_factory=list)
    software: list[str] = Field(default_factory=list)
    destination: list[str] = Field(default_factory=list)
    document_code: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["local", "global", "naive", "hybrid", "mix"] = "mix"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    vlm_enhanced: bool = False


class Evidence(BaseModel):
    document_id: str | None = None
    file_name: str | None = None
    status: str | None = None
    version: str | None = None
    page: int | None = None
    figure_id: str | None = None
    content_type: str | None = None
    excerpt: str


class SearchResponse(BaseModel):
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    retrieval_context: str = ""
    applied_filters: SearchFilters = Field(default_factory=SearchFilters)
    warning: str | None = None
