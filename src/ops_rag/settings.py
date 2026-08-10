from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def _workspace_credentials() -> dict[str, str]:
    credential_file = os.getenv("OPS_RAG_CREDENTIAL_FILE")
    if not credential_file:
        return {}
    path = Path(credential_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"百炼凭证文件不存在：{path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            row[0].strip(): row[1].strip()
            for row in csv.reader(stream)
            if len(row) >= 2 and row[0].strip()
        }


@dataclass(frozen=True)
class Settings:
    source_dir: Path
    manifest_path: Path
    working_dir: Path
    output_dir: Path
    audit_log: Path
    parser: str
    parse_method: str
    max_workers: int
    llm_model: str
    vision_model: str
    embedding_model: str
    embedding_dim: int
    crewai_model: str
    api_key: str | None
    base_url: str | None
    enable_thinking: bool = False
    parser_backend: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = _workspace_credentials()
        return cls(
            source_dir=_path("OPS_RAG_SOURCE_DIR", "./data"),
            manifest_path=_path("OPS_RAG_MANIFEST", "./config/manifest.yaml"),
            working_dir=_path("OPS_RAG_WORKING_DIR", "./runtime/rag_storage"),
            output_dir=_path("OPS_RAG_OUTPUT_DIR", "./runtime/parser_output"),
            audit_log=_path("OPS_RAG_AUDIT_LOG", "./runtime/query_audit.jsonl"),
            parser=os.getenv("PARSER", "mineru"),
            parse_method=os.getenv("PARSE_METHOD", "auto"),
            max_workers=int(os.getenv("MAX_CONCURRENT_FILES", "1")),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            vision_model=os.getenv("VISION_MODEL", "gpt-4o"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
            crewai_model=os.getenv("CREWAI_MODEL", "openai/gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY") or workspace.get("apiKey") or None,
            base_url=(
                os.getenv("OPENAI_BASE_URL")
                or workspace.get("openAiCompatible")
                or None
            ),
            enable_thinking=os.getenv("ENABLE_THINKING", "false").lower() == "true",
            parser_backend=os.getenv("MINERU_BACKEND") or None,
        )

    def ensure_runtime_dirs(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
