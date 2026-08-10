from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Iterable

import yaml

from .models import DocumentRecord, Manifest, SearchFilters

SUPPORTED_SUFFIXES = {".pdf", ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, timeout=30
        )
        # pdftotext/pdfinfo emit UTF-8 regardless of the Windows locale
        # encoding; decoding in the main thread avoids the reader-thread
        # UnicodeDecodeError that turned stdout into None on cp936 locales.
        return result.stdout.decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def _pdf_pages(path: Path) -> int | None:
    text = _run_text(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _pdf_first_pages(path: Path) -> str:
    return _run_text(
        ["pdftotext", "-f", "1", "-l", "2", "-layout", str(path), "-"]
    )


def _split_semicolon(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def _field(text: str, name: str, next_names: Iterable[str]) -> list[str]:
    next_pattern = "|".join(re.escape(item) for item in next_names)
    match = re.search(
        rf"^\s*{re.escape(name)}\s+(.+?)(?=^\s*(?:{next_pattern})\s+)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        return []
    value = re.sub(r"\s+", " ", match.group(1))
    if name == "Destination":
        value = re.sub(r"\bHardware\b\s*", "", value)
    return _split_semicolon(value)


def _profile(path: Path) -> tuple[str, str, str]:
    name = path.stem.lower()
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return "terminology", "docling", "auto"
    if re.match(r"[a-z]{2,10}-[0-9a-z]+", name):
        return "operations_record", "mineru", "txt"
    if "_pp" in name:
        return "presentation", "mineru", "auto"
    if "module" in name:
        return "training_module", "mineru", "auto"
    return "other", "mineru", "auto"


def inspect_document(path: Path) -> DocumentRecord:
    text = _pdf_first_pages(path) if path.suffix.lower() == ".pdf" else ""
    code_match = re.search(r"\b([A-Z]{2,10}-[0-9A-Z]+)\b", path.name, re.I)
    status_match = re.search(r"\((Draft|Provisional|Final|Released)\)", path.name, re.I)
    problem_match = re.search(r"Problem ID:\s*(\d+)", text)
    updated_match = re.search(r"Updated by .*? on ([0-9T:.\-+Z]+)", text)
    profile, parser, parse_method = _profile(path)
    document_code = code_match.group(1).upper() if code_match else None
    problem_id = problem_match.group(1) if problem_match else None
    short_hash = _sha256(path)
    stable_id_parts = [document_code or path.stem, problem_id or short_hash[:12]]

    return DocumentRecord(
        document_id="-".join(stable_id_parts),
        file_name=path.name,
        source_path=str(path.resolve()),
        sha256=short_hash,
        size_bytes=path.stat().st_size,
        pages=_pdf_pages(path) if path.suffix.lower() == ".pdf" else None,
        document_code=document_code,
        problem_id=problem_id,
        status=status_match.group(1).title() if status_match else None,
        version="1" if re.search(r"\)\s*1$", path.stem) else None,
        updated_at=updated_match.group(1) if updated_match else None,
        systems=_field(text, "System", ["Software"]),
        software=_field(text, "Software", ["Destination"]),
        destinations=_field(text, "Destination", ["Family"]),
        content_profile=profile,
        parser=parser,
        parse_method=parse_method,
    )


def build_manifest(source_dir: Path) -> Manifest:
    paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return Manifest(
        source_root=str(source_dir.resolve()),
        documents=[inspect_document(path) for path in paths],
    )


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            manifest.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        ),
        encoding="utf-8",
    )


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def filter_documents(
    documents: list[DocumentRecord], filters: SearchFilters
) -> list[DocumentRecord]:
    def overlaps(actual: list[str], wanted: list[str]) -> bool:
        return not wanted or bool(
            {item.casefold() for item in actual} & {item.casefold() for item in wanted}
        )

    result = []
    for doc in documents:
        if filters.status and (doc.status or "").casefold() not in {
            item.casefold() for item in filters.status
        }:
            continue
        if filters.document_code and (doc.document_code or "").casefold() not in {
            item.casefold() for item in filters.document_code
        }:
            continue
        if not overlaps(doc.systems, filters.system):
            continue
        if not overlaps(doc.software, filters.software):
            continue
        if not overlaps(doc.destinations, filters.destination):
            continue
        result.append(doc)
    return result
