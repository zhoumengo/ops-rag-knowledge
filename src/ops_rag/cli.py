from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

from .backend import RAGService, create_rag
from .crew import build_crew
from .manifest import build_manifest, load_manifest, save_manifest
from .models import SearchFilters, SearchRequest
from .settings import Settings


def _settings() -> Settings:
    load_dotenv(override=False)
    settings = Settings.from_env()
    settings.ensure_runtime_dirs()
    os.environ.setdefault(
        "XDG_DATA_HOME", str(settings.working_dir.parent / "crewai_data")
    )
    os.environ.setdefault("CREWAI_STORAGE_DIR", "ops_rag_knowledge")
    return settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ops-rag", description="RAG-Anything + CrewAI 运维知识库"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="扫描资料并生成 manifest")
    manifest.add_argument("--source-dir", type=Path)
    manifest.add_argument("--output", type=Path)

    sub.add_parser("doctor", help="检查路径、命令、依赖和凭证")
    ingest = sub.add_parser("ingest", help="按 manifest 摄取文档，可断点复用解析缓存")
    ingest.add_argument(
        "--document-code",
        action="append",
        default=[],
        help="只摄取指定文档编号，可重复使用，例如 OPS-100",
    )

    query = sub.add_parser("query", help="直接查询 RAG-Anything")
    query.add_argument("question")
    query.add_argument("--mode", default="mix")
    query.add_argument("--status", action="append", default=[])
    query.add_argument("--system", action="append", default=[])
    query.add_argument("--software", action="append", default=[])
    query.add_argument("--destination", action="append", default=[])
    query.add_argument("--document-code", action="append", default=[])
    query.add_argument("--vlm", action="store_true")

    ask = sub.add_parser("ask", help="通过 CrewAI Agent 查询")
    ask.add_argument("question")
    web = sub.add_parser("web", help="启动网页聊天界面")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    return parser


async def _with_service(settings: Settings, action: str, args: argparse.Namespace) -> int:
    rag = create_rag(settings)
    service = RAGService(rag, settings)
    try:
        if action == "ingest":
            selected = None
            if args.document_code:
                wanted = {value.casefold() for value in args.document_code}
                selected = [
                    doc
                    for doc in service.manifest.documents
                    if doc.document_code
                    and doc.document_code.casefold() in wanted
                ]
                if not selected:
                    raise ValueError("没有 manifest 文档匹配 --document-code")
            results = await service.ingest(selected)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0 if all(item["ok"] for item in results) else 1
        if action == "query":
            response = await service.search(
                SearchRequest(
                    query=args.question,
                    mode=args.mode,
                    filters=SearchFilters(
                        status=args.status,
                        system=args.system,
                        software=args.software,
                        destination=args.destination,
                        document_code=args.document_code,
                    ),
                    vlm_enhanced=args.vlm,
                )
            )
            print(response.model_dump_json(indent=2))
            return 0
        if action == "ask":
            crew = build_crew(service, settings)
            result = await crew.kickoff_async(inputs={"question": args.question})
            print(result)
            return 0
        raise ValueError(action)
    finally:
        await service.close()


def _doctor(settings: Settings) -> int:
    checks = {
        "source_dir": settings.source_dir.is_dir(),
        "manifest": settings.manifest_path.is_file(),
        "pdfinfo": bool(shutil.which("pdfinfo")),
        "pdftotext": bool(shutil.which("pdftotext")),
        "libreoffice": bool(shutil.which("libreoffice")),
        "OPENAI_API_KEY": bool(settings.api_key),
    }
    for module in ("raganything", "crewai"):
        try:
            __import__(module)
            checks[f"python:{module}"] = True
        except ImportError:
            checks[f"python:{module}"] = False
    if checks["manifest"]:
        manifest = load_manifest(settings.manifest_path)
        checks["manifest_documents"] = len(manifest.documents)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if all(value for value in checks.values() if isinstance(value, bool)) else 1


def main() -> None:
    args = _parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    settings = _settings()
    try:
        if args.command == "web":
            from .web import run_server

            run_server(host=args.host, port=args.port)
            return
        if args.command == "manifest":
            source = (args.source_dir or settings.source_dir).resolve()
            output = (args.output or settings.manifest_path).resolve()
            manifest = build_manifest(source)
            save_manifest(manifest, output)
            print(f"已写入 {output}，共 {len(manifest.documents)} 份资料。")
            return
        if args.command == "doctor":
            raise SystemExit(_doctor(settings))
        raise SystemExit(asyncio.run(_with_service(settings, args.command, args)))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
