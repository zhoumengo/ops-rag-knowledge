from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backend import RAGService, create_rag
from .crew import build_crew
from .manifest import inspect_document, load_manifest, save_manifest
from .models import Manifest
from .settings import Settings

STATIC_DIR = Path(__file__).resolve().parent / "web_static"
SUPPORTED_SUFFIXES = {".pdf", ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
TASK_HISTORY_LIMIT = 30


def _settings() -> Settings:
    load_dotenv(override=False)
    settings = Settings.from_env()
    settings.ensure_runtime_dirs()
    os.environ.setdefault(
        "XDG_DATA_HOME", str(settings.working_dir.parent / "crewai_data")
    )
    os.environ.setdefault("CREWAI_STORAGE_DIR", "ops_rag_knowledge")
    return settings


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    # 前端传来的会话历史。当前 ask 流程按单轮知识库问答处理，
    # 保留该字段便于将来扩展多轮记忆，后端暂不使用。
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    question: str
    answer: str


class UploadTask:
    """一次文档上传-解析任务的进度状态。"""

    def __init__(self, task_id: str, file_name: str) -> None:
        self.task_id = task_id
        self.file_name = file_name
        self.status = "queued"
        self.message = "等待处理…"
        self.ok: bool | None = None
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "file_name": self.file_name,
            "status": self.status,
            "message": self.message,
            "ok": self.ok,
            "created_at": self.created_at,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rebuild_manifest(settings: Settings) -> list[Any]:
    """重建 manifest：结果与 `uv run ops-rag manifest` 一致（同一文件集、
    同一顺序、同一解析逻辑），但只对内容变化/新增的文件执行完整解析，
    未变化的旧文件复用现有条目，避免每次上传都重跑 pdfinfo/pdftotext。"""
    existing: dict[str, Any] = {}
    if settings.manifest_path.is_file():
        try:
            existing = {
                doc.file_name: doc
                for doc in load_manifest(settings.manifest_path).documents
            }
        except Exception:
            existing = {}

    paths = sorted(
        path
        for path in settings.source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = []
    for path in paths:
        old = existing.get(path.name)
        if old is not None and old.sha256 == _file_sha256(path):
            documents.append(old)
            continue
        documents.append(inspect_document(path))

    manifest = Manifest(
        source_root=str(settings.source_dir.resolve()), documents=documents
    )
    save_manifest(manifest, settings.manifest_path)
    return documents


class ChatService:
    """长驻的 CrewAI + RAG 问答服务，等价于 `ops-rag ask`。

    与每次消息都执行 `uv run ops-rag ask` 相比，进程内复用避免了反复
    加载知识图谱、向量库和 CrewAI 知识源的开销。上传摄取与对话共用
    `_busy` 锁，保证同一时间只有一个操作在读写索引。
    """

    def __init__(self) -> None:
        self.settings = _settings()
        self._service: RAGService | None = None
        self._crew: Any | None = None
        self._init_lock = asyncio.Lock()
        self._busy = asyncio.Lock()
        self._tasks: dict[str, UploadTask] = {}
        self._task_seq = 0

    async def _ensure_service(self) -> RAGService:
        if self._service is None:
            async with self._init_lock:
                if self._service is None:
                    rag = create_rag(self.settings)
                    self._service = RAGService(rag, self.settings)
        return self._service

    async def _ensure_ready(self) -> None:
        service = await self._ensure_service()
        if self._crew is None:
            async with self._init_lock:
                if self._crew is None:
                    self._crew = build_crew(service, self.settings)

    async def health(self) -> dict[str, Any]:
        manifest = self.settings.manifest_path.is_file()
        documents = 0
        if manifest:
            try:
                from .manifest import load_manifest as _load

                documents = len(_load(self.settings.manifest_path).documents)
            except Exception:
                documents = 0
        return {
            "status": "ok",
            "manifest": manifest,
            "documents": documents,
            "ready": self._crew is not None,
        }

    async def ask(self, question: str) -> str:
        await self._ensure_ready()
        async with self._busy:
            assert self._crew is not None
            result = await self._crew.kickoff_async(inputs={"question": question})
        raw = getattr(result, "raw", None)
        return str(raw if raw is not None else result)

    def create_upload_task(self, file_name: str) -> UploadTask:
        self._task_seq += 1
        task = UploadTask(task_id=f"t{self._task_seq}", file_name=file_name)
        self._tasks[task.task_id] = task
        while len(self._tasks) > TASK_HISTORY_LIMIT:
            self._tasks.pop(next(iter(self._tasks)))
        return task

    def get_task(self, task_id: str) -> UploadTask | None:
        return self._tasks.get(task_id)

    async def run_upload_task(self, task: UploadTask, dest: Path) -> None:
        """后台执行：更新 manifest -> 解析并摄取单个文档。"""
        try:
            async with self._busy:
                task.status = "manifest"
                task.message = "正在生成资料清单 (manifest)…"
                documents = await asyncio.to_thread(_rebuild_manifest, self.settings)
                record = next(
                    (doc for doc in documents if doc.file_name == dest.name), None
                )
                if record is None:
                    raise RuntimeError("资料清单中找不到刚上传的文件")
                service = await self._ensure_service()
                service.manifest = load_manifest(self.settings.manifest_path)

                task.status = "parsing"
                task.message = "文件解析中（版面识别 + 图表理解）…"
                results = await service.ingest([record])
                result = results[0] if results else {"ok": False, "error": "没有处理结果"}

                task.ok = bool(result.get("ok"))
                if task.ok:
                    task.status = "done"
                    task.message = f"完成：{result.get('file_name', task.file_name)} 已入库"
                else:
                    task.status = "error"
                    task.message = f"摄取失败：{result.get('error', '未知错误')}"
        except Exception as exc:  # noqa: BLE001 - 面向用户返回可读错误
            task.status = "error"
            task.ok = False
            task.message = f"处理失败：{exc}"

    async def close(self) -> None:
        if self._service is not None:
            try:
                await self._service.close()
            finally:
                self._service = None
                self._crew = None


service = ChatService()

app = FastAPI(title="ops-rag 知识助手", version="0.2.0")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return await service.health()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="消息不能为空")
    try:
        answer = await service.ask(question)
    except Exception as exc:  # noqa: BLE001 - 面向用户返回可读错误
        raise HTTPException(status_code=500, detail=f"问答服务出错：{exc}") from exc
    return ChatResponse(question=question, answer=answer)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    raw_name = file.filename or ""
    filename = Path(raw_name.replace("\\", "/")).name.strip()
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"不支持的文件类型：{suffix or '未知'}。"
                f"支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
            ),
        )

    dest = service.settings.source_dir / filename
    size = 0
    try:
        with dest.open("wb") as stream:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制",
                    )
                stream.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    task = service.create_upload_task(filename)
    asyncio.create_task(service.run_upload_task(task, dest))
    return {
        "task_id": task.task_id,
        "file_name": filename,
        "status": task.status,
        "size_bytes": size,
    }


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.to_dict()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
