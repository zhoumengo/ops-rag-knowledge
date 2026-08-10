from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from .models import SearchFilters, SearchRequest


class RAGAnythingInput(BaseModel):
    query: str = Field(..., description="需要从技术资料中检索的问题")
    mode: Literal["local", "global", "naive", "hybrid", "mix"] = "mix"
    status: list[str] = Field(default_factory=list, description="例如 Final")
    system: list[str] = Field(default_factory=list, description="例如 XT_1950Hi")
    software: list[str] = Field(default_factory=list, description="例如 5.1.0")
    destination: list[str] = Field(default_factory=list)
    document_code: list[str] = Field(default_factory=list, description="例如 OPS-100")
    vlm_enhanced: bool = Field(
        default=False, description="涉及图表、架构图、流程图或截图时设为 true"
    )


def _sync_await(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def create_tool(service: Any) -> Any:
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:
        raise RuntimeError("缺少 crewai；请先执行 `uv sync --extra dev`。") from exc

    class RAGAnythingTool(BaseTool):
        name: str = "multimodal_knowledge_search"
        description: str = (
            "搜索已入库的运维 PDF/Office 资料，返回答案、原始检索证据、来源和冲突。"
            "回答文档事实前必须调用；视觉问题必须设置 vlm_enhanced=true。"
        )
        args_schema: type[BaseModel] = RAGAnythingInput
        _service: Any = PrivateAttr()

        def __init__(self, rag_service: Any):
            super().__init__()
            self._service = rag_service

        async def _arun(self, **kwargs: Any) -> str:
            args = RAGAnythingInput.model_validate(kwargs)
            response = await self._service.search(
                SearchRequest(
                    query=args.query,
                    mode=args.mode,
                    filters=SearchFilters(
                        status=args.status,
                        system=args.system,
                        software=args.software,
                        destination=args.destination,
                        document_code=args.document_code,
                    ),
                    vlm_enhanced=args.vlm_enhanced,
                )
            )
            return response.model_dump_json(indent=2)

        def _run(self, **kwargs: Any) -> str:
            return _sync_await(self._arun(**kwargs))

    return RAGAnythingTool(service)
