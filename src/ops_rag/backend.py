from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from .manifest import filter_documents, load_manifest
from .models import DocumentRecord, Evidence, SearchRequest, SearchResponse
from .settings import Settings

GROUNDING_PROMPT = """You answer questions about operations documents.
Use only the retrieved knowledge. Never invent commands, parameters, versions, or steps.
Mention the source filename, document status/version, and page/figure when present.
If sources conflict, list each source separately. If evidence is insufficient, say so explicitly.
"""


def create_rag(settings: Settings) -> Any:
    """Create a RAG-Anything instance using an OpenAI-compatible endpoint."""
    if not settings.api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY；请复制 .env.example 为 .env 并配置模型。")

    try:
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig
    except ImportError as exc:
        raise RuntimeError(
            "缺少运行依赖；请先执行 `uv sync --extra dev`。"
        ) from exc

    def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body.setdefault("enable_thinking", settings.enable_thinking)
        return openai_complete_if_cache(
            settings.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=settings.api_key,
            base_url=settings.base_url,
            extra_body=extra_body,
            **kwargs,
        )

    def vision_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        image_data: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body.setdefault("enable_thinking", settings.enable_thinking)
        if messages:
            return openai_complete_if_cache(
                settings.vision_model,
                "",
                messages=messages,
                api_key=settings.api_key,
                base_url=settings.base_url,
                extra_body=extra_body,
                **kwargs,
            )
        if image_data:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
            vision_messages: list[dict[str, Any]] = []
            if system_prompt:
                vision_messages.append({"role": "system", "content": system_prompt})
            vision_messages.append({"role": "user", "content": content})
            return openai_complete_if_cache(
                settings.vision_model,
                "",
                messages=vision_messages,
                api_key=settings.api_key,
                base_url=settings.base_url,
                extra_body=extra_body,
                **kwargs,
            )
        return llm_model_func(
            prompt,
            system_prompt,
            history_messages,
            extra_body=extra_body,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=8192,
        model_name=settings.embedding_model,
        func=partial(
            openai_embed.func,
            model=settings.embedding_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        ),
    )
    config = RAGAnythingConfig(
        working_dir=str(settings.working_dir),
        parser=settings.parser,
        parse_method=settings.parse_method,
        parser_output_dir=str(settings.output_dir),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        max_concurrent_files=settings.max_workers,
        context_window=2,
        context_mode="page",
        max_context_tokens=3000,
        include_headers=True,
        include_captions=True,
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "top_k": int(__import__("os").getenv("TOP_K", "20")),
            "chunk_top_k": int(__import__("os").getenv("CHUNK_TOP_K", "10")),
            "max_total_tokens": int(
                __import__("os").getenv("MAX_TOTAL_TOKENS", "12000")
            ),
            "enable_llm_cache": True,
        },
    )
    # RAG-Anything 1.2.x snapshots LightRAG with dataclasses.asdict() when it
    # creates its modal processors. LightRAG 1.4.x keeps the per-role runtime
    # callables outside dataclass fields, so that snapshot misses
    # ``role_llm_funcs`` and multimodal entity extraction fails. Refresh the
    # processor configuration from LightRAG's runtime-aware builder.
    build_global_config = getattr(rag.lightrag, "_build_global_config", None)
    if callable(build_global_config):
        runtime_config = build_global_config()
        for processor in rag.modal_processors.values():
            processor.global_config = runtime_config
    return rag


def _allowed_sources(docs: list[DocumentRecord]) -> str:
    if not docs:
        return "没有文档符合过滤条件。"
    return "\n".join(
        f"- {doc.file_name} | document_id={doc.document_id} | "
        f"status={doc.status or 'unknown'} | version={doc.version or 'unknown'}"
        for doc in docs
    )


def _extract_evidence(context: str, documents: list[DocumentRecord]) -> list[Evidence]:
    """Best-effort evidence extraction from LightRAG's context response."""
    if not context.strip():
        return []

    evidence: list[Evidence] = []
    lowered = context.casefold()
    for doc in documents:
        if doc.file_name.casefold() not in lowered and doc.document_id.casefold() not in lowered:
            continue
        position = max(lowered.find(doc.file_name.casefold()), 0)
        start = position
        end = min(len(context), position + len(doc.file_name) + 760)
        segment = context[start:end]
        page_match = re.search(
            r"(?:page_idx|page index)[\"':=\s]+(\d+)", segment, re.I
        )
        figure_match = re.search(
            r"\b(?:figure|fig\.?)\s*([A-Z0-9._-]+)", segment, re.I
        )
        content_type_match = re.search(
            r"[\"']type[\"']\s*:\s*[\"'](text|image|table|equation)[\"']",
            segment,
            re.I,
        )
        excerpt = re.sub(r"\s+", " ", segment).strip()
        evidence.append(
            Evidence(
                document_id=doc.document_id,
                file_name=doc.file_name,
                status=doc.status,
                version=doc.version,
                page=int(page_match.group(1)) + 1 if page_match else None,
                figure_id=figure_match.group(1) if figure_match else None,
                content_type=content_type_match.group(1) if content_type_match else None,
                excerpt=excerpt[:1000],
            )
        )
    if not evidence:
        evidence.append(Evidence(excerpt=re.sub(r"\s+", " ", context).strip()[:1500]))
    return evidence


def _document_for_item(
    item: dict[str, Any], documents: list[DocumentRecord]
) -> DocumentRecord | None:
    full_doc_id = str(item.get("full_doc_id") or item.get("document_id") or "")
    if full_doc_id:
        for doc in documents:
            if doc.document_id == full_doc_id:
                return doc

    raw_paths = str(item.get("file_path") or "")
    path_names = {
        Path(value.strip()).name.casefold()
        for value in raw_paths.split("<SEP>")
        if value.strip()
    }
    for doc in documents:
        if doc.file_name.casefold() in path_names:
            return doc
    return None


async def _structured_evidence(
    query_data: dict[str, Any],
    documents: list[DocumentRecord],
    text_chunks: Any,
    limit: int = 12,
) -> list[Evidence]:
    """Build evidence from LightRAG's stable structured retrieval result."""
    data = query_data.get("data", {}) if isinstance(query_data, dict) else {}
    retrieved_chunks = list(data.get("chunks", []))

    # Structured entity results retain source chunk IDs even when the rendered
    # context omits its Reference List. Resolve those IDs back to stored chunks.
    source_ids: list[str] = []
    for item in [*data.get("entities", []), *data.get("relationships", [])]:
        source_ids.extend(
            value.strip()
            for value in str(item.get("source_id") or "").split("<SEP>")
            if value.strip()
        )
    existing_ids = {
        str(item.get("chunk_id") or item.get("_id") or "")
        for item in retrieved_chunks
    }
    missing_ids = list(dict.fromkeys(value for value in source_ids if value not in existing_ids))
    get_by_ids = getattr(text_chunks, "get_by_ids", None)
    if missing_ids and callable(get_by_ids):
        stored_chunks = await get_by_ids(missing_ids)
        for chunk_id, chunk in zip(missing_ids, stored_chunks, strict=False):
            if chunk:
                retrieved_chunks.append({"chunk_id": chunk_id, **chunk})

    evidence: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for item in retrieved_chunks:
        doc = _document_for_item(item, documents)
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if doc is None or not content:
            continue
        chunk_id = str(item.get("chunk_id") or item.get("_id") or "")
        identity = (doc.document_id, chunk_id or content[:200])
        if identity in seen:
            continue
        seen.add(identity)
        raw_page = item.get("page_idx")
        page = int(raw_page) + 1 if isinstance(raw_page, int) else None
        original_type = item.get("original_type")
        content_type = str(original_type) if original_type else None
        evidence.append(
            Evidence(
                document_id=doc.document_id,
                file_name=doc.file_name,
                status=doc.status,
                version=doc.version,
                page=page,
                content_type=content_type,
                excerpt=content[:1800],
            )
        )
        if len(evidence) >= limit:
            break
    return evidence


def _detect_conflicts(evidence: list[Evidence]) -> list[str]:
    # Multiple document states are provenance, not proof that their claims
    # conflict. Evidence currently does not contain normalized claim pairs, so
    # remain conservative instead of reporting false semantic conflicts.
    return []


def _grounded_prompt(query: str, evidence: list[Evidence]) -> str:
    rendered = "\n\n".join(
        f"来源：{item.file_name} | status={item.status or 'unknown'} | "
        f"version={item.version or 'unknown'} | page={item.page or 'unknown'}\n"
        f"内容：{item.excerpt}"
        for item in evidence
    )
    return f"用户问题：{query}\n\n以下是检索到的证据：\n{rendered}"


class RAGService:
    def __init__(self, rag: Any, settings: Settings):
        self.rag = rag
        self.settings = settings
        self.manifest = load_manifest(settings.manifest_path)

    async def _ensure_ready(self) -> None:
        ensure = getattr(self.rag, "_ensure_lightrag_initialized", None)
        if callable(ensure):
            result = await ensure()
            if isinstance(result, dict) and not result.get("success", False):
                raise RuntimeError(result.get("error", "LightRAG 初始化失败"))

        lightrag = getattr(self.rag, "lightrag", None)
        build_global_config = getattr(lightrag, "_build_global_config", None)
        if callable(build_global_config):
            runtime_config = build_global_config()
            # RAG-Anything 1.2.x passes ``lightrag.__dict__`` directly to
            # LightRAG's entity extraction for multimodal chunks. The
            # per-role callables are exposed as a property in LightRAG 1.4.x,
            # so they are absent from ``__dict__`` unless copied explicitly.
            role_llm_funcs = runtime_config.get("role_llm_funcs")
            if role_llm_funcs:
                lightrag.__dict__["role_llm_funcs"] = role_llm_funcs
            for processor in self.rag.modal_processors.values():
                processor.global_config = runtime_config

    async def ingest(self, documents: list[DocumentRecord] | None = None) -> list[dict[str, Any]]:
        await self._ensure_ready()
        selected = documents or [
            doc for doc in self.manifest.documents if doc.content_profile != "terminology"
        ]
        results: list[dict[str, Any]] = []
        for doc in selected:
            try:
                parser_options: dict[str, Any] = {}
                if self.settings.parser_backend:
                    parser_options["backend"] = self.settings.parser_backend
                await self.rag.process_document_complete(
                    file_path=doc.source_path,
                    output_dir=str(self.settings.output_dir),
                    parse_method=doc.parse_method,
                    display_stats=True,
                    doc_id=doc.document_id,
                    **parser_options,
                )
                results.append(
                    {"document_id": doc.document_id, "file_name": doc.file_name, "ok": True}
                )
            except Exception as exc:
                results.append(
                    {
                        "document_id": doc.document_id,
                        "file_name": doc.file_name,
                        "ok": False,
                        "error": str(exc),
                    }
                )
        return results

    async def search(self, request: SearchRequest) -> SearchResponse:
        await self._ensure_ready()
        candidates = filter_documents(self.manifest.documents, request.filters)
        candidates = [doc for doc in candidates if doc.content_profile != "terminology"]
        manifest_matched = bool(candidates)
        full_docs = getattr(getattr(self.rag, "lightrag", None), "full_docs", None)
        filter_keys = getattr(full_docs, "filter_keys", None)
        if callable(filter_keys) and candidates:
            missing_ids = await filter_keys({doc.document_id for doc in candidates})
            candidates = [
                doc for doc in candidates if doc.document_id not in missing_ids
            ]
        if not candidates:
            response = SearchResponse(
                answer="无法从已入库资料确认：没有已入库文档符合指定过滤条件。",
                applied_filters=request.filters,
                warning=(
                    "filters_matched_no_ingested_documents"
                    if manifest_matched
                    else "filters_matched_no_documents"
                ),
            )
            self._audit(request, response)
            return response

        scoped_query = (
            f"{request.query}\n\n只允许使用以下候选来源；若检索内容来自其他文件则忽略：\n"
            f"{_allowed_sources(candidates)}"
        )
        lightrag = getattr(self.rag, "lightrag", None)
        query_data_func = getattr(lightrag, "aquery_data", None)
        query_data: dict[str, Any] | None = None
        if callable(query_data_func):
            from lightrag import QueryParam

            query_data = await query_data_func(
                request.query,
                QueryParam(mode=request.mode, include_references=True),
            )
            evidence = await _structured_evidence(
                query_data,
                candidates,
                getattr(lightrag, "text_chunks", None),
            )
            context = json.dumps(query_data, ensure_ascii=False)
        else:
            context = await self.rag.aquery(
                scoped_query,
                mode=request.mode,
                vlm_enhanced=False,
                only_need_context=True,
                include_references=True,
            )
            evidence = _extract_evidence(str(context), candidates)
        if not context or not str(context).strip():
            response = SearchResponse(
                answer="无法从已入库资料确认：检索未返回证据。",
                evidence=[],
                retrieval_context="",
                applied_filters=request.filters,
                warning="empty_retrieval",
            )
            self._audit(request, response)
            return response

        if evidence and callable(getattr(self.rag, "llm_model_func", None)):
            answer = await self.rag.llm_model_func(
                _grounded_prompt(request.query, evidence),
                system_prompt=GROUNDING_PROMPT,
            )
        else:
            answer = await self.rag.aquery(
                scoped_query,
                mode=request.mode,
                vlm_enhanced=request.vlm_enhanced,
                system_prompt=GROUNDING_PROMPT,
                include_references=True,
            )
        response = SearchResponse(
            answer=str(answer),
            evidence=evidence,
            conflicts=_detect_conflicts(evidence),
            retrieval_context=str(context)[:12000],
            applied_filters=request.filters,
            warning=(
                None
                if any(item.file_name for item in evidence)
                else "context_returned_without_parseable_source_metadata"
            ),
        )
        self._audit(request, response)
        return response

    def _audit(self, request: SearchRequest, response: SearchResponse) -> None:
        self.settings.audit_log.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json", exclude={"retrieval_context"}),
        }
        with self.settings.audit_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def close(self) -> None:
        finalize = getattr(self.rag, "finalize_storages", None)
        if finalize:
            await finalize()
