from __future__ import annotations

from pathlib import Path
from typing import Any

from .settings import Settings
from .tool import create_tool


def _stable_knowledge(project_root: Path) -> str:
    parts = [
        (project_root / "config" / "answer_policy.md").read_text(encoding="utf-8")
    ]
    terminology = project_root / "data" / "terminology" / "专业术语库.csv"
    if terminology.exists():
        parts.append("# 专业术语表\n\n" + terminology.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def build_crew(service: Any, settings: Settings) -> Any:
    try:
        from crewai import Agent, Crew, LLM, Process, Task
        from crewai.knowledge.source.string_knowledge_source import (
            StringKnowledgeSource,
        )
    except ImportError as exc:
        raise RuntimeError("缺少 crewai；请先执行 `uv sync --extra dev`。") from exc

    project_root = Path(__file__).resolve().parents[2]
    stable_source = StringKnowledgeSource(content=_stable_knowledge(project_root))
    rag_tool = create_tool(service)
    embedder = {
        "provider": "openai",
        "config": {
            "model": settings.embedding_model,
            "api_key": settings.api_key,
            "api_base": settings.base_url,
            "dimensions": settings.embedding_dim,
        },
    }
    crew_llm = LLM(
        model=settings.crewai_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        extra_body={"enable_thinking": settings.enable_thinking},
    )
    analyst = Agent(
        role="多模态运维知识分析员",
        goal="只基于已入库证据回答问题，并准确区分文档状态、版本和适用范围",
        backstory=(
            "你负责审阅 MO/MG、培训模块、表格、架构图和操作截图。"
            "所有文档事实必须先调用 multimodal_knowledge_search。"
            "证据不足时拒绝推测，资料冲突时逐项列出来源。"
        ),
        tools=[rag_tool],
        llm=crew_llm,
        allow_delegation=False,
        max_iter=6,
        verbose=True,
    )
    task = Task(
        description=(
            "回答用户问题：{question}\n\n"
            "强制要求：\n"
            "1. 文档事实必须先调用 multimodal_knowledge_search；\n"
            "2. 涉及图表、架构图、流程图或截图时设置 vlm_enhanced=true；\n"
            "3. 只根据工具返回的 answer 与 evidence 作答；\n"
            "4. 标注文件名、状态、版本、页码和图号（字段存在时）；\n"
            "5. evidence 为空时回答“无法从已入库资料确认”；\n"
            "6. conflicts 非空时逐项列出，不合并结论。"
        ),
        expected_output="中文答案，末尾附“证据来源”和“冲突/不确定性”小节。",
        agent=analyst,
    )
    return Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        knowledge_sources=[stable_source],
        embedder=embedder,
        verbose=True,
        memory=False,
    )
