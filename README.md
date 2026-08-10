# 运维多模态 RAG 知识库

基于 RAG-Anything、LightRAG 和 CrewAI 的运维知识库原型。项目支持 PDF、Office
文档、表格和图片的摄取，使用 manifest 保存来源元数据，并在查询时返回答案、证据、
冲突信息和审计记录。

## 主要能力

- 使用 RAG-Anything / LightRAG 完成多模态解析、知识图谱构建与混合检索；
- 使用 CrewAI 约束回答规范、工具调用和证据引用；
- 在检索前按状态、系统、软件、目的地和文档编号过滤来源；
- 对同一编号的多个版本或状态显式报告冲突；
- 将索引、解析缓存和查询审计统一保存在本地 `runtime/` 目录。

## 仓库结构

```text
.
├── config/                 # 回答规范和 manifest 示例
├── data/                   # 本地源文档（不提交）
├── runtime/                # 索引、缓存和日志（不提交）
├── src/ops_rag/            # Python 包
├── tests/                  # 单元测试
├── .env.example            # 环境变量模板
└── pyproject.toml
```

## 环境要求

- Python 3.10–3.13；
- [`uv`](https://docs.astral.sh/uv/)；
- `pdfinfo` 和 `pdftotext`（通常由 Poppler 提供）；
- LibreOffice（处理 Office 文档时需要）。

## 快速开始

```bash
git clone <your-repository-url>
cd ops-rag-knowledge
cp .env.example .env
uv sync --extra dev
```

把待摄取的文档放入 `data/`，然后根据模型服务修改 `.env`。凭证也可以放在本地
Workspace CSV 中，并通过 `OPS_RAG_CREDENTIAL_FILE` 指向该文件。`.env`、凭证 CSV、
业务文档和运行产物均已排除在 Git 之外。

检查环境并生成 manifest：

```bash
uv run ops-rag doctor
uv run ops-rag manifest
```

生成的 `config/manifest.yaml` 只供本地使用。仓库中的
`config/manifest.example.yaml` 展示其结构。

首次摄取可能下载解析模型并调用远程 LLM/VLM，耗时和费用取决于所选模型服务：

```bash
uv run ops-rag ingest
```

若修改 Embedding 模型或维度，请使用新的 `OPS_RAG_WORKING_DIR`，不要让不同维度的
向量覆盖已有索引。

## 查询

直接查询 RAG：

```bash
uv run ops-rag query "某故障适用于哪些系统？" \
  --status Final --document-code OPS-100

uv run ops-rag query "培训资料中的架构图说明了什么？" --vlm
```

通过 CrewAI Agent 查询：

```bash
uv run ops-rag ask "比较同一文档的不同版本，不要合并冲突。"
```

## 网页聊天界面

启动带机器人形象的本地聊天页面，对话时复用 `ops-rag ask` 的 CrewAI+RAG 流程：

```bash
uv run ops-rag web --port 8000
```

浏览器打开 http://127.0.0.1:8000 即可与机器人对话。后端在进程内常驻
知识图谱与 Agent，首次提问会加载索引和回答规范（约 30–60 秒），之后每次
提问复用同一实例，避免每条消息都重新加载。前端为 `src/ops_rag/web_static/`
下的静态页面，后端接口为 `POST /api/chat` 与 `GET /api/health`。

页面还支持直接上传文档（PDF / Office / 表格）。上传后会自动写入
`data/`、更新 manifest 并摄取新文件，输入框上方会显示解析进度
（“文件解析中 → 索引构建中 → 完成”）。摄取期间对话会排队等待，
避免同一时间读写索引。

## 行为边界

- metadata 过滤在检索前缩小允许来源，候选文件会写入查询约束；
- Tool 先通过 `only_need_context=True` 获取证据，再生成答案；
- 无匹配文档或检索结果为空时直接拒答；
- 同编号命中多个状态、版本或 Problem ID 时生成 `conflicts`；
- `page_idx` 是零基索引，输出 evidence 时转换为用户可见页码；
- 每次查询写入 `runtime/query_audit.jsonl`，但不写入原始长上下文；
- evidence 解析属于 best-effort：无法识别来源元数据时会返回警告。

## 开发与测试

```bash
uv run pytest
uv run ruff check .
```

提交前请确认 `git status --ignored` 中的 `.env`、源文档、manifest、运行索引和日志均处于
ignored 状态。若计划公开仓库，还应确认你有权发布所有被跟踪的资料，并按项目需要选择
合适的开源许可证。
