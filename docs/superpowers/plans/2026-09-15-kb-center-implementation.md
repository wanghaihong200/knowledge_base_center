# 掌柜智库 RAG 知识库 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完整实现掌柜智库 RAG 知识库——导入流水线 7 节点 + 检索流水线 7 节点 + Web/API 层 + 三层测试，端到端可验收。

**Architecture:** FastAPI 双服务（导入 8000 / 查询 8001）+ LangGraph 节点流水线（模板方法基类、TypedDict 状态传递）+ Milvus（纯稠密向量）/ MongoDB（会话历史）/ MinIO（图片与文档对象存储）/ MinerU 云 API（PDF→MD）/ 智谱 BigModel 全 API 模型（glm-5.3-flash LLM/VL @ coding 端点；embedding-3 / rerank @ 标准 paas/v4 端点，key 复用）。

**Tech Stack:** Python 3.11、LangGraph 1.2、LangChain 1.4（langchain-openai）、pymilvus、pymongo、minio、openai、openai-agents、httpx、FastAPI、pytest。

**Spec:** `doc/origin_doc/1.笔记/01-20`（每篇含节点完整实现代码，本计划只写"与笔记的差异 + 精确签名 + 测试"，实现体按笔记）；架构决策见 `docs/adr/0001-api-embedding-over-local-bge.md`；术语见 `CONTEXT.md`。

## Global Constraints

- **全 API，无本地模型**：禁止引入 torch / FlagEmbedding / modelscope / 本地 MinerU / dashscope SDK。Embedding = 智谱 `embedding-3`（`https://open.bigmodel.cn/api/paas/v4/embeddings`），`EMBEDDING_DIM=1024`（embedding-3 仅支持 256/512/1024/2048），纯稠密。Rerank = 智谱 `rerank`（`.../paas/v4/rerank`）。LLM/VL/主体识别 = `glm-5.3-flash`（coding 端点）。Embedding/Rerank 复用 `OPENAI_API_KEY`。
- **Milvus schema 无 `sparse_vector` 字段**；检索用 `client.search`（dense, COSINE），不用 `WeightedRanker`/`AnnSearchRequest`。
- **目录结构**：保持现有 `config/`、`processor/import_processor/`、`processor/query_processor/`、`utils/`、`web/`、`tool/`；禁止创建 `knowledge.*` 包路径。
- **config 拆分约定**：服务级配置 = `config/<name>_config.py`（dataclass + 模块级单例实例）；导入流程专属配置 = `processor/import_processor/config.py`（`ImportConfig` + `get_config()`）。删除 `config/config.py`。
- **节点约定**：类继承基类，类属性 `name` 覆盖；核心方法 `process(self, state)`；内部阶段方法 `_step_N_xxx`；异常经导入侧基类包装为 `ImportProcessError`。
- **节点名**：`node_embedding`（不叫 node_bge_embedding）；`task_utils._NODE_NAME_TO_CN` 中 `"node_bge_embedding"` 键同步改为 `"node_embedding"`。
- **commit**：Conventional Commits（feat/test/chore/docs），**不加** Co-Authored-By trailer（用户全局规则，优先级最高）。
- **中文编码**：所有新文件 UTF-8；requirements.txt 重写为 UTF-8（现文件是 UTF-16）。
- **测试分层标记**：`@pytest.mark.unit` / `integration` / `e2e`，pyproject 配置 marker；integration/e2e 在依赖服务不可达时 skip。
- **降级原则**：网络搜索 MCP、rerank API 调用失败 → 记日志、返回空/原样，流程继续，不抛断主链路。

## 文件全景图

```
config/
├── lm_config.py            LLMConfig: base_url/api_key/llm_model/vl_model/item_model/llm_temperature
├── embedding_config.py     EmbeddingConfig: api_base/api_key(复用OPENAI_API_KEY)/embedding_model="embedding-3"/embedding_dim(1024)/embedding_batch_size(8)
├── milvus_config.py        MilvusConfig: milvus_url/chunks_collection/item_name_collection
├── minio_config.py         MinIOConfig: endpoint/access_key/secret_key/bucket_name/img_dir="upload-images"
├── mineru_config.py        MineruConfig: base_url/api_token
├── mcp_config.py           McpConfig: mcp_base_url(MCP_ZHIPU_BASE_URL)/api_key(复用OPENAI_API_KEY)
├── reranker_config.py      RerankerConfig: api_base/api_key(复用OPENAI_API_KEY)/text_rerank_model="rerank"
└── __init__.py
processor/import_processor/
├── config.py               ImportConfig(自 config/config.py 迁入) + get_config()
├── base.py                 [修改] import 路径修正
├── state.py  exceptions.py [保留]
├── main_graph.py           KBImportWorkflow
└── nodes/
    ├── node_entry.py            NodeEntry            name="node_entry"
    ├── node_pdf_to_md.py        NodePDFToMD          name="node_pdf_to_md"
    ├── node_md_img.py           NodeMDImg            name="node_md_img"
    ├── node_document_split.py   NodeDocumentSplit    name="node_document_split"
    ├── node_item_name_recognition.py NodeItemNameRecognition name="node_item_name_recognition"
    ├── node_embedding.py        NodeEmbedding        name="node_embedding"
    └── node_import_milvus.py    NodeImportMilvus     name="node_import_milvus"
processor/query_processor/
├── __init__.py  state.py    QueryGraphState(补 hyde_doc)
├── base.py                 NodeBase（新建，简化版基类）
├── main_graph.py           KBQueryWorkflowV2
├── prompt/
│   ├── item_name_confirm.py     ITEM_NAME_EXTRACT_SYSTEM_PROMPT / ITEM_NAME_EXTRACT_TEMPLATE
│   ├── item_name_recognition.py ITEM_NAME_SYSTEM_PROMPT / ITEM_NAME_USER_PROMPT_TEMPLATE
│   ├── search_embedding_hyde.py HYDE_PROMPT
│   └── answer_prompt.py         ANSWER_PROMPT
└── nodes/
    ├── node_item_name_confirm.py     NodeItemNameConfirm
    ├── node_search_embedding.py      NodeSearchEmbedding
    ├── node_search_embedding_hyde.py NodeSearchEmbeddingHyde
    ├── node_web_search_mcp.py        NodeWebSearchMcp
    ├── node_rrf.py                   NodeRrf
    ├── node_rerank.py                NodeRerank
    └── node_answer_output.py         NodeAnswerOutput  ← 自根目录 g_node_answer_output.py 迁入并修正 import
utils/
├── llm_utils.py  embedding_utils.py  milvus_utils.py  minio_utils.py
├── mongo_history_utils.py  reranker_http_utils.py  json_format_utils.py
├── sse_utils.py  task_utils.py [已有，task_utils 仅改节点名映射]
web/api/import_service.py  web/api/query_service.py [import 对齐后自然可用]
main.py                    [重写] CLI: python main.py import|query|all
tests/
├── conftest.py             服务可达性探测 fixtures
├── unit/…                  纯逻辑 + mock 外部
├── integration/…           docker 真服务
└── e2e/…                   httpx 打真实 API
```

---

### Task 0: 基线提交

**Files:**
- Modify: `.gitignore`（追加 `.idea/`、`.playwright-mcp/`、`temp-files/`、`data/`、`__pycache__/`、`*.pyc`、`.pytest_cache/`）
- Commit: 当前全部已有代码 + `CONTEXT.md` + `docs/adr/0001*.md` + 本计划文件

- [ ] **Step 1**: 追加 .gitignore 条目（见上）
- [ ] **Step 2**:
```bash
git add -A && git commit -m "chore: 项目基线——骨架代码、Web层、文档与实施计划"
```
- [ ] **Step 3**: `git status` 确认工作区干净、`.env` 未入库（`git ls-files | grep -c "\.env$"` 应为 0，`.env.example` 除外）

---

### Task 1: 依赖与配置层

**Files:**
- Rewrite: `requirements.txt`（UTF-8）
- Create: `config/lm_config.py`、`embedding_config.py`、`milvus_config.py`、`minio_config.py`、`mineru_config.py`、`mcp_config.py`、`reranker_config.py`
- Create: `processor/import_processor/config.py`；Delete: `config/config.py`
- Modify: `processor/import_processor/base.py:11-13`（import 修正）
- Modify: `utils/task_utils.py:35`（节点名映射）
- Modify: `.env.example` 与 `.env`（追加变量）
- Test: `tests/unit/test_config.py`

**Interfaces (Produces):**
```python
# config/lm_config.py
@dataclass class LLMConfig: base_url:str; api_key:str; llm_model:str; vl_model:str; item_model:str; llm_temperature:float
lm_config: LLMConfig  # 环境变量: OPENAI_API_BASE/OPENAI_API_KEY/LLM_DEFAULT_MODEL/VL_MODEL/ITEM_MODEL/LLM_DEFAULT_TEMPERATURE
# config/embedding_config.py（智谱）
@dataclass class EmbeddingConfig: api_base:str; api_key:str; embedding_model:str="embedding-3"; embedding_dim:int=1024; embedding_batch_size:int=8
embedding_config: EmbeddingConfig  # EMBEDDING_API_BASE/OPENAI_API_KEY/EMBEDDING_MODEL/EMBEDDING_DIM
# config/milvus_config.py
@dataclass class MilvusConfig: milvus_url:str; chunks_collection:str; item_name_collection:str
milvus_config: MilvusConfig  # MILVUS_URL/CHUNKS_COLLECTION/ITEM_NAME_COLLECTION
# config/minio_config.py
@dataclass class MinIOConfig: endpoint:str; access_key:str; secret_key:str; bucket_name:str; img_dir:str="upload-images"
minio_config: MinIOConfig  # MINIO_ENDPOINT/MINIO_ACCESS_KEY/MINIO_SECRET_KEY/MINIO_BUCKET_NAME/MINIO_IMG_DIR
# config/mineru_config.py
@dataclass class MineruConfig: base_url:str; api_token:str
mineru_config: MineruConfig  # MINERU_BASE_URL/MINERU_API_TOKEN
# config/mcp_config.py（智谱 web_search_prime MCP）
@dataclass class McpConfig: mcp_base_url:str; api_key:str
mcp_config: McpConfig  # MCP_ZHIPU_BASE_URL/OPENAI_API_KEY
# config/reranker_config.py（智谱）
@dataclass class RerankerConfig: api_base:str; api_key:str; text_rerank_model:str="rerank"
reranker_config: RerankerConfig  # RERANK_API_BASE/OPENAI_API_KEY/TEXT_RERANK_MODEL
# processor/import_processor/config.py —— 原 ImportConfig 原样迁入，仅两处改动:
#   embedding_dim 默认 "1024"；embedding_batch_size=8（智谱批量上限）；删 minio_img_dir 无关项不加
```

- [ ] **Step 1: 写失败测试** `tests/unit/test_config.py`
```python
import pytest
from config.lm_config import lm_config
from config.embedding_config import embedding_config
from config.milvus_config import milvus_config
from config.minio_config import minio_config
from config.reranker_config import reranker_config
from processor.import_processor.config import get_config

pytestmark = pytest.mark.unit

def test_lm_config_from_env():
    assert lm_config.api_key, "OPENAI_API_KEY 未配置"
    assert lm_config.llm_model == "qwen-flash"
    assert lm_config.llm_temperature == 0.1

def test_embedding_config_dense_only():
    assert embedding_config.api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert embedding_config.embedding_model == "embedding-3"
    assert embedding_config.embedding_dim == 1024
    assert embedding_config.embedding_batch_size == 8

def test_reranker_config_zhipu():
    assert reranker_config.api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert reranker_config.text_rerank_model == "rerank"

def test_import_config_singleton():
    assert get_config() is get_config()
    assert get_config().max_content_length == 2000
    assert get_config().embedding_dim == 1024
```
- [ ] **Step 2**: `pytest tests/unit/test_config.py -v` → 预期 FAIL（模块不存在）
- [ ] **Step 3**: 实现 7 个 config 文件 + 迁移 ImportConfig（模板统一如下，字段值取自 Interfaces 与 .env.example；每个文件形如）
```python
"""LLM 服务配置"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()

@dataclass
class LLMConfig:
    base_url: str = os.getenv("OPENAI_API_BASE", "")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_DEFAULT_MODEL", "qwen-flash")
    vl_model: str = os.getenv("VL_MODEL", "qwen3-vl-flash")
    item_model: str = os.getenv("ITEM_MODEL", "qwen-flash")
    llm_temperature: float = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))

lm_config = LLMConfig()
```
  ImportConfig 迁移：`config/config.py` 全文移至 `processor/import_processor/config.py`，改默认 `embedding_dim` env fallback 为 `"1024"`、`embedding_batch_size: int = 8`；删除 `config/config.py`。`base.py` 头部三行 import 修正为：
```python
from processor.import_processor.config import ImportConfig, get_config
from processor.import_processor.exceptions import ImportProcessError
from utils.task_utils import add_running_task, add_done_task
```
  `task_utils.py` 映射：`"node_bge_embedding": "向量生成",` → `"node_embedding": "向量生成",`
- [ ] **Step 4**: `.env`/`.env.example` 对齐（已完成于 2026-09-15，执行时核对即可）：`EMBEDDING_API_BASE/RERANK_API_BASE=https://open.bigmodel.cn/api/paas/v4`、`EMBEDDING_DIM=1024`、`EMBEDDING_MODEL=embedding-3`、`TEXT_RERANK_MODEL=rerank`、`DATA_BASED_ROOT_DIR=./data`、`MINIO_IMG_DIR=upload-images`；BGE/MODELSCOPE 段已删除
- [ ] **Step 5**: 重写 `requirements.txt`（UTF-8，保留现 venv 已装版本，新增未装项）：
```
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
colorlog>=6.9.0
fastapi==0.141.1
grandalf==0.8
httpx>=0.28.0
langchain==1.4.0
langchain-core==1.6.3
langchain-openai==1.6.2
langchain-text-splitters==1.1.2
langgraph==1.2.11
minio==7.2.20
numpy==2.4.6
openai==3.13.0
openai-agents>=0.2.0
pydantic==2.13.5
pymilvus>=2.5.6
pymongo==4.18.1
python-dotenv==1.2.3
PyYAML==6.0.3
pytest>=8.0.0
pytest-asyncio>=0.25.0
requests==2.34.2
tiktoken==0.14.0
uvicorn[standard]>=0.30.0
```
  执行 `.venv/Scripts/python -m pip install -r requirements.txt`（openai-agents/pymilvus/colorlog/uvicorn/pytest/httpx 等会新装；注意 GitHub 不通不影响 pip 默认源）。
- [ ] **Step 6**: `pytest tests/unit/test_config.py -v` → PASS
- [ ] **Step 7**:
```bash
git add -A && git commit -m "feat: 配置层拆分为服务级 dataclass 单例,ImportConfig 迁入导入流程,依赖清单重写"
```

---

### Task 2: utils 工具层

**Files:**
- Create: `utils/llm_utils.py`、`utils/embedding_utils.py`、`utils/milvus_utils.py`、`utils/minio_utils.py`、`utils/mongo_history_utils.py`、`utils/reranker_http_utils.py`、`utils/json_format_utils.py`
- Test: `tests/unit/test_embedding_utils.py`、`tests/unit/test_milvus_utils.py`、`tests/unit/test_mongo_history_utils.py`、`tests/integration/test_milvus_roundtrip.py`、`tests/integration/test_mongo_history.py`
- Modify: `tests/conftest.py`（新建：服务可达性探测）

**Interfaces (Produces):**
```python
# utils/llm_utils.py
def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI
#   (model, json_mode) 二元组缓存; extra_body={"enable_thinking": False};
#   json_mode 时 model_kwargs={"response_format": {"type": "json_object"}}

# utils/embedding_utils.py —— 全 API 版(与笔记 BGE 版不同,签名简化)
def generate_embeddings(texts: list[str]) -> list[list[float]]
#   OpenAI 兼容 client.embeddings.create(model=embedding_config.embedding_model,
#   input=batch, dimensions=embedding_config.embedding_dim); 按 embedding_batch_size=10 分批拼接

# utils/milvus_utils.py
def get_milvus_client() -> MilvusClient            # 单例
def escape_milvus_string(value: str) -> str        # 转义 \ " '
def ensure_collection(client, collection_name: str, vector_dim: int) -> None
#   不存在则建(name含"item"→item schema,否则chunks schema,见下); 存在但 dim 不符 → drop+重建
def vector_search(client, collection_name: str, query_vector: list[float],
                  expr: str | None = None, limit: int = 10,
                  output_fields: list[str] | None = None) -> list[dict]
#   client.search(anns_field="dense_vector", search_params={"metric_type":"COSINE"}, ...) 失败返回 []

# utils/minio_utils.py
def get_minio_client() -> Minio   # 模块级单例; 桶不存在则建; 设 Public Read 桶策略(s3:GetObject)

# utils/mongo_history_utils.py
def save_chat_message(session_id, role, text, rewritten_query="", item_names=None,
                      image_urls=None, message_id=None) -> str   # 返回 message_id(str(ObjectId)); 有 id 走 update_one $set
def get_recent_messages(session_id, limit=10) -> list[dict]        # sort ts ASCENDING
def update_message_item_names(ids: list[str], item_names: list[str]) -> None  # update_many $in
def clear_history(session_id) -> int
#   文档: {session_id, role, text, rewritten_query, item_names, image_urls, ts(秒级)}
#   集合 chat_message, 复合索引 (session_id:1, ts:-1), 库 MONGO_DB_NAME

# utils/reranker_http_utils.py（智谱 rerank，httpx POST）
def rerank_documents(query: str, documents: list[str]) -> list[float]
#   POST {reranker_config.api_base}/rerank，header: Authorization: Bearer {reranker_config.api_key}
#   json: {"model": reranker_config.text_rerank_model, "query": query,
#          "documents": documents, "top_n": len(documents)}
#   响应 {"data": [{"index": int, "relevance_score": float}, ...]}，按 index 回填为与输入等长的分数列表;
#   非 2xx 或异常 → 记日志并返回与输入等长的 0.0 列表(降级)

# utils/json_format_utils.py
def format_json(obj) -> str        # ensure_ascii=False, indent=2, ObjectId/datetime 容错
def serialize_json(obj) -> str     # 同上但紧凑
```

**kb_chunks schema**（`ensure_collection` 内建；dim 动态取 `vector_dim`）:
```python
schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field("chunk_id", DataType.INT64, is_primary=True)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("title", DataType.VARCHAR, max_length=100)
schema.add_field("parent_title", DataType.VARCHAR, max_length=100)
schema.add_field("part", DataType.INT8)
schema.add_field("file_title", DataType.VARCHAR, max_length=100)
schema.add_field("item_name", DataType.VARCHAR, max_length=100)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=vector_dim)
index_params = client.prepare_index_params()
index_params.add_index("dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
```
**kb_item_names schema**：`pk INT64 pk`、`file_title VARCHAR 100`、`item_name VARCHAR 100`、`dense_vector FLOAT_VECTOR dim`；索引同上。

- [ ] **Step 1: 写失败测试**（unit：mock 外部；integration：真服务）
```python
# tests/unit/test_milvus_utils.py
import pytest
from utils.milvus_utils import escape_milvus_string
pytestmark = pytest.mark.unit

def test_escape_milvus_string():
    assert escape_milvus_string("a'b\"c\\d") == "a\\'b\\\"c\\\\d"

# tests/unit/test_embedding_utils.py
import pytest
from unittest.mock import patch, MagicMock
from utils import embedding_utils
pytestmark = pytest.mark.unit

def test_generate_embeddings_batches_and_order():
    fake = MagicMock()
    def make(data):  # 12 条输入 → 两批(10+2)
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[float(i)] * 4) for i in range(len(data))]
        return resp
    fake.embeddings.create.side_effect = lambda **kw: make(kw["input"])
    with patch.object(embedding_utils, "_get_openai_client", return_value=fake):
        result = embedding_utils.generate_embeddings([str(i) for i in range(12)])
    assert len(result) == 12
    assert fake.embeddings.create.call_count == 2          # 10 + 2 两批
    assert result[11] == [11.0] * 4                        # 顺序保持
```
```python
# tests/integration/test_milvus_roundtrip.py
import pytest
pytestmark = [pytest.mark.integration]
from utils.milvus_utils import get_milvus_client, ensure_collection, vector_search
from config.milvus_config import milvus_config

def test_insert_and_search_roundtrip():
    client = get_milvus_client()
    dim = 8
    ensure_collection(client, milvus_config.chunks_collection + "_it", dim)
    rows = [{"content": "路由器默认地址是192.168.1.1", "title": "t", "parent_title": "",
             "part": 0, "file_title": "it_doc", "item_name": "IT测试路由",
             "dense_vector": [1.0] + [0.0] * (dim - 1)}]
    client.insert(milvus_config.chunks_collection + "_it", rows)
    hits = vector_search(client, milvus_config.chunks_collection + "_it",
                         [1.0] + [0.0] * (dim - 1), limit=1,
                         output_fields=["content", "file_title"])
    assert hits and hits[0]["file_title"] == "it_doc"
    client.drop_collection(milvus_config.chunks_collection + "_it")
```
```python
# tests/conftest.py
import socket, pytest

def _reachable(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False

def pytest_collection_modifyitems(config, items):
    skips = {}
    if not _reachable("localhost", 19530):
        skips[pytest.mark.integration] = "Milvus(19530) 不可达"
    if not _reachable("localhost", 27017):
        skips[pytest.mark.integration] = "MongoDB(27017) 不可达"
    for item in items:
        for mark, reason in skips.items():
            if mark.markname in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
```
- [ ] **Step 2**: `pytest tests/unit -m unit -v` → embedding/milvus 用例 FAIL（模块不存在）
- [ ] **Step 3**: 按笔记 09/10/12/17/18 + 上方 Interfaces 实现 7 个 util（注意三处与笔记的差异：① embedding 走智谱 OpenAI 兼容 `/embeddings`、返回纯 `list[list[float]]`；② milvus 用 `vector_search` 替代 `create_hybrid_search_requests`+`hybrid_search`；③ reranker 用 httpx POST 智谱 `paas/v4/rerank` 端点，非 dashscope SDK/本地 BGE）
- [ ] **Step 4**: `pytest tests/unit -m unit -v` → PASS；`docker/` 下 `docker compose up -d` 后 `pytest tests/integration -m integration -v` → PASS
- [ ] **Step 5**:
```bash
git add -A && git commit -m "feat: utils 工具层——LLM/Embedding(API)/Milvus(纯稠密)/MinIO/Mongo历史/Rerank/JSON"
```

---

### Task 3: 导入流水线（7 节点 + 图）

**Files:**
- Create: `processor/import_processor/nodes/`（7 个节点 + `__init__.py`）、`processor/import_processor/main_graph.py`
- Test: `tests/unit/test_node_entry.py`、`test_node_document_split.py`、`test_node_md_img.py`、`test_node_item_name_recognition.py`、`test_node_embedding.py`、`test_node_import_milvus.py`、`test_import_workflow.py`

**Interfaces (Produces):** 节点类与 `name` 见文件全景图；chunk 字典结构（下游检索依赖，字段名必须一字不差）：
```python
chunk = {"title": str, "content": str, "parent_title": str, "part": int,
         "file_title": str, "item_name": str, "dense_vector": list[float], "chunk_id": str}
```

**实现来源（Spec 指引）与差异表：**

| 节点 | 笔记 | 差异（全 API 适配） |
|------|------|---------------------|
| NodeEntry | 04 | 无。校验路径/后缀(.pdf/.md)/设置 is_*_enabled、file_title=Path.stem |
| NodePDFToMD | 05 | 无（本就是云 API）。`POST /file-urls/batch`→PUT 上传→轮询 `extract-results/batch/{batch_id}`(600s/3s)→下载 zip 解压，full.md 改名 `{stem}.md` |
| NodeMDImg | 06 | 无。VLM 摘要(`lm_config.vl_model`=glm-5.3-flash, base64, 限流 10 次/60s 滑窗)、MinIO 上传(img_dir/文档名)、MD 内 URL 替换、备份 `*_new.md`。注意：若该模型不支持图片输入，摘要失败走笔记的兜底文案"图片描述"，验收时验证 |
| NodeDocumentSplit | 07 | 无。标题初切→RecursiveCharacterTextSplitter 二切→短块合并；备份 chunks.json（路径用 `Path(file_dir)/"chunks.json"`，修掉笔记 str/Path 混用 bug） |
| NodeItemNameRecognition | 08 | `_step_5_generate_vectors`：`generate_embeddings([item_name])` 返回 `list[list[float]]`，取 `[0]` 存 `dense_vector`（无 sparse）；建集合用 `ensure_collection(client, item_name_collection, len(vec))` |
| NodeEmbedding | 09 | 类名 NodeEmbedding；batch_size 取 `embedding_config.embedding_batch_size`(10)；输入文本 `f"{item_name}\n{content}" if item_name else content`；`doc["dense_vector"]=vec`（无 sparse_vector） |
| NodeImportMilvus | 10 | `_step_1` 提取 `vector_dimension=len(chunks[0]["dense_vector"])`；`ensure_collection`；按 file_title 幂等删旧；insert 后回填 `chunk_id=str(id)` |
| KBImportWorkflow | 03 | `StateGraph(ImportGraphState)`；`route_after_entry` 按 is_pdf_read_enabled/is_md_read_enabled 条件路由；`run(state, stream=False)` stream 时 `self.graph.stream(state, stream_mode="values")` |

- [ ] **Step 1: 写失败测试**——核心纯逻辑（文档切片）必须真实覆盖；其余节点以 mock 数据驱动单节点
```python
# tests/unit/test_node_document_split.py（节选，完整用例：标题切分/无标题/超长二切/短块合并/代码块边界）
import pytest
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
pytestmark = pytest.mark.unit

def make_state(md):
    return {"file_title": "测试手册", "md_content": md, "chunks": []}

def test_split_by_titles_basic():
    md = "# 概述\n内容A" + "x" * 100 + "\n# 安装\n内容B"
    node = NodeDocumentSplit()
    state = node.process(make_state(md))
    titles = [c["title"] for c in state["chunks"]]
    assert titles == ["概述", "安装"]
    assert all(c["file_title"] == "测试手册" for c in state["chunks"])

def test_long_section_split_and_short_merge():
    body = "。".join(["句子" * 30] * 12) + "。"          # 远超 2000 字符 → 二切成多块
    md = "# 章节\n" + body + "\n# 短节\n短"              # 短节 <500 → 与前块尝试合并
    node = NodeDocumentSplit()
    state = node.process(make_state(md))
    assert all(len(c["content"]) <= 2000 + 100 for c in state["chunks"])   # 标题前缀容差
    assert state["chunks"][0].get("part") is not None

def test_code_fence_not_split():
    md = "# 配置\n```bash\n# 这不是标题\nexport A=1\n```\n后文"
    node = NodeDocumentSplit()
    state = node.process(make_state(md))
    assert len(state["chunks"]) == 1                     # 围栏内 # 不切分
```
```python
# tests/unit/test_node_entry.py
import pytest
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.exceptions import ValidationError
pytestmark = pytest.mark.unit

def test_pdf_enabled(tmp_path):
    pdf = tmp_path / "H3C ER2100 用户手册.pdf"; pdf.touch()
    state = NodeEntry().process({"import_file_path": str(pdf)})
    assert state["is_pdf_read_enabled"] and state["file_title"].endswith("用户手册")

def test_unsupported_extension(tmp_path):
    f = tmp_path / "a.docx"; f.touch()
    with pytest.raises(Exception):
        NodeEntry().process({"import_file_path": str(f)})
```
- [ ] **Step 2**: `pytest tests/unit -k "node_entry or document_split" -v` → FAIL
- [ ] **Step 3**: 逐节点实现（顺序：entry → document_split → embedding → import_milvus → item_name_recognition → pdf_to_md → md_img → main_graph）。每节点代码以对应笔记为实现底本，套用差异表。每个 API 类节点（pdf_to_md/md_img/item_name_recognition）在类内做依赖注入友好的小函数拆分，测试用 `unittest.mock.patch` 打在 `utils.*` 入口上
- [ ] **Step 4**: `pytest tests/unit -m unit -v` → 全 PASS；新增用例（mock MinerU 上传轮询、mock VLM+MinIO 的图片替换、mock embedding 的回填断言、mock milvus 的 insert/幂等删除断言、`KBImportWorkflow.build_graph` 的路由断言——MD 文件走 node_md_img 直达、其他类型 END）
- [ ] **Step 5**:
```bash
git add -A && git commit -m "feat: 导入流水线 7 节点与 LangGraph 编排(纯稠密向量适配)"
```

---

### Task 4: 检索流水线（7 节点 + 图 + prompts）

**Files:**
- Create: `processor/query_processor/base.py`、`state.py`、`prompt/`（4 文件）、`nodes/`（7 文件 + `__init__.py`）、`main_graph.py`
- Move+Modify: 根目录 `g_node_answer_output.py` → `processor/query_processor/nodes/node_answer_output.py`（修正 import；删除根目录文件）
- Test: `tests/unit/test_node_rrf.py`、`test_node_rerank.py`、`test_node_item_name_confirm.py`、`test_node_answer_output.py`、`test_query_workflow.py`

**Interfaces (Produces):**
```python
# processor/query_processor/state.py
class QueryGraphState(TypedDict, total=False):
    session_id: str; message_id: str; original_query: str
    embedding_chunks: list; hyde_embedding_chunks: list; web_search_docs: list
    rrf_chunks: list; reranked_docs: list; prompt: str; answer: str
    item_names: list; rewritten_query: str; history: list; is_stream: bool
    hyde_doc: str          # 笔记14 运行期键,补进声明

# processor/query_processor/base.py —— 简化基类(与导入侧不同:无config注入,异常不包装,用 tool.logger)
class NodeBase(ABC):
    name: str = "base_node"
    def __call__(self, state):
        logger.info(f"--- {self.name} 开始 ---")
        add_running_task(state.get("session_id"), self.name, state.get("is_stream"))  # 笔记20改造
        result = self.process(state)
        logger.info(f"--- {self.name} 完成 ---")
        return result
    @abstractmethod
    def process(self, state): ...
```
并发节点（三路搜索）在各自 `process` return 前自行 `add_done_task(state.get("session_id"), self.name, state.get("is_stream"))`；串行节点由基类负责不可行（无统一收口），按笔记 20 约定：**全部节点**在 process 尾部自调 add_done_task，基类只调 add_running_task。

**reranked_docs 结构**（node_answer_output 依赖，含全字段）：
```python
doc = {"chunk_id": str|None, "source": "local"|"web", "title": str,
       "url": str|None, "content": str, "score": float}
```

**实现来源与差异表：**

| 节点 | 笔记 | 差异 |
|------|------|------|
| NodeItemNameConfirm | 12 | `_step_5` 向量化后用 `vector_search(client, item_name_collection, vec, limit=5, output_fields=["item_name"])` 逐主体查询（替代 hybrid）；对齐阈值 0.85/>=0.6 取前3/弃用 三分支不变 |
| NodeSearchEmbedding | 13 | `vector_search(..., expr=item_name in [...], limit=10, output_fields=["chunk_id","content","item_name","title","parent_title","part","file_title"])`；返回 `{"embedding_chunks": hits}`；**补**：给每条 hit 补 `source="local", url=None`（rerank 合并需要） |
| NodeSearchEmbeddingHyde | 14 | HYDE_PROMPT 生成 ≤300 字范文；`combined = rewritten_query + " " + hyde_doc` 向量化检索同上；返回 `{"hyde_embedding_chunks":..., "hyde_doc":...}` |
| NodeWebSearchMcp | 15 | 改用智谱 `web_search_prime` MCP：openai-agents `MCPServerStreamableHttp(url=mcp_config.mcp_base_url, headers={"Authorization": Bearer api_key}, timeout=10, cache_tools_list=True, max_retry_attempts=3)`；`call_tool("web_search_prime", {"search_query": query, "content_size": "medium"})`；解析 `result.content[0].text` JSON，按序尝试 `search_result`/`results`/`pages` 键，每条映射 `title`/`link或url`/`content或snippet或summary`；**降级**：connect/call_tool/解析 任何异常（含 429 配额耗尽）→ log warning + 返回 `{}` |
| NodeRrf | 16 | 无差异。`_rrf_merge([(embedding_chunks,1.0),(hyde_embedding_chunks,1.0)], k=60)`；`chunk_scores[id] += weight/(k+rank)`；首见文档保留 `chunk_data.setdefault` |
| NodeRerank | 17 | 打分来源改为智谱 rerank（`utils/reranker_http_utils.rerank_documents`）。合并 local/web → 打分降序 → 断崖截断(ABS=0.5, RATIO=0.25, TOPK 10/3)；**降级**：`rerank_documents` 返回全 0 分时跳过重排按原序截断 top5 |
| NodeAnswerOutput | 20 | 迁移现有代码；import 修正为 `from processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT`、`from processor.query_processor.base import NodeBase` 等；逻辑不动 |
| KBQueryWorkflowV2 | 11/17 | `_route_after_item_name_confirm`：`state.get("answer")` 有值→`["node_answer_output"]`；否则返回 `["node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"]` 并发；三路 `add_edge` 汇入 `node_rrf`；`run()` 内 `print_ascii()` 留 debug 开关 |

**4 个 prompt 文件全文**（HYDE/主体识别按笔记原文；confirm 与 answer 按笔记要求项编写）：
```python
# prompt/item_name_recognition.py（笔记08原文）
ITEM_NAME_SYSTEM_PROMPT = "你是一个专业的商品名称识别模型，请根据提供的信息，识别商品名称。"
ITEM_NAME_USER_PROMPT_TEMPLATE = """请根据以下信息识别出文档所描述设备的完整名称（含品牌与型号）。
文件名：{file_title}
正文切片：
{context}
要求：只返回名称本身，不要任何解释；无法识别时返回空字符串。
示例：苏伯尓5000W大功率电磁炉"""

# prompt/search_embedding_hyde.py（笔记14原文）
HYDE_PROMPT = """请基于以下用户查询生成一个简洁的回答范文。
用户查询: {rewritten_query}
要求：
1. 回答要简洁明了，直接陈述关键信息
2. 假设你是该领域的专家，回答要专业准确
3. 不要使用"假设"、"可能"等不确定词语
4. 保持回答与查询主题高度相关
5. 使用中文回答且不超过300字"""

# prompt/item_name_confirm.py（笔记12要求：JSON输出+指代消解）
ITEM_NAME_EXTRACT_SYSTEM_PROMPT = "你是一个专业的客服助手，负责从用户问题中提取产品名称并改写问题。"
ITEM_NAME_EXTRACT_TEMPLATE = """请分析用户问题，结合历史对话完成两项任务：
1. 提取问题中提到的产品名称列表 item_names（无法提取时返回空数组）
2. 结合历史对话进行指代消解，把问题改写为独立完整的问题 rewritten_query
{history_text}
用户问题：{query}
严格按以下JSON格式输出：{{"item_names": ["..."], "rewritten_query": "..."}}"""

# prompt/answer_prompt.py（笔记20要求：基于参考内容、不编造、图片区块）
ANSWER_PROMPT = """你是企业设备知识库助手。请严格依据【参考内容】回答【用户问题】，要求：
1. 只使用参考内容中的信息，不编造；参考内容不足以回答时明确说明
2. 回答用中文、条理清晰，可直接引用参考内容中的操作步骤
3. 如参考内容包含图片链接且对回答有辅助作用，在回答最后单独输出一个区块：
【图片】
每行一个图片URL（从参考内容的Markdown图片语法中提取）
4. 提问产品：{item_names}

【历史对话】
{history}

【参考内容】
{context}

【用户问题】
{question}"""
```

- [ ] **Step 1: 写失败测试**
```python
# tests/unit/test_node_rrf.py
import pytest
from processor.query_processor.nodes.node_rrf import NodeRrf
pytestmark = pytest.mark.unit

def _doc(i, src): return {"chunk_id": f"c{i}", "content": f"内容{i}", "source": src}

def test_rrf_two_routes_weighted_by_rank():
    state = {"embedding_chunks": [_doc(1, "local"), _doc(2, "local")],
             "hyde_embedding_chunks": [_doc(2, "local"), _doc(3, "local")]}
    out = NodeRrf().process(state)
    ids = [d["chunk_id"] for d in out["rrf_chunks"]]
    assert ids[0] == "c2"                       # 两路都命中 → 分数最高
    assert set(ids) == {"c1", "c2", "c3"}
    assert out["rrf_chunks"][0]["content"] == "内容2"   # 首见文档保留

# tests/unit/test_node_rerank.py（打 rerank_documents 打点）
def test_cliff_cutoff():
    scores = [0.9, 0.85, 0.8, 0.2, 0.15]        # 0.8→0.2 断崖(abs 0.6 ≥ 0.5) → 截到3条
    with patch("processor.query_processor.nodes.node_rerank.rerank_documents",
               return_value=scores): ...
    # 断言 len(reranked_docs) == 3

def test_rerank_degrade_on_zero_scores():
    # rerank_documents 返回全0 → 按原序保留前5，不抛异常
```
- [ ] **Step 2**: FAIL 确认
- [ ] **Step 3**: 实现基类/state/prompts/7 节点/main_graph（按差异表 + 笔记底本）；迁移并修正 `g_node_answer_output.py`（`git mv` 或新建+删除）
- [ ] **Step 4**: `pytest tests/unit -m unit -v` → 全 PASS
- [ ] **Step 5**:
```bash
git add -A && git commit -m "feat: 检索流水线 7 节点、V2 并发路由、提示词与答案输出归位"
```

---

### Task 5: Web 层对齐 + 统一入口

**Files:**
- Modify: `web/api/import_service.py`、`web/api/query_service.py`（仅 import 对齐与启动参数）
- Rewrite: `main.py`
- Test: `tests/e2e/test_health.py`

**main.py 全文：**
```python
"""掌柜智库统一入口: python main.py import|query|all"""
import argparse
import multiprocessing


def _run(service: str):
    if service == "import":
        from web.api.import_service import app
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        from web.api.query_service import app
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8001)


def main():
    parser = argparse.ArgumentParser(description="掌柜智库服务启动器")
    parser.add_argument("service", choices=["import", "query", "all"], nargs="?", default="all")
    args = parser.parse_args()
    targets = ["import", "query"] if args.service == "all" else [args.service]
    procs = [multiprocessing.Process(target=_run, args=(t,)) for t in targets]
    for p in procs: p.start()
    for p in procs: p.join()


if __name__ == "__main__":
    main()
```
- [ ] **Step 1**: 两个 service 的 import 现在全部可解析（`config.minio_config`、`utils.minio_utils`、`KBImportWorkflow`、`KBQueryWorkflow`、`utils.mongo_history_utils`）；`KBImportWorkflow`/`KBQueryWorkflowV2` 以 `KBQueryWorkflow = KBQueryWorkflowV2` 别名导出保持 query_service 不改逻辑。逐个 `python -c "import web.api.import_service"` 验证
- [ ] **Step 2**:
```python
# tests/e2e/test_health.py
import httpx, pytest
pytestmark = pytest.mark.e2e

def test_import_health():
    assert httpx.get("http://127.0.0.1:8000/health" if False else "http://127.0.0.1:8000/docs").status_code == 200
```
  （注：import 服务无 /health，用 /docs；query 服务有 /health——补一个 `test_query_health` 打 `http://127.0.0.1:8001/health` 断言 `{"ok": true}`）
- [ ] **Step 3**: `python main.py all` 后运行 e2e health → PASS；Ctrl+C 干净退出（multiprocessing join）
- [ ] **Step 4**:
```bash
git add -A && git commit -m "feat: Web层导入对齐与统一启动入口 main.py(import|query|all)"
```

---

### Task 6: 集成与 E2E 测试套件

**Files:**
- Test: `tests/integration/test_import_pipeline.py`（真实 MinerU API + 全链路导入一份小 MD/PDF）、`tests/e2e/test_import_api.py`（httpx 上传→轮询 status→completed）、`tests/e2e/test_query_api.py`（POST /query 非流式→answer；流式 EventSource→delta/final）

- [ ] **Step 1**: e2e import 测试用 `httpx` multipart 上传 `tmp_path` 生成的小 MD 文件（不依赖 MinerU），轮询 `/status/{task_id}` 至 completed（超时 120s），断言 done_list 含全部节点中文名
- [ ] **Step 2**: e2e query 测试：先直连 Milvus 灌入一条测试 chunk（fixture），`POST /query {"query":"...","is_stream":false}` → `answer` 非空；流式用 `httpx.stream("GET", "/stream/{sid}")` 断言收到 `event: delta` 与 `event: final`（final 含 image_urls 字段）
- [ ] **Step 3**: integration 全链路：真实上传一份 5 页以内 PDF 走完 MinerU→Milvus（标记 `@pytest.mark.integration`，MinerU token 缺失则 skip）
- [ ] **Step 4**: `pytest -m "unit or integration or e2e"` 全绿；生成覆盖率报告 `pytest --cov=processor --cov=utils --cov-report=term`，核心逻辑（split/rrf/rerank/confirm）行覆盖 ≥80%
- [ ] **Step 5**:
```bash
git add -A && git commit -m "test: 集成与 E2E 测试套件(导入/查询/SSE流式/MinerU全链路)"
```

---

### Task 7: 端到端人工验收 + 文档

**Files:**
- Modify: `docker/README.md` 无需动；Create: `README.md`（快速启动：docker compose up → pip install → python main.py all → 页面地址）
- 验收脚本执行（非测试代码，逐项人工/半自动验证）

- [ ] **Step 1**: `docker compose up -d` 五服务健康；`python main.py all`
- [ ] **Step 2**: 浏览器 `http://127.0.0.1:8000/import.html` 上传 `H3C ER2100企业级路由器 用户手册-6W104-整本手册.pdf` → 轮询进度全部绿色 → Attu(localhost:7000) 中确认 `kb_chunks`/`kb_item_names` 有数据
- [ ] **Step 3**: `http://127.0.0.1:8001/chat.html` 提问「H3C ER2100 的默认管理地址是多少」→ 断言：流式打字输出、答案引用手册内容、`【图片】`/image_urls 有图、MongoDB chat_message 有 user+assistant 记录；再问「它怎么配置 NAT」验证多轮指代消解（rewritten_query 独立完整）
- [ ] **Step 4**: 提一个不存在的产品（「苹果手机怎么拆机」）→ 应得到主体确认失败的话术而不是编造。注意：`web_search_prime` MCP 账号配额至 2026-10-02 前耗尽，网络搜索结果为空属**预期降级**，观察日志出现降级 warning 而非流程失败
- [ ] **Step 5**:
```bash
git add -A && git commit -m "docs: README 快速启动指南,端到端验收记录"
```

---

## Self-Review 记录

1. **Spec 覆盖**：笔记 01(架构)=全局约束；03-10(导入)=Task 3；11-17(检索)=Task 4；18-19(Web/SSE/前端)=Task 5（前端 HTML 已存在，SSE utils 已存在）；20(答案输出+任务追踪)=Task 4 差异表；测试模式(笔记各篇 __main__ 自测)升级为 Task 6 三层。ADR-0001 的纯稠密适配贯穿 Task 2/3/4 差异表。无遗漏。
2. **占位符扫描**：test_node_rerank 中 `patch(...)` 处为示意省略号——执行时按 test_rrf 的完整样式展开（已在 Step 3 指明 mock 打点 `rerank_documents`）；其余无 TBD/TODO。
3. **类型一致性**：`generate_embeddings` 返回 `list[list[float]]`（Task 2 定义）→ Task 3 两个节点消费一致；`vector_search` 返回 `list[dict]` → Task 4 两路搜索消费一致；reranked_docs 字段集与现有 `g_node_answer_output.py` 的 `_format_reranked_docs`/`_extract_images_from_docs` 读取的 `content/source/chunk_id/url/title/score` 完全对齐。
