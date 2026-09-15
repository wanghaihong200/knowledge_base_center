# 掌柜智库（knowledge_base_center）

面向垂直领域设备手册/技术文档的企业级 RAG 智能知识库。上传 PDF/Markdown 手册，自动完成解析、切片、向量化入库；通过对话页面检索提问，返回带图片引用的答案。

术语表见 [CONTEXT.md](CONTEXT.md)；架构决策见 [docs/adr/0001](docs/adr/0001-api-embedding-over-local-bge.md)；实施计划见 [docs/superpowers/plans](docs/superpowers/plans/)。

## 架构

```
导入流水线 (:8000)                        检索流水线 (:8001)
upload → node_entry ─┬─ PDF → node_pdf_to_md   node_item_name_confirm（主体确认/反问/拒绝）
                     │  (MinerU云API)            ├─ answer 直通 node_answer_output
                     │   node_md_img             └─ 三路并发:
                     │  (VLM+MinIO)                  ├ node_search_embedding（向量+主体过滤）
                     ↓                               ├ node_search_embedding_hyde（HyDE）
              node_document_split                    └ node_web_search_mcp（失败自动降级）
                     ↓                           ↓
        node_item_name_recognition          node_rrf → node_rerank（断崖截断）
                     ↓                           ↓
              node_embedding                node_answer_output（SSE 流式）
                     ↓
              node_import_milvus
```

## 技术栈

| 环节 | 方案 |
|------|------|
| LLM / 视觉 / 主体识别 | 智谱 `glm-5.3-flash`（coding 端点） |
| Embedding | 智谱 `embedding-3`（1024 维纯稠密，标准 paas/v4 端点） |
| Rerank | 智谱 `rerank`（`paas/v4/rerank`，失败降级保留原序） |
| 网络搜索 | 智谱 `web_search_prime` MCP（失败自动降级跳过） |
| PDF 解析 | MinerU 云 API（结果下载对 Python TLS 受限时自动回退 curl） |
| 向量库 / 历史 / 对象存储 | Milvus 2.5（Attu :7000）/ MongoDB / MinIO（:9000，控制台 :9003） |

## 快速启动

```bash
# 1. 启动依赖服务（Milvus/Attu/MinIO/etcd/Mongo）
cd docker && docker compose up -d && cd ..
# 注意：除 mongo 外的服务无 restart 策略，Docker Desktop 重启后需重新 up

# 2. 安装依赖
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

# 3. 配置 .env（参考 .env.example，填入智谱 API Key 与 MinerU Token）

# 4. 启动服务（import=8000 / query=8001 / all）
python main.py all

# 5. 使用
#    导入页: http://127.0.0.1:8000/import.html  上传 PDF/MD
#    问答页: http://127.0.0.1:8001/chat.html    对话提问
#    Milvus 管理台: http://localhost:7000（地址填 standalone:19530）
```

## 测试

```bash
python -m pytest tests/unit -m unit                  # 单元测试（mock 外部依赖）
python -m pytest tests/integration -m integration    # 集成测试（需 docker 服务）
python -m pytest tests/e2e -m e2e                    # 端到端（需服务已启动: python main.py all）
python -m pytest                                     # 全部
```

分层说明：单测覆盖切片/RRF/断崖截断/主体对齐等核心逻辑与全部降级路径；集成测试验证 Milvus 往返、Mongo 历史与 MinerU 全链路；e2e 通过真实 API 贯通导入→检索（含 SSE 流式与图片提取）。

## 目录结构

```
config/                  服务级配置（dataclass 单例，每服务一文件）
processor/import_processor/   导入流程：BaseNode/state/exceptions/config + nodes/ + main_graph
processor/query_processor/    检索流程：NodeBase/state/prompt/ + nodes/ + main_graph
utils/                   LLM/Embedding/Milvus/MinIO/Mongo/Rerank/SSE/任务追踪 工具
web/api/                 FastAPI 服务（导入 8000 / 查询 8001）
web/page/                前端页面（import.html / chat.html）
tests/                   unit / integration / e2e 三层测试
doc/origin_doc/          课程设计笔记（gitignore，设计底本）
```
