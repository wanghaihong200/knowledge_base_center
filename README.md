# 测试知识库（knowledge_base_center）

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
                                                 ├─ api文档获取，直接从MinIO读取(未实现)
                     ↓                           ↓
        node_item_name_recognition          node_rrf → node_rerank（断崖截断）
                     ↓                           ↓
              node_embedding                node_answer_output（SSE 流式）
                     ↓
              node_import_milvus
```

## 导入流水线设计与实现

### 设计

**职责**：把一份非结构化文档（PDF/Markdown）加工成"可检索的知识单元"——解析出文本与图片、按语义边界切片、识别文档所属主体、生成向量并写入向量库。一次导入 = 一个后台任务（task_id），前端轮询节点级进度。

**编排**（LangGraph `StateGraph`，状态字典 `ImportGraphState` 在节点间传递，节点返回增量键由图自动合并）：

```
START → node_entry ──┬─ .pdf → node_pdf_to_md → node_md_img ─┐
                     ├─ .md  ────────────→ node_md_img ──────┤
                     └─ 其他 → END（拒绝）                    ↓
                        node_document_split → node_item_name_recognition
                        → node_embedding → node_import_milvus → END
```

### 节点实现流程

| 节点 | 实现流程与关键技术点 |
|------|---------------------|
| `node_entry` | `Path` 校验存在性 → 按后缀路由（`.pdf`/`.md`）→ 产出 `file_title`（`Path.stem`）与分支开关。异常分级：字段缺失 `StateFieldError`、文件不存在 `FileProcessingError`、类型不支持 `ValidationError` |
| `node_pdf_to_md` | MinerU 云 API 四步：`POST /file-urls/batch` 申请签名上传链接 → `PUT` 上传二进制 → 轮询 `GET /extract-results/batch/{batch_id}`（600s 超时 / 3s 间隔，`done` 取 `full_zip_url`，`failed` 抛 `PdfConversionError`）→ 下载 zip 解压。**两个工程细节**：① `full.md` 移出解压目录的同时必须同步迁移 `images/`（下游图片节点依赖"md 同级 images/"布局）；② 结果 CDN 对 Python OpenSSL 的 TLS 握手有干扰，requests 失败自动回退系统 `curl` |
| `node_md_img` | 扫描 md 同级 `images/` 下、且在正文中被 `![...](...)` 引用的图片 → 取前后各 100 字符上下文 → **滑动窗口限流**（60s 内 ≤10 次）调用 VLM 生成 ≤15 字中文标题（输出净化：取首行/去格式残留/限长 30，多行 alt 会破坏 Markdown 图片语法）→ 图片上传 MinIO（先按文档目录幂等清理旧对象）→ 正文替换为 `![标题](MinIO URL)` → 备份 `*_new.md` |
| `node_document_split` | **三层切片策略**：① 按标题正则 `^#{1,6}\s+` 初切（带代码围栏状态机：```/~~~ 开合配对，围栏内 # 不切）→ ② 超长章节（>2000 字符，预留标题长度）用 `RecursiveCharacterTextSplitter` 二次切分（中文标点分隔符优先）→ ③ 同章节内 <500 字符的尾块向前合并（`part` 取最新）。产物字段：`title/parent_title/part/file_title`，并备份 `chunks.json` |
| `node_item_name_recognition` | 取前 3 个切片（≤2500 字符）构建上下文 → LLM 识别主体名（只输出名称本身；超长 >50 字视为无效输出回退文件标题）→ 回填每个切片 → 主体名向量化写入 `kb_item_names` 集合（同名先删，幂等） |
| `node_embedding` | 输入文本 = `主体名\n内容`（无主体则裸内容），按批量上限 8 分批调智谱 `embedding-3`，回填 `dense_vector`（1024 维） |
| `node_import_milvus` | 首块向量长度决定集合维度 → `kb_chunks` 集合不存在则按 schema 创建（`chunk_id` 自增主键/content/title/parent_title/part/file_title/item_name/dense_vector，AUTOINDEX+COSINE）→ 按 `file_title` **幂等删除旧数据**（重复导入覆盖）→ insert 后回填 `chunk_id` |

**基类约定**（模板方法模式）：`BaseNode.__call__` 统一做开始/完成日志、任务进度上报（`add_running_task`/`add_done_task`）与异常包装（`ImportProcessError` 携带节点名与根因）；子类只实现 `process(state)`，内部阶段方法命名 `_step_N_xxx`。

## 检索流水线设计与实现

### 设计

**职责**：把一句用户的自然语言提问，变成一句有据可查、可带图的答案。核心思想是**先确认主体、再多路召回、最后收敛**——宁可反问，不猜不编。

**编排**（`KBQueryWorkflowV2`，条件路由返回**节点列表**实现 LangGraph 并发分支，三路各写独立状态键避免写冲突）：

```
START → node_item_name_confirm ─┬─ answer 已有（反问/拒绝）→ node_answer_output → END
                                └─ 三路并发: node_search_embedding（向量+主体过滤）
                                          node_search_embedding_hyde（HyDE 假设性文档）
                                          node_web_search_mcp（联网，失败自动降级）
                                    ↓
                        node_rrf → node_rerank（断崖截断）→ node_answer_output → END
```

### 节点实现流程

| 节点 | 实现流程与关键技术点 |
|------|---------------------|
| `node_item_name_confirm` | 8 步：校验参数 → 读 MongoDB 最近 10 条历史 → 存本轮 user 消息 → **LLM JSON 模式**提取主体并做指代消解（清理 ```` ```json ```` 包裹，失败兜底原问题）→ 主体向量化查 `kb_item_names` → **对齐三分支**：相似度 >0.85 确认为库内规范名并回溯回填历史消息；0.6~0.85 取前 3 作候选反问"您是想问哪个产品"；全低则拒绝。空答案 = 放行检索 |
| `node_search_embedding` | 改写后问题向量化 → 拼接 `item_name in [...]` 过滤表达式（无主体则全库）→ 纯稠密检索 limit=10 → 每条补 `source=local/url=None` 供后续合并 |
| `node_search_embedding_hyde` | LLM 按 HYDE_PROMPT 生成 ≤300 字"专家回答范文" → `原问题 + 范文` 拼接向量化再检索——用答案的形态去匹配文档，弥合问句与正文的表述差异。范文生成失败退化为普通检索 |
| `node_web_search_mcp` | openai-agents `MCPServerStreamableHttp` 连智谱 `web_search_prime` MCP（Bearer 鉴权，10s 超时，3 次重试）→ `call_tool("web_search_prime", {"search_query": ...})` → 兼容解析多种响应键（`search_result`/`results`/`pages`）。**任何异常（含配额 429）→ 记日志返回空，流程继续** |
| `node_rrf` | 只融合两路向量结果（网络结果在 rerank 才并入）。公式 `score(d) = Σ weight/(k + rank)`，k=60；两路命中同一 chunk 分数叠加自然置顶；保留首见文档数据 |
| `node_rerank` | 合并本地切片（`rrf_chunks`）与网络结果（`web_search_docs`）为统一结构 → 智谱 rerank API 打分（响应键 `results`，兼容 `data`；**全 0 分视为 API 不可用，降级保留原序**）→ **断崖截断**：相邻分数绝对差 ≥0.5 或相对差 ≥0.25 处切掉低相关尾部（保留下限 3、上限 10） |
| `node_answer_output` | 已有 answer（反问/拒绝）直通；否则构建提示词（参考内容带字符预算 12000，元数据标签 `[source][chunk_id][score]` 便于引用溯源）→ LLM 生成（流式逐 delta 推 SSE）→ **图片双保险**：前端展示的 `image_urls` 由代码从参考内容正则提取，不信任 LLM 输出 → 写 MongoDB 历史 → 推 `final` 事件 |

### 支撑机制

- **SSE 协议**：`ready`（握手）→ `progress`（节点进度）→ `delta`（答案增量）→ `final`（完整答案+image_urls）/ `error`。队列按 session_id 隔离，后台任务生产、流式响应消费。
- **会话历史**：MongoDB `chat_message` 集合（session_id+ts 复合索引，排序加 `_id` 兜底保证同秒消息稳定），支持产品确认后按 message_id 回填与批量回溯。
- **任务追踪**：内存态 done/running/status，导入侧按 task_id、检索侧按 session_id，节点中文名映射供前端展示。

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
doc/origin_doc/          项目设计笔记（gitignore，设计底本）
```
