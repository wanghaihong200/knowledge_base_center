# 全 API 模型路线替代本地 BGE-M3

课程笔记的检索方案依赖本地 BGE-M3（dense 1024 维 + sparse 双向量混合检索）与本地部署环境；本机无 torch 环境、模型权重未下载，故决定全链路采用云 API。LLM/VL/主体识别早已是智谱 `glm-5.3-flash`（.env 实配，coding 端点）；2026-09-15 用户指定 embedding 与 rerank 也走智谱 BigModel 标准 `paas/v4` 端点：`embedding-3`（1024 维，纯稠密）与 `rerank`，API key 复用 `OPENAI_API_KEY`。

Milvus `kb_chunks`/`kb_item_names` schema 不设 `sparse_vector` 字段，笔记的混合检索（`WeightedRanker` dense+sparse）退化为单路稠密检索 + 主体过滤；多路召回（向量检索 + HyDE）与 RRF 融合架构保留不变。

**Considered Options**:
- 本地 BGE-M3（笔记原方案）——需安装 torch cu128 + FlagEmbedding 并下载 2-3GB 模型，环境成本高，否决
- DashScope text-embedding-v4 + qwen3-rerank（本 ADR 初版方案）——后因项目密钥实际为智谱、且 text-embedding-v4 的 1536 维与 embedding-3 可选维度不重叠而作废
- Milvus 2.5 内置 BM25 全文检索补稀疏路——可完整保留混合检索，但中文分词调优复杂，留作日后恢复稀疏路的候选方案
- 智谱 embedding-3 纯稠密（采纳）——相关性由主体过滤 + rerank 兜底，实现最简

**Consequences**:
- 更换 embedding 模型需全库重嵌入并重建 Milvus 集合（维度 1024）
- .env 中 `BGE_*`、`MODELSCOPE_*` 本地模型配置弃用；不引入 dashscope SDK
- 网络搜索节点走智谱 `web_search_prime` MCP（`MCP_ZHIPU_BASE_URL`），与笔记的百炼 MCP 不同源；账号配额耗尽期间（至 2026-10-02）该节点按降级路径跳过，属预期行为
- LLM 走智谱 coding 端点、Embedding/Rerank 走标准 paas/v4 端点，两者 base_url 不同，配置必须分开
