"""配置层单元测试：验证各服务配置单例从 .env 正确加载"""
import pytest

from config.lm_config import lm_config
from config.embedding_config import embedding_config
from config.milvus_config import milvus_config
from config.minio_config import minio_config
from config.mineru_config import mineru_config
from config.mcp_config import mcp_config
from config.reranker_config import reranker_config
from processor.import_processor.config import get_config

pytestmark = pytest.mark.unit


def test_lm_config_from_env():
    assert lm_config.api_key, "OPENAI_API_KEY 未配置"
    assert lm_config.base_url, "OPENAI_API_BASE 未配置"
    assert lm_config.llm_model == "glm-5.3-flash"
    assert lm_config.llm_temperature == 0.1


def test_embedding_config_dense_only():
    assert embedding_config.api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert embedding_config.embedding_model == "embedding-3"
    assert embedding_config.embedding_dim == 1024
    assert embedding_config.embedding_batch_size == 8


def test_reranker_config_zhipu():
    assert reranker_config.api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert reranker_config.text_rerank_model == "rerank"


def test_mineru_config():
    assert mineru_config.api_token, "MINERU_API_TOKEN 未配置"
    assert mineru_config.base_url == "https://mineru.net/api/v4"


def test_mcp_config():
    assert mcp_config.mcp_base_url == "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp"
    assert mcp_config.api_key == lm_config.api_key  # 复用同一智谱 key


def test_minio_config():
    assert minio_config.endpoint == "localhost:9000"
    assert minio_config.bucket_name == "knowledge-base"
    assert minio_config.img_dir == "upload-images"


def test_milvus_config():
    assert milvus_config.milvus_url == "http://localhost:19530"
    assert milvus_config.chunks_collection == "kb_chunks"
    assert milvus_config.item_name_collection == "kb_item_names"


def test_import_config_singleton():
    assert get_config() is get_config()
    assert get_config().max_content_length == 2000
    assert get_config().embedding_dim == 1024
    assert get_config().embedding_batch_size == 8
