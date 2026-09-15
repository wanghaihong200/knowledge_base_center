"""智谱 web_search_prime MCP 网络搜索服务配置"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class McpConfig:
    """MCP 服务配置（鉴权复用智谱 API key）"""

    mcp_base_url: str = os.getenv(
        "MCP_ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/mcp/web_search_prime/mcp"
    )
    api_key: str = os.getenv("OPENAI_API_KEY", "")


mcp_config = McpConfig()
