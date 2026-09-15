"""MinerU 云端 PDF 解析服务配置"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MineruConfig:
    """MinerU API 配置"""

    base_url: str = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4")
    api_token: str = os.getenv("MINERU_API_TOKEN", "")


mineru_config = MineruConfig()
