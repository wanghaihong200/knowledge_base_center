"""LLM 服务配置（智谱 BigModel，OpenAI 兼容格式）"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM/视觉/主体识别模型配置"""

    base_url: str = os.getenv("OPENAI_API_BASE", "")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_DEFAULT_MODEL", "glm-5.3-flash")
    vl_model: str = os.getenv("VL_MODEL", "glm-5.3-flash")
    item_model: str = os.getenv("ITEM_MODEL", "glm-5.3-flash")
    llm_temperature: float = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))


lm_config = LLMConfig()
