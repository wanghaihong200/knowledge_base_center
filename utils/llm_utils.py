"""LLM 客户端工具：统一创建带缓存的 ChatOpenAI 实例"""
from typing import Optional

from langchain_openai import ChatOpenAI

from config.lm_config import lm_config

# (model, json_mode) -> ChatOpenAI 实例缓存
_llm_client_cache: dict = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取 LLM 客户端实例（按 (model, json_mode) 缓存复用）

    Args:
        model: 模型名，默认取 lm_config.llm_model
        json_mode: 是否开启 JSON 输出模式（用于结构化提取）
    """
    model = model or lm_config.llm_model
    cache_key = (model, json_mode)
    if cache_key in _llm_client_cache:
        return _llm_client_cache[cache_key]

    kwargs = dict(
        model=model,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        temperature=lm_config.llm_temperature,
        # 注：笔记为 qwen 设计的 extra_body={"enable_thinking": False} 在智谱
        # glm-5.3-flash 上会报错 1210（该模型始终思考，不支持关闭），故不传
    )
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    client = ChatOpenAI(**kwargs)
    _llm_client_cache[cache_key] = client
    return client
