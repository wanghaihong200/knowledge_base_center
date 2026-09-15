"""主体确认提示词——从用户问题中提取主体并做指代消解"""

ITEM_NAME_EXTRACT_SYSTEM_PROMPT = "你是一个专业的客服助手，负责从用户问题中提取产品名称并改写问题。"

ITEM_NAME_EXTRACT_TEMPLATE = """请分析用户问题，结合历史对话完成两项任务：
1. 提取问题中提到的产品名称列表 item_names（无法提取时返回空数组）
2. 结合历史对话进行指代消解，把问题改写为不依赖上下文、独立完整的问题 rewritten_query

历史对话：
{history_text}

用户问题：{query}

严格按以下JSON格式输出：{{"item_names": ["..."], "rewritten_query": "..."}}"""
