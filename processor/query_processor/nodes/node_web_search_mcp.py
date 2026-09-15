"""网络搜索节点：智谱 web_search_prime MCP（失败自动降级跳过）"""
import asyncio
import json

from config.mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from tool.logger import logger
from utils.task_utils import add_done_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能: 联网搜索（MCP）
    流程: 改写问题 → MCP call_tool(web_search_prime) → 解析结果列表
    降级: 连接/调用/解析任何异常（含配额 429）→ 记日志返回空，流程继续
    输出: {"web_search_docs": [{title, url, content}]}
    """

    name: str = "node_web_search_mcp"

    # MCP 响应里结果列表可能出现的键（按序尝试），单条内字段做同义映射
    LIST_KEYS = ("search_result", "results", "pages")
    URL_KEYS = ("url", "link")
    CONTENT_KEYS = ("content", "snippet", "summary", "text")

    def process(self, state: dict) -> dict:
        query = state.get("rewritten_query") or state.get("original_query", "")
        try:
            docs = asyncio.run(self._mcp_call(query))
            self.log_step("网络搜索完成", f"{len(docs)} 条结果")
        except Exception as e:
            logger.warning(f"[{self.name}] 网络搜索降级跳过: {e}")
            docs = []

        add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
        return {"web_search_docs": docs}

    async def _mcp_call(self, query: str) -> list:
        """连接 MCP 服务 → 调用搜索工具 → finally 清理连接"""
        from agents.mcp import MCPServerStreamableHttp

        server = MCPServerStreamableHttp(
            name="web_search_prime",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )
        await server.connect()
        try:
            result = await server.call_tool(
                "web_search_prime",
                {"search_query": query, "content_size": "medium"},
            )
            return self._parse_docs(result)
        finally:
            await server.cleanup()

    def _parse_docs(self, result) -> list:
        """解析 MCP 工具响应：content[0].text 为 JSON，按候选键提取结果列表"""
        try:
            text = result.content[0].text
            data = json.loads(text)
        except Exception as e:
            logger.warning(f"[{self.name}] MCP 响应解析失败: {e}")
            return []

        items = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in self.LIST_KEYS:
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        if not items:
            return []

        docs = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = next(
                (str(item[k]).strip() for k in self.URL_KEYS if item.get(k)), ""
            )
            content = next(
                (str(item[k]).strip() for k in self.CONTENT_KEYS if item.get(k)), ""
            )
            if content:
                docs.append({"title": title, "url": url or None, "content": content})
        return docs
