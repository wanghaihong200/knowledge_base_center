"""主体确认节点：提取主体 → 库内对齐 → 确认/反问/拒绝 三分支"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import (
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT,
    ITEM_NAME_EXTRACT_TEMPLATE,
)
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)
from utils.milvus_utils import get_milvus_client, vector_search
from utils.task_utils import add_done_task


class NodeItemNameConfirm(NodeBase):
    """
    节点功能: 主体（产品）确认
    流程: 校验参数 → 取历史 → 存 user 消息 → LLM 提取主体并改写问题
          → 主体向量化查库对齐 → 确认/候选反问/未找到 三分支 → 写历史与回溯
    """

    name: str = "node_item_name_confirm"

    HIGH_SCORE = 0.85        # 高于该相似度直接确认
    MID_SCORE = 0.60         # 低于高分时，达到该值的取前 3 作候选
    MAX_CANDIDATES = 3
    HISTORY_LIMIT = 10       # 参与提取的历史消息条数

    def process(self, state: dict) -> dict:
        # 阶段一：校验参数
        session_id = state.get("session_id")
        original_query = state.get("original_query")
        if not session_id or not original_query:
            raise ValueError("session_id 和 original_query 不能为空")

        # 阶段二：读取历史对话
        state["history"] = get_recent_messages(session_id, limit=self.HISTORY_LIMIT)

        # 阶段三：保存本轮 user 消息，拿到 message_id
        state["message_id"] = save_chat_message(session_id, "user", original_query)

        # 阶段四：LLM 提取主体 + 指代消解改写
        extracted = self._step_4_extract_info(state)
        state["rewritten_query"] = extracted["rewritten_query"]
        state["item_names"] = extracted["item_names"]
        self.log_step("提取结果", f"item_names={extracted['item_names']}, "
                                 f"rewritten={extracted['rewritten_query'][:50]}")

        # 阶段五：主体向量化并查询主体集合
        matches = self._step_5_vectorize_and_query(extracted["item_names"])

        # 阶段六：对齐（确认 / 候选）
        confirmed, options = self._step_6_align_item_names(matches)

        # 阶段七：三分支——确认 / 反问 / 拒绝
        answer = self._step_7_check_confirmation(state, confirmed, options)

        # 阶段八：写历史（assistant 回答 + user 消息回填主体）
        self._step_8_write_history(state, answer)

        add_done_task(session_id, self.name, bool(state.get("is_stream")))
        return state

    # ==================== 阶段方法 ====================

    def _step_4_extract_info(self, state: dict) -> dict:
        """LLM JSON 模式提取；清理 ```json 包裹；任何异常兜底为空主体 + 原问题"""
        query = state["original_query"]
        history_text = "\n".join(
            f"{m.get('role', '')}: {m.get('text', '')}" for m in (state.get("history") or [])
        )
        try:
            llm = get_llm_client(model=lm_config.item_model, json_mode=True)
            response = llm.invoke([
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=ITEM_NAME_EXTRACT_TEMPLATE.format(
                    history_text=history_text, query=query)),
            ])
            content = (response.content or "").strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            data = json.loads(content)
            item_names = [str(n).strip() for n in data.get("item_names", []) if str(n).strip()]
            rewritten = str(data.get("rewritten_query", "")).strip() or query
            return {"item_names": item_names, "rewritten_query": rewritten}
        except Exception as e:
            self.log_step("主体提取失败，兜底原始问题", str(e))
            return {"item_names": [], "rewritten_query": query}

    def _step_5_vectorize_and_query(self, item_names: list) -> list:
        """逐个主体向量化，查主体集合，返回 [{extracted_name, matches: [{item_name, score}]}]"""
        results = []
        if not item_names:
            return results
        client = get_milvus_client()
        vectors = generate_embeddings(item_names)
        for name, vector in zip(item_names, vectors):
            hits = vector_search(
                client,
                milvus_config.item_name_collection,
                vector,
                limit=5,
                output_fields=["item_name"],
            )
            matches = [
                {"item_name": h.get("entity", {}).get("item_name", ""),
                 "score": float(h.get("distance") or 0.0)}
                for h in hits
            ]
            results.append({"extracted_name": name, "matches": matches})
        return results

    def _step_6_align_item_names(self, matches: list):
        """对齐规则：>0.85 全部确认；无高分则 >=0.6 的前 3 作候选；全低丢弃"""
        confirmed = set()
        options = set()
        for entry in matches:
            high = [m["item_name"] for m in entry["matches"] if m["score"] > self.HIGH_SCORE]
            if high:
                confirmed.update(high)
                continue
            mid = [m["item_name"] for m in entry["matches"]
                   if m["score"] >= self.MID_SCORE][: self.MAX_CANDIDATES]
            options.update(mid)
        return list(confirmed), list(options)

    def _step_7_check_confirmation(self, state: dict, confirmed: list, options: list) -> str:
        """确认 → 回溯历史并放行；候选 → 反问；无 → 婉拒。返回 answer（空串表示放行检索）"""
        session_id = state["session_id"]
        if confirmed:
            # 回溯历史中尚无主体的消息，统一回填
            ids_to_update = [
                m["_id"] for m in (state.get("history") or [])
                if not m.get("item_names")
            ]
            update_message_item_names([str(i) for i in ids_to_update], confirmed)
            state["item_names"] = list(confirmed)
            state["answer"] = ""
            return ""
        if options:
            options_str = "、".join(sorted(options))
            state["item_names"] = []
            state["answer"] = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            return state["answer"]
        state["item_names"] = []
        state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        return state["answer"]

    def _step_8_write_history(self, state: dict, answer: str) -> None:
        """有反问/拒绝答案时存 assistant 消息；回填本轮 user 消息的主体"""
        session_id = state["session_id"]
        try:
            if answer:
                save_chat_message(session_id, "assistant", answer)
            if state.get("item_names"):
                update_message_item_names([state["message_id"]], state["item_names"])
        except Exception as e:
            self.log_step("历史写入失败（不影响主流程）", str(e))
