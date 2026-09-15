"""MongoDB 会话历史工具：多轮对话消息的持久化读写"""
import os
import time
from typing import List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient

load_dotenv()


class HistoryMongoTool:
    """chat_message 集合的读写封装"""

    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        db = self.client[os.getenv("MONGO_DB_NAME", "kb001")]
        self.collection = db["chat_message"]
        # 会话内按时间顺序检索
        self.collection.create_index([("session_id", 1), ("ts", -1)])

    def clear_history(self, session_id: str) -> int:
        """清空指定会话，返回删除条数"""
        return self.collection.delete_many({"session_id": session_id}).deleted_count

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: Optional[List[str]] = None,
        image_urls: Optional[List[str]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """
        保存一条消息；传入 message_id 时改为覆盖更新（用于产品确认后回填）

        Returns:
            消息 ID（str(ObjectId)）
        """
        document = {
            "session_id": session_id,
            "role": role,
            "text": text,
            "rewritten_query": rewritten_query or "",
            "item_names": item_names or [],
            "image_urls": image_urls or [],
            "ts": int(time.time()),
        }
        if message_id:
            self.collection.update_one({"_id": ObjectId(message_id)}, {"$set": document})
            return message_id
        return str(self.collection.insert_one(document).inserted_id)

    def update_message_item_names(self, ids: List[str], item_names: List[str]) -> None:
        """批量回填消息的主体名称（产品确认后回溯历史消息）"""
        object_ids = [ObjectId(i) for i in ids if i]
        if not object_ids:
            return
        self.collection.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": item_names}},
        )

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[dict]:
        """按时间升序获取会话最近 limit 条消息（_id 兜底排序，保证同秒消息顺序稳定）"""
        cursor = (
            self.collection.find({"session_id": session_id})
            .sort([("ts", ASCENDING), ("_id", ASCENDING)])
            .limit(limit)
        )
        return list(cursor)


_history_mongo_tool: Optional[HistoryMongoTool] = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取历史工具单例（惰性创建，避免 import 时连接数据库）"""
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


def clear_history(session_id: str) -> int:
    return get_history_mongo_tool().clear_history(session_id)


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    message_id: Optional[str] = None,
) -> str:
    return get_history_mongo_tool().save_chat_message(
        session_id, role, text, rewritten_query, item_names, image_urls, message_id
    )


def update_message_item_names(ids: List[str], item_names: List[str]) -> None:
    return get_history_mongo_tool().update_message_item_names(ids, item_names)


def get_recent_messages(session_id: str, limit: int = 10) -> List[dict]:
    return get_history_mongo_tool().get_recent_messages(session_id, limit)
