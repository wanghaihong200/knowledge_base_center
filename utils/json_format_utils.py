"""JSON 序列化工具：ObjectId / datetime 等类型容错"""
import json
from datetime import date, datetime

from bson import ObjectId


class CustomJSONEncoder(json.JSONEncoder):
    """支持 ObjectId 与日期类型的 JSON 编码器"""

    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)


def serialize_json(obj) -> str:
    """紧凑序列化（日志/单行输出）"""
    return json.dumps(obj, ensure_ascii=False, cls=CustomJSONEncoder)


def format_json(obj) -> str:
    """缩进序列化（调试打印）"""
    return json.dumps(obj, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
