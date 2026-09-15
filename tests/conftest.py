"""测试公共配置：外部服务可达性探测（不可达时自动跳过对应集成测试）"""
import socket

import pytest


def _reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


require_milvus = pytest.mark.skipif(
    not _reachable("localhost", 19530), reason="Milvus(19530) 不可达"
)
require_mongo = pytest.mark.skipif(
    not _reachable("localhost", 27017), reason="MongoDB(27017) 不可达"
)
require_minio = pytest.mark.skipif(
    not _reachable("localhost", 9000), reason="MinIO(9000) 不可达"
)
