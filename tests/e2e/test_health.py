"""健康检查 e2e：需要服务已启动（python main.py all 或单服务）"""
import httpx
import pytest

pytestmark = pytest.mark.e2e


def test_query_service_health():
    resp = httpx.get("http://127.0.0.1:8001/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_import_service_docs():
    resp = httpx.get("http://127.0.0.1:8000/docs", timeout=5)
    assert resp.status_code == 200
