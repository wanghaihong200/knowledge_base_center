"""导入 API e2e：上传 MD → 轮询任务状态 → 全节点完成（真实 LLM/向量化/Milvus）"""
import time

import httpx
import pytest

pytestmark = pytest.mark.e2e

BASE = "http://127.0.0.1:8000"

# 导入流程应完成的节点中文名（顺序无关）
EXPECTED_NODES = ["检查文件", "文档切分", "主体名称识别", "向量生成", "导入向量库"]


def test_upload_md_and_complete(tmp_path):
    md = tmp_path / "e2e测试手册.md"
    body = "# 概述\n" + "这是 H3C ER2100 路由器的接口测试手册。" * 30
    body += "\n\n# 默认配置\n设备默认管理地址为 192.168.1.1。\n"
    md.write_text(body, encoding="utf-8")

    # 1. 上传
    with open(md, "rb") as f:
        resp = httpx.post(
            f"{BASE}/upload",
            files={"files": ("e2e测试手册.md", f, "text/markdown")},
            timeout=30,
        )
    assert resp.status_code == 200, resp.text
    task_ids = resp.json()["task_ids"]
    assert task_ids, "应返回任务 ID 列表"

    # 2. 轮询至完成（全链路含真实 LLM 识别与向量化，给足超时）
    deadline = time.time() + 180
    status = {}
    with httpx.Client(timeout=10) as client:
        while time.time() < deadline:
            status = client.get(f"{BASE}/status/{task_ids[0]}").json()
            if status.get("status") in ("completed", "failed"):
                break
            time.sleep(2)

    assert status.get("status") == "completed", f"导入未成功: {status}"
    done = "".join(status.get("done_list", []))
    for node_cn in EXPECTED_NODES:
        assert node_cn in done, f"节点「{node_cn}」未出现在完成列表: {done}"
