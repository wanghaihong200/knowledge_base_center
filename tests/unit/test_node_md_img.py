"""MD图片处理节点测试：mock VLM 与 MinIO，验证图片上传、URL 替换与备份"""
from unittest.mock import MagicMock, patch

import pytest

from processor.import_processor.nodes.node_md_img import NodeMDImg

pytestmark = pytest.mark.unit

_NS = "processor.import_processor.nodes.node_md_img"


@pytest.fixture
def md_env(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "pic1.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    md = tmp_path / "doc.md"
    content = "# 标题\n看图 ![原图](images/pic1.jpg) 说明文字"
    md.write_text(content, encoding="utf-8")
    return str(md), content


@pytest.fixture
def mocks():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="这是路由器面板图")
    fake_minio = MagicMock()
    fake_minio.list_objects.return_value = []
    with patch(f"{_NS}.get_llm_client", return_value=fake_llm), \
         patch(f"{_NS}.get_minio_client", return_value=fake_minio):
        yield fake_llm, fake_minio


def test_upload_replace_and_backup(md_env, mocks):
    md_path, content = md_env
    fake_llm, fake_minio = mocks
    state = NodeMDImg().process({"md_path": md_path, "md_content": content})

    # MinIO 上传：目录 = upload-images/doc，对象名含图片名
    object_name = fake_minio.fput_object.call_args.kwargs["object_name"]
    assert object_name == "upload-images/doc/pic1.jpg"
    # MD 内容替换：alt 换为 VLM 摘要，链接换为 MinIO URL
    assert "![这是路由器面板图](http://localhost:9000/knowledge-base/upload-images/doc/pic1.jpg)" \
        in state["md_content"]
    assert "images/pic1.jpg" not in state["md_content"]
    # 备份文件 *_new.md 已写盘
    assert state["md_path"].endswith("doc_new.md")


def test_vlm_failure_uses_fallback_summary(md_env, mocks):
    md_path, content = md_env
    fake_llm, fake_minio = mocks
    fake_llm.invoke.side_effect = RuntimeError("VLM 挂了")
    state = NodeMDImg().process({"md_path": md_path, "md_content": content})
    assert "![图片描述](http://localhost:9000" in state["md_content"]
