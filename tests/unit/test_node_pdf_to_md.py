"""PDF转MD节点测试：mock MinerU 云 API（batch 上传/轮询/下载解压）"""
import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.exceptions import PdfConversionError

pytestmark = pytest.mark.unit

_NS = "processor.import_processor.nodes.node_pdf_to_md"


def _zip_bytes(filename="full.md", content="# 标题\n正文内容") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(filename, content)
    return buf.getvalue()


@pytest.fixture
def mineru_env(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    file_dir = tmp_path / "out"
    file_dir.mkdir()
    return str(pdf), str(file_dir)


def test_pdf_to_md_happy_path(mineru_env):
    pdf_path, file_dir = mineru_env
    post_resp = MagicMock()
    post_resp.json.return_value = {"data": {"file_urls": ["http://signed-upload"], "batch_id": "b1"}}
    poll_resp = MagicMock()
    poll_resp.json.return_value = {"data": {"extract_result": [
        {"state": "done", "full_zip_url": "http://zip-download"}]}}
    zip_resp = MagicMock()
    zip_resp.content = _zip_bytes()

    with patch(f"{_NS}.requests") as req:
        req.post.return_value = post_resp          # 申请上传链接
        req.put.return_value = MagicMock(status_code=200)  # PUT 上传
        req.get.side_effect = [poll_resp, zip_resp]  # 先轮询，后下载 zip
        state = NodePDFToMD().process({"pdf_path": pdf_path, "file_dir": file_dir})

    assert state["md_content"] == "# 标题\n正文内容"
    assert state["md_path"].endswith("doc.md")
    assert (state["md_path"] and __import__("pathlib").Path(state["md_path"]).exists())


def test_pdf_to_md_failed_state_raises(mineru_env):
    pdf_path, file_dir = mineru_env
    post_resp = MagicMock()
    post_resp.json.return_value = {"data": {"file_urls": ["u"], "batch_id": "b1"}}
    poll_resp = MagicMock()
    poll_resp.json.return_value = {"data": {"extract_result": [
        {"state": "failed", "err_msg": "解析失败"}]}}
    with patch(f"{_NS}.requests") as req:
        req.post.return_value = post_resp
        req.put.return_value = MagicMock(status_code=200)
        req.get.return_value = poll_resp
        with pytest.raises(PdfConversionError):
            NodePDFToMD().process({"pdf_path": pdf_path, "file_dir": file_dir})


def test_pdf_missing_raises(mineru_env):
    _, file_dir = mineru_env
    with pytest.raises(Exception):
        NodePDFToMD().process({"pdf_path": "", "file_dir": file_dir})
