"""MinerU 全链路集成测试：手工构造的最小 PDF 走完 PDF→MD→切片→入库"""
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from tests.conftest import require_milvus


def _minimal_pdf_bytes() -> bytes:
    """手工构造一个最小可解析的单页 PDF（含标题与正文文本）"""
    content = b"""BT /F1 18 Tf 50 750 Td (H3C ER2100 Test Manual) Tj ET
BT /F1 11 Tf 50 720 Td (The default management address is 192.168.1.1) Tj ET
BT /F1 11 Tf 50 700 Td (Configure NAT in Network > NAT settings page) Tj ET"""
    stream = b"q\n" + content + b"\nQ"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return out


@pytest.fixture
def pdf_file(tmp_path) -> Path:
    pdf = tmp_path / "e2e最小手册.pdf"
    pdf.write_bytes(_minimal_pdf_bytes())
    return pdf


def _run_import(pdf_file: Path) -> dict:
    """直接调用 LangGraph 全流程（不经 HTTP），返回最终 state"""
    import uuid

    from processor.import_processor.main_graph import KBImportWorkflow

    task_id = str(uuid.uuid4())
    file_dir = pdf_file.parent / task_id
    file_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "task_id": task_id,
        "file_dir": str(file_dir),
        "import_file_path": str(pdf_file),
    }
    final_state = None
    for event in KBImportWorkflow().run(state, stream=True):
        pass  # 逐节点消费更新事件
    # invoke 拿最终态
    final_state = KBImportWorkflow().run(state)
    return final_state


def test_mineru_full_chain(pdf_file):
    final_state = _run_import(pdf_file)
    assert final_state.get("md_content"), "MinerU 应产出 Markdown 内容"
    assert final_state.get("chunks"), "应有切片产生"
    assert final_state.get("item_name"), "应识别出主体名"
    first = final_state["chunks"][0]
    assert first.get("dense_vector"), "切片应带稠密向量"
    assert first.get("chunk_id"), "切片应回填 chunk_id"
