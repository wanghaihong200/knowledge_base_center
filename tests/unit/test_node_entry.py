"""入口节点测试：路径校验与类型路由"""
import pytest

from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.exceptions import (
    FileProcessingError,
    StateFieldError,
    ValidationError,
)

pytestmark = pytest.mark.unit


def test_pdf_enabled(tmp_path):
    pdf = tmp_path / "H3C ER2100 用户手册.pdf"
    pdf.touch()
    state = NodeEntry().process({"import_file_path": str(pdf)})
    assert state["is_pdf_read_enabled"] is True
    assert state["is_md_read_enabled"] is False
    assert state["pdf_path"] == str(pdf)
    assert state["file_title"].endswith("用户手册")


def test_md_enabled(tmp_path):
    md = tmp_path / "note.md"
    md.touch()
    state = NodeEntry().process({"import_file_path": str(md)})
    assert state["is_md_read_enabled"] is True
    assert state["md_path"] == str(md)


def test_missing_path_raises():
    with pytest.raises(StateFieldError):
        NodeEntry().process({"import_file_path": ""})


def test_not_exist_file_raises(tmp_path):
    with pytest.raises(FileProcessingError):
        NodeEntry().process({"import_file_path": str(tmp_path / "ghost.pdf")})


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "a.docx"
    f.touch()
    with pytest.raises(ValidationError):
        NodeEntry().process({"import_file_path": str(f)})
