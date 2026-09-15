"""文档切片节点测试：标题切分/超长二切/短块合并/代码块边界/无标题兜底"""
import pytest

from processor.import_processor.nodes.node_document_split import NodeDocumentSplit

pytestmark = pytest.mark.unit


def make_state(md: str, file_title: str = "测试手册") -> dict:
    return {"file_title": file_title, "md_content": md, "chunks": []}


def test_split_by_titles_basic():
    md = "# 概述\n内容A" + "x" * 100 + "\n# 安装\n内容B"
    state = NodeDocumentSplit().process(make_state(md))
    titles = [c["title"] for c in state["chunks"]]
    assert titles == ["概述", "安装"]
    assert all(c["file_title"] == "测试手册" for c in state["chunks"])
    assert all(c["parent_title"] == c["title"] for c in state["chunks"])


def test_long_section_split_produces_parts():
    body = "。".join(["句子" * 90] * 15) + "。"  # ≈2715 字符，超过 max_content_length=2000
    md = "# 章节\n" + body
    state = NodeDocumentSplit().process(make_state(md))
    assert len(state["chunks"]) >= 2, "超长章节应被二次切分"
    for c in state["chunks"]:
        assert len(c["content"]) <= 2000
        assert c["parent_title"] == "章节"
        assert c["part"] == state["chunks"].index(c)  # part 递增


def test_merge_short_sections_same_parent():
    node = NodeDocumentSplit()
    chunks = [
        {"title": "章节-0", "content": "x" * 600, "parent_title": "章节",
         "part": 0, "file_title": "f"},
        {"title": "章节-1", "content": "短尾", "parent_title": "章节",
         "part": 1, "file_title": "f"},
    ]
    merged = node._merge_short_sections(chunks)
    assert len(merged) == 1
    assert merged[0]["part"] == 1
    assert "短尾" in merged[0]["content"]


def test_no_merge_across_sections():
    node = NodeDocumentSplit()
    chunks = [
        {"title": "甲", "content": "x" * 600, "parent_title": "甲", "part": 0, "file_title": "f"},
        {"title": "乙", "content": "短", "parent_title": "乙", "part": 0, "file_title": "f"},
    ]
    merged = node._merge_short_sections(chunks)
    assert len(merged) == 2, "不同章节即使内容很短也不合并"


def test_code_fence_not_split():
    md = "# 配置\n```bash\n# 这不是标题\nexport A=1\n```\n后文"
    state = NodeDocumentSplit().process(make_state(md))
    assert len(state["chunks"]) == 1, "围栏内的 # 行不应被识别为标题"


def test_no_title_document_fallback():
    md = "没有任何标题的纯文本内容"
    state = NodeDocumentSplit().process(make_state(md))
    assert len(state["chunks"]) == 1
    assert state["chunks"][0]["title"] == "无标题"


def test_crlf_normalized():
    md = "# 概述\r\n内容A\r\n# 安装\r\n内容B"
    state = NodeDocumentSplit().process(make_state(md))
    titles = [c["title"] for c in state["chunks"]]
    assert titles == ["概述", "安装"]
