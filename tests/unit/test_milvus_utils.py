"""Milvus 工具单元测试：表达式转义等纯逻辑"""
import pytest

from utils.milvus_utils import escape_milvus_string

pytestmark = pytest.mark.unit


def test_escape_milvus_string():
    assert escape_milvus_string("a'b") == "a\\'b"
    assert escape_milvus_string('a"b') == 'a\\"b'
    assert escape_milvus_string("a\\b") == "a\\\\b"
    assert escape_milvus_string("普通名称-001") == "普通名称-001"
