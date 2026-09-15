"""入口节点：校验导入文件路径与类型，决定走 PDF 还是 MD 分支"""
from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import (
    FileProcessingError,
    StateFieldError,
    ValidationError,
)


class NodeEntry(BaseNode):
    """
    节点功能: 入口检查
    流程: 校验路径 → 按后缀路由（.pdf → PDF 分支 / .md → MD 分支 / 其他 → 拒绝）
    """

    name: str = "node_entry"

    def process(self, state: dict) -> dict:
        # 阶段一：校验路径字段
        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise StateFieldError(
                node_name=self.name, field_name="import_file_path", expected_type=str
            )
        self.log_step("校验文件路径", import_file_path)

        path = Path(import_file_path)
        if not path.exists():
            raise FileProcessingError(f"文件不存在: {import_file_path}", node_name=self.name)

        # 阶段二：按后缀路由
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            self.log_step("识别为 PDF 文件", path.name)
            return {
                "is_pdf_read_enabled": True,
                "is_md_read_enabled": False,
                "pdf_path": str(path),
                "file_title": path.stem,
            }
        if suffix == ".md":
            self.log_step("识别为 Markdown 文件", path.name)
            return {
                "is_pdf_read_enabled": False,
                "is_md_read_enabled": True,
                "md_path": str(path),
                "file_title": path.stem,
            }
        raise ValidationError(f"不支持的文件类型: {suffix}（仅支持 .pdf/.md）", node_name=self.name)
