"""文档切片节点：标题初切 → 超长二次切分 → 过短合并"""
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import (
    DocumentSplitError,
    StateFieldError,
)


class NodeDocumentSplit(BaseNode):
    """
    节点功能: 文档切片
    流程: 统一换行符 → 按标题切分章节（跳过代码块围栏）→ 超长章节递归二次切分
          → 同章节内过短子块向前合并 → 统计并备份 chunks.json
    """

    name: str = "node_document_split"

    # 标题行：1-6 级 # 开头
    TITLE_PATTERN = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
    # 代码围栏开始/结束（``` 或 ~~~，允许携带语言标识）
    FENCE_PATTERN = re.compile(r"^(`{3,}|~{3,})")

    def process(self, state: dict) -> dict:
        # 阶段一：获取并校验输入，统一换行符
        file_title, md_content = self._step_1_get_inputs(state)

        # 阶段二：按标题切分章节
        sections = self._step_2_split_by_titles(md_content, file_title)

        # 阶段三：无标题兜底
        if not sections:
            sections = [{"title": "无标题", "content": md_content, "file_title": file_title}]

        # 阶段四：超长切分 + 短块合并
        try:
            chunks = self._step_4_refine_chunks(sections)
        except Exception as e:
            raise DocumentSplitError(f"切片细化失败: {e}", node_name=self.name, cause=e)

        # 阶段五：统计日志
        self._step_5_print_stats(chunks)

        # 阶段六：备份 chunks.json（失败仅记日志，不中断流程）
        self._step_6_backup(state, chunks)

        return {"chunks": chunks}

    # ==================== 阶段方法 ====================

    def _step_1_get_inputs(self, state: dict):
        file_title = state.get("file_title")
        md_content = state.get("md_content")
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        if not md_content:
            raise StateFieldError(node_name=self.name, field_name="md_content", expected_type=str)
        return file_title, re.sub(r"\r\n|\r", "\n", md_content)

    def _step_2_split_by_titles(self, md_content: str, file_title: str) -> list:
        """逐行扫描，标题行（围栏外）开启新章节；返回 [{title, content, file_title}]"""
        sections: list = []
        current_title = None
        current_lines: list = []
        fence_marker = None  # 当前所处代码围栏标记（``` 或 ~~~），None 表示不在围栏内

        def flush():
            nonlocal current_lines
            if current_title is not None:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                    "file_title": file_title,
                })
            current_lines = []

        for line in md_content.split("\n"):
            stripped = line.strip()

            # 代码围栏状态机：开始/结束必须使用相同标记
            if fence_marker is None and self.FENCE_PATTERN.match(stripped):
                fence_marker = stripped[:3]
                current_lines.append(line)
                continue
            if fence_marker is not None:
                current_lines.append(line)
                if stripped.startswith(fence_marker):
                    fence_marker = None
                continue

            match = self.TITLE_PATTERN.match(line)
            if match:
                flush()
                current_title = match.group(2).strip()
            else:
                if current_title is None:
                    # 标题前的散落内容也归属一个隐式章节，标题定为「无标题」
                    current_title = "无标题"
                current_lines.append(line)

        flush()
        # 过滤空章节
        return [s for s in sections if s["content"]]

    def _step_4_refine_chunks(self, sections: list) -> list:
        """超长章节二次切分，再对同章节过短子块合并"""
        max_length = self.config.max_content_length
        refined: list = []

        for sec in sections:
            title = sec["title"]
            # 预留标题长度，保证拼接元数据后不超限
            available = max(max_length - len(title), 200)
            if len(sec["content"]) <= available:
                refined.append({
                    "title": title,
                    "content": sec["content"],
                    "parent_title": title,
                    "part": 0,
                    "file_title": sec["file_title"],
                })
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=available,
                chunk_overlap=0,
                separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            )
            sub_texts = splitter.split_text(sec["content"])
            for idx, sub in enumerate(sub_texts):
                refined.append({
                    "title": title if idx == 0 else f"{title}-{idx}",
                    "content": sub,
                    "parent_title": title,
                    "part": idx,
                    "file_title": sec["file_title"],
                })

        return self._merge_short_sections(refined)

    def _merge_short_sections(self, chunks: list) -> list:
        """相邻且同 parent_title 的子块，内容长度低于 min_content_length 时向前合并"""
        min_length = self.config.min_content_length
        merged: list = []
        for chunk in chunks:
            if (
                merged
                and len(chunk["content"]) < min_length
                and chunk["parent_title"] == merged[-1]["parent_title"]
            ):
                prev = merged[-1]
                prev["content"] = f"{prev['content']}\n{chunk['content']}".strip()
                prev["part"] = chunk["part"]  # part 取最新
                prev["title"] = prev["title"] if prev["part"] == 0 else chunk["title"]
            else:
                merged.append(dict(chunk))
        return merged

    def _step_5_print_stats(self, chunks: list) -> None:
        lengths = [len(c["content"]) for c in chunks]
        self.log_step(
            "切片统计",
            f"共 {len(chunks)} 块，最长 {max(lengths) if lengths else 0} 字符，"
            f"最短 {min(lengths) if lengths else 0} 字符",
        )

    def _step_6_backup(self, state: dict, chunks: list) -> None:
        file_dir = state.get("file_dir")
        if not file_dir:
            return
        try:
            backup_path = Path(file_dir) / "chunks.json"
            backup_path.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.log_step("切片备份", str(backup_path))
        except Exception as e:
            self.logger.warning(f"备份 chunks.json 失败（不中断）: {e}")
