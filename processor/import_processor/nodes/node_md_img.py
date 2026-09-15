"""MD图片处理节点：本地图片上传 MinIO，VLM 生成摘要替换图片描述"""
import base64
import re
import time
from collections import deque
from pathlib import Path

from langchain_core.messages import HumanMessage

from config.minio_config import minio_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import ImageProcessingError, StateFieldError
from config.lm_config import lm_config
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    节点功能: Markdown 图片处理
    流程: 扫描 MD 引用的本地图片（取前后文做 VLM 提示）→ 限流调用 VLM 生成摘要
          → 图片上传 MinIO（先幂等清理文档目录）→ MD 中替换为 MinIO URL + 摘要
          → 备份为 *_new.md 并更新 state
    """

    name: str = "node_md_img"

    RATE_LIMIT_MAX_REQUESTS = 10  # 滑动窗口限流：60s 内最多 10 次 VLM 调用
    RATE_LIMIT_WINDOW_SECONDS = 60
    CONTEXT_CHARS = 100           # 图片在 MD 中前后各取的上下文字符数

    def process(self, state: dict) -> dict:
        # 阶段一：获取输入与图片目录
        md_path, md_content = self._step_1_get_content(state)
        md_file = Path(md_path)
        images_dir = md_file.parent / "images"
        doc_stem = md_file.stem

        # 阶段二：扫描 MD 中的本地图片引用（含上下文）
        image_refs = self._step_2_scan_images(md_content, images_dir)
        self.log_step("扫描图片", f"发现 {len(image_refs)} 张本地图片")
        if not image_refs:
            return {"md_path": md_path, "md_content": md_content}

        # 阶段三：限流调用 VLM 生成摘要
        summaries = self._step_3_generate_summaries(image_refs)

        # 阶段四：上传 MinIO 并替换 MD 引用
        new_content = self._step_4_upload_and_replace(
            md_content, image_refs, summaries, doc_stem
        )

        # 阶段五：备份 *_new.md 并更新 state
        new_md_path = self._step_5_backup_new_md_file(md_file, new_content)

        return {"md_path": str(new_md_path), "md_content": new_content}

    # ==================== 阶段方法 ====================

    def _step_1_get_content(self, state: dict):
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(node_name=self.name, field_name="md_path", expected_type=str)
        md_content = state.get("md_content")
        if not md_content:
            # MD 直传分支：entry 节点只设置了 md_path，此处读取文件内容
            path = Path(md_path)
            if not path.exists():
                raise StateFieldError(
                    node_name=self.name, field_name="md_path",
                    message=f"MD 文件不存在: {md_path}",
                )
            md_content = path.read_text(encoding="utf-8")
        return md_path, md_content

    def _step_2_scan_images(self, md_content: str, images_dir: Path) -> list:
        """扫描 images 目录下、且在 MD 中被引用的图片，取前后上下文供 VLM 理解"""
        if not images_dir.exists():
            return []
        refs = []
        for image_file in sorted(images_dir.iterdir()):
            if image_file.suffix.lower() not in self.config.image_extensions:
                continue
            found = self._find_image_in_md(md_content, image_file.name)
            if found is None:
                continue
            start, end = found
            refs.append({
                "name": image_file.name,
                "path": str(image_file),
                "before": md_content[max(0, start - self.CONTEXT_CHARS):start],
                "after": md_content[end:end + self.CONTEXT_CHARS],
            })
        return refs

    def _find_image_in_md(self, md_content: str, image_name: str):
        """定位 `![...](...图片名...)` 引用位置，返回 (start, end)；未引用返回 None"""
        pattern = re.compile(r"!\[[^\]]*\]\([^)]*?" + re.escape(image_name) + r"[^)]*\)")
        match = pattern.search(md_content)
        return (match.start(), match.end()) if match else None

    def _step_3_generate_summaries(self, image_refs: list) -> list:
        """逐图调用 VLM 生成中文摘要；单图失败兜底「图片描述」"""
        request_times: deque = deque()
        summaries = []
        for ref in image_refs:
            self._apply_api_rate_limit(request_times)
            summaries.append(self._summarize_image(ref))
        return summaries

    def _apply_api_rate_limit(self, request_times: deque) -> None:
        """滑动窗口限流：窗口内达到上限时等待最旧请求滑出窗口"""
        now = time.time()
        while request_times and now - request_times[0] >= self.RATE_LIMIT_WINDOW_SECONDS:
            request_times.popleft()
        if len(request_times) >= self.RATE_LIMIT_MAX_REQUESTS:
            sleep_seconds = self.RATE_LIMIT_WINDOW_SECONDS - (now - request_times[0])
            if sleep_seconds > 0:
                self.log_step("限流等待", f"{sleep_seconds:.1f}s")
                time.sleep(sleep_seconds)
        request_times.append(time.time())

    def _summarize_image(self, ref: dict) -> str:
        """VLM 按图片内容 + MD 上下文生成简短中文标题（base64 内联）"""
        root_folder = Path(ref["path"]).parent.parent.stem
        prompt = (
            f'这是"{root_folder}"文件中的一张图片，'
            f'图片上文部分为"{ref["before"]}"，下文部分为"{ref["after"]}"，'
            "请为这张图片起一个中文标题。只输出标题本身，不超过15个字，"
            "不要任何解释、引号、标点符号或格式。"
        )
        try:
            image_b64 = base64.b64encode(Path(ref["path"]).read_bytes()).decode("utf-8")
            suffix = Path(ref["path"]).suffix.lstrip(".").lower()
            mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{image_b64}"}},
            ])
            llm = get_llm_client(model=lm_config.vl_model)
            summary = (llm.invoke([message]).content or "").strip()
            return self._sanitize_summary(summary)
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败（{ref['name']}），使用兜底文案: {e}")
            return "图片描述"

    @staticmethod
    def _sanitize_summary(summary: str) -> str:
        """摘要净化：取首行、去格式残留、限长——alt 文本必须单行，否则破坏 MD 图片语法"""
        line = summary.splitlines()[0].strip() if summary.strip() else ""
        line = line.strip("*#`\"'“”‘’[]【】。．")
        return line[:30] or "图片描述"

    def _step_4_upload_and_replace(self, md_content: str, image_refs: list,
                                   summaries: list, doc_stem: str) -> str:
        """清理文档图片目录（幂等）→ 上传全部图片 → MD 中替换 alt 与 URL"""
        try:
            client = get_minio_client()
            dir_prefix = f"{minio_config.img_dir}/{doc_stem}".replace(" ", "")
            self._clean_minio_directory(client, dir_prefix)
            for ref, summary in zip(image_refs, summaries):
                object_name = f"{dir_prefix}/{ref['name']}".replace(" ", "")
                url = self._upload_to_minio(client, object_name, ref)
                md_content = self._replace_image_ref(
                    md_content, ref["name"], summary, url
                )
        except Exception as e:
            raise ImageProcessingError(f"图片上传/替换失败: {e}", node_name=self.name, cause=e)
        return md_content

    def _clean_minio_directory(self, client, dir_prefix: str) -> None:
        """删除文档图片目录下的旧对象（幂等清理）"""
        old_objects = [
            obj.object_name for obj in client.list_objects(
                bucket_name=minio_config.bucket_name, prefix=f"{dir_prefix}/", recursive=True
            )
        ]
        if old_objects:
            from minio.deleteobjects import DeleteObject
            client.remove_objects(
                bucket_name=minio_config.bucket_name,
                delete_object_list=[DeleteObject(name) for name in old_objects],
            )
            self.log_step("清理旧图片", f"{len(old_objects)} 个对象")

    def _upload_to_minio(self, client, object_name: str, ref: dict) -> str:
        """上传单张图片，返回可公开访问的 URL"""
        content_type = f"image/{Path(ref['path']).suffix.lstrip('.').lower()}"
        client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=ref["path"],
            content_type=content_type,
        )
        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
        self.log_step("图片已上传", url)
        return url

    @staticmethod
    def _replace_image_ref(md_content: str, image_name: str, summary: str, url: str) -> str:
        """把 MD 中该图片的引用替换为 `![摘要](MinIO URL)`"""
        pattern = re.compile(r"!\[[^\]]*\]\([^)]*?" + re.escape(image_name) + r"[^)]*\)")
        return pattern.sub(lambda m: f"![{summary}]({url})", md_content)

    def _step_5_backup_new_md_file(self, md_file: Path, new_content: str) -> Path:
        """替换结果备份为 {stem}_new.md"""
        new_path = md_file.with_name(f"{md_file.stem}_new.md")
        new_path.write_text(new_content, encoding="utf-8")
        self.log_step("备份新 MD", str(new_path))
        return new_path
