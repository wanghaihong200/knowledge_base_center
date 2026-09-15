"""PDF转Markdown节点：MinerU 云 API（申请链接→上传→轮询→下载解压）"""
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import (
    PdfConversionError,
    StateFieldError,
)


class NodePDFToMD(BaseNode):
    """
    节点功能: PDF 转 Markdown（MinerU 云端解析）
    流程: 校验路径 → 申请签名上传链接并上传 → 轮询解析结果
          → 下载结果 zip 解压 → 读取 Markdown 内容
    """

    name: str = "node_pdf_to_md"

    TIMEOUT_SECONDS = 600  # 轮询总超时
    POLL_INTERVAL = 3      # 轮询间隔（秒）
    REQUEST_TIMEOUT = 10   # 单请求超时（秒）

    def process(self, state: dict) -> dict:
        # 阶段一：校验路径并准备输出目录
        pdf_path, output_dir = self._step_1_validate_paths(state)
        file_stem = Path(pdf_path).stem

        # 阶段二：上传并轮询解析结果
        full_zip_url = self._step_2_upload_and_poll(pdf_path)

        # 阶段三：下载解压并读取 Markdown
        md_path, md_content = self._step_3_download_and_extract(
            full_zip_url, output_dir, file_stem
        )

        self.log_step("PDF 转换完成", f"{file_stem}.pdf → {file_stem}.md（{len(md_content)} 字符）")
        return {"pdf_path": pdf_path, "md_path": str(md_path), "md_content": md_content}

    # ==================== 阶段方法 ====================

    def _step_1_validate_paths(self, state: dict):
        pdf_path = state.get("pdf_path")
        file_dir = state.get("file_dir")
        if not pdf_path:
            raise StateFieldError(node_name=self.name, field_name="pdf_path", expected_type=str)
        if not Path(pdf_path).exists():
            raise PdfConversionError(f"PDF 文件不存在: {pdf_path}", node_name=self.name)
        output_dir = Path(file_dir) if file_dir else Path(pdf_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return pdf_path, output_dir

    def _step_2_upload_and_poll(self, pdf_path: str) -> str:
        """申请签名上传链接 → PUT 上传 → 轮询解析状态，返回结果 zip 下载地址"""
        headers = {"Authorization": f"Bearer {mineru_config.api_token}"}
        filename = Path(pdf_path).name

        # 1. 申请上传链接
        try:
            resp = requests.post(
                f"{mineru_config.base_url}/file-urls/batch",
                headers={**headers, "Content-Type": "application/json"},
                json={"files": [{"name": filename}], "model_version": "vlm"},
                timeout=self.REQUEST_TIMEOUT,
            )
            data = resp.json()["data"]
            upload_url = data["file_urls"][0]
            batch_id = data["batch_id"]
        except Exception as e:
            raise PdfConversionError(f"申请 MinerU 上传链接失败: {e}", node_name=self.name, cause=e)
        self.log_step("已获取上传链接", f"batch_id={batch_id}")

        # 2. 上传文件二进制
        put_resp = requests.put(upload_url, data=Path(pdf_path).read_bytes(), timeout=120)
        if put_resp.status_code != 200:
            raise PdfConversionError(
                f"MinerU 文件上传失败: HTTP {put_resp.status_code}", node_name=self.name
            )

        # 3. 轮询解析结果
        deadline = time.time() + self.TIMEOUT_SECONDS
        poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
        while time.time() < deadline:
            try:
                poll = requests.get(poll_url, headers=headers, timeout=self.REQUEST_TIMEOUT)
                extract_result = poll.json()["data"]["extract_result"][0]
            except Exception as e:
                self.logger.warning(f"轮询响应异常，将重试: {e}")
                time.sleep(self.POLL_INTERVAL)
                continue

            state_name = extract_result.get("state")
            if state_name == "done":
                full_zip_url = extract_result.get("full_zip_url")
                if not full_zip_url:
                    raise PdfConversionError("解析完成但缺少 full_zip_url", node_name=self.name)
                return full_zip_url
            if state_name == "failed":
                raise PdfConversionError(
                    f"MinerU 解析失败: {extract_result.get('err_msg')}", node_name=self.name
                )
            self.log_step("解析中", f"state={state_name}，{self.POLL_INTERVAL}s 后重试")
            time.sleep(self.POLL_INTERVAL)

        raise PdfConversionError(
            f"MinerU 解析超时（>{self.TIMEOUT_SECONDS}s）", node_name=self.name
        )

    def _step_3_download_and_extract(self, full_zip_url: str, output_dir: Path, file_stem: str):
        """下载结果 zip → 清理解压目录 → 解压 → full.md 重命名为 {stem}.md 并读取"""
        zip_path = output_dir / f"{file_stem}_result.zip"
        self._download_zip(full_zip_url, zip_path)

        extract_dir = output_dir / file_stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)  # 幂等：清空旧解压目录
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # full.md → {stem}.md
        source_md = self._find_file(extract_dir, "full.md")
        if source_md is None:
            raise PdfConversionError(
                f"解压结果中未找到 full.md: {extract_dir}", node_name=self.name
            )
        md_path = output_dir / f"{file_stem}.md"
        shutil.move(str(source_md), str(md_path))
        return md_path, md_path.read_text(encoding="utf-8")

    @staticmethod
    def _find_file(root: Path, name: str):
        """在解压目录中递归查找指定文件名"""
        for path in root.rglob(name):
            if path.is_file():
                return path
        return None

    def _download_zip(self, url: str, dest: Path) -> None:
        """
        下载结果 zip。

        注：cdn-mineru.openxlab.org.cn 对 Python OpenSSL 的 TLS 握手有干扰
        （SSLEOFError，系统 curl/Schannel 可正常下载），故 Python 请求失败时
        回退到系统 curl。
        """
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            self.log_step("Python 下载失败，回退 curl", f"{type(e).__name__}: {str(e)[:120]}")

        result = subprocess.run(
            ["curl", "-sSL", "--max-time", "180", "-o", str(dest), url],
            capture_output=True,
            timeout=200,
        )
        if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            stderr = result.stderr.decode(errors="replace")[:200]
            raise PdfConversionError(f"下载解析结果失败（curl 回退也失败）: {stderr}", node_name=self.name)
