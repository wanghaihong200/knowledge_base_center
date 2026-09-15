"""MinIO 对象存储配置"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MinIOConfig:
    """MinIO 连接与桶配置"""

    endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket_name: str = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")
    # Markdown 图片上传目录（按文档名分目录）
    img_dir: str = os.getenv("MINIO_IMG_DIR", "upload-images")
    secure: bool = False


minio_config = MinIOConfig()
