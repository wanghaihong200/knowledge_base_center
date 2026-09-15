"""MinIO 工具：客户端单例、桶初始化与公共读策略"""
import json
import logging

from minio import Minio

from config.minio_config import minio_config

logger = logging.getLogger("minio")

_minio_client = None


def get_minio_client() -> Minio:
    """获取 MinIO 客户端单例；首次调用时确保桶存在并设置公共读策略"""
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.secure,
        )
        _ensure_bucket(_minio_client)
    return _minio_client


def _ensure_bucket(client: Minio) -> None:
    """桶不存在则创建；设置匿名公共读策略（前端 <img> 直接访问图片 URL）"""
    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)
        logger.info(f"已创建桶 {minio_config.bucket_name}")

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"],
            }
        ],
    }
    client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))
