# knowledge_base_center Docker 环境

AI 知识库项目全套依赖，统一由一个 `docker-compose.yml` 管理
（compose 项目名：`knowledge_base_center`，见 yml 顶层 `name:` 字段）。

## 启动 / 停止

```bash
cd docker
docker compose up -d      # 启动全部 5 个服务
docker compose down       # 停止；数据卷保留，再次 up 数据仍在
docker compose down -v    # 停止并销毁卷(危险：mongo 数据清空)
```

镜像来源：课件离线包 `docker load -i *.tar`（milvus / milvus-etcd /
milvus-minio / attu / mongo），无需联网拉取。

## 服务一览

| 服务 | 容器名 | 端口 | 说明 |
|------|--------|------|------|
| Milvus v2.5.5 | milvus-standalone | 19530 | SDK 连接 `http://localhost:19530` |
| | | 9091 | 健康检查 `http://localhost:9091/healthz` |
| Attu v2.5.10 | milvus-attu | 7000 | Web 管理台 http://localhost:7000 （Milvus 地址填 `standalone:19530`） |
| MinIO | milvus-minio | 9000 | S3 API（控制台分享链接默认签到此端口，勿改成非标准映射） |
| | | 9003 | MinIO 控制台 http://localhost:9003 （minioadmin/minioadmin） |
| etcd v3.5.18 | milvus-etcd | - | Milvus 元数据，仅内部访问 |
| MongoDB 8 | mongo | 27017 | 连接串 `mongodb://localhost:27017`，无认证 |

## 数据存放

| 数据 | 位置 | 说明 |
|------|------|------|
| Milvus/etcd/minio | `docker/volumes/` | bind mount，已 gitignore |
| MongoDB | named volume `knowledge_base_center_mongo_data` | `docker compose down -v` 才会删除 |

## 连接方式（Python 侧）

```python
from pymilvus import MilvusClient
milvus = MilvusClient(uri="http://localhost:19530")

from pymongo import MongoClient
mongo = MongoClient("mongodb://localhost:27017")
```

## 注意事项

- mongo 之外的服务未配置 restart 策略，Docker Desktop 重启后需重新
  `docker compose up -d`（mongo 会自动拉起）。
- compose 顶层 `name: knowledge_base_center` 固定了项目名，在任意目录
  执行 compose 命令操作的均是同一套环境。
- 课件目录里另有 `minio.tar`(独立版 minio) 与 `sk-all.tar`，本环境暂未使用。
