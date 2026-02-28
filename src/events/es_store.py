"""
Elasticsearch 告警存储模块
将告警事件写入 ES，支持全文检索和聚合分析。
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# 索引映射
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "event_type":    {"type": "keyword"},
            "sub_type":      {"type": "keyword"},
            "camera_id":     {"type": "keyword"},
            "track_id":      {"type": "integer"},
            "class_name":    {"type": "keyword"},
            "confidence":    {"type": "float"},
            "bbox":          {"type": "float"},
            "detail":        {"type": "text", "analyzer": "standard"},
            "screenshot":    {"type": "keyword"},
            "crossing_direction": {"type": "keyword"},
            "tripwire_id":   {"type": "keyword"},
            "tripwire_name": {"type": "keyword"},
            "timestamp":     {"type": "date", "format": "epoch_second||epoch_millis||strict_date_optional_time"},
            "@timestamp":    {"type": "date"},
            "created_at":    {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "5s"
    }
}


class ESEventStore:
    """Elasticsearch 告警存储"""

    def __init__(self, host: str = "http://localhost:9222",
                 index_prefix: str = "warehouse-alerts",
                 enabled: bool = True):
        self.host = host
        self.index_prefix = index_prefix
        self.enabled = enabled
        self._es = None
        self._lock = threading.Lock()

        if not enabled:
            logger.info("ES 存储已禁用")
            return

        try:
            from elasticsearch import Elasticsearch
            self._es = Elasticsearch(host)
            info = self._es.info()
            logger.info(f"ES 连接成功: {info['version']['number']}")
            self._ensure_index()
        except Exception as e:
            logger.error(f"ES 连接失败: {e}，告警仍会写入本地数据库")
            self._es = None

    def _get_index_name(self) -> str:
        """按月分索引: warehouse-alerts-2026.02"""
        return f"{self.index_prefix}-{datetime.now().strftime('%Y.%m')}"

    def _ensure_index(self):
        """确保索引存在"""
        if not self._es:
            return
        idx = self._get_index_name()
        try:
            if not self._es.indices.exists(index=idx):
                self._es.indices.create(index=idx, body=INDEX_MAPPING)
                logger.info(f"ES 索引已创建: {idx}")
        except Exception as e:
            logger.error(f"ES 创建索引失败: {e}")

    def store_event(self, event: Dict[str, Any]):
        """写入单条告警事件到 ES"""
        if not self._es:
            return

        try:
            ts = event.get("timestamp", time.time())
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            doc = {
                "event_type": event.get("type", "unknown"),
                "sub_type": event.get("sub_type", ""),
                "camera_id": event.get("camera_id", ""),
                "track_id": event.get("track_id", -1),
                "class_name": event.get("class_name", ""),
                "confidence": event.get("confidence", 0),
                "bbox": event.get("bbox", []),
                "detail": event.get("detail", ""),
                "screenshot": event.get("screenshot", ""),
                "crossing_direction": event.get("crossing_direction", ""),
                "tripwire_id": event.get("tripwire_id", ""),
                "tripwire_name": event.get("tripwire_name", ""),
                "timestamp": ts,
                "@timestamp": dt.isoformat(),
                "created_at": dt.isoformat(),
            }

            idx = self._get_index_name()
            self._es.index(index=idx, body=doc)
        except Exception as e:
            logger.error(f"ES 写入失败: {e}")

    def bulk_store(self, events: List[Dict[str, Any]]):
        """批量写入"""
        if not self._es or not events:
            return

        try:
            from elasticsearch.helpers import bulk
            idx = self._get_index_name()
            actions = []
            for event in events:
                ts = event.get("timestamp", time.time())
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                actions.append({
                    "_index": idx,
                    "_source": {
                        "event_type": event.get("type", "unknown"),
                        "sub_type": event.get("sub_type", ""),
                        "camera_id": event.get("camera_id", ""),
                        "track_id": event.get("track_id", -1),
                        "class_name": event.get("class_name", ""),
                        "confidence": event.get("confidence", 0),
                        "bbox": event.get("bbox", []),
                        "detail": event.get("detail", ""),
                        "screenshot": event.get("screenshot", ""),
                        "crossing_direction": event.get("crossing_direction", ""),
                        "tripwire_id": event.get("tripwire_id", ""),
                        "tripwire_name": event.get("tripwire_name", ""),
                        "timestamp": ts,
                        "@timestamp": dt.isoformat(),
                        "created_at": dt.isoformat(),
                    }
                })
            bulk(self._es, actions)
            logger.debug(f"ES 批量写入 {len(actions)} 条")
        except Exception as e:
            logger.error(f"ES 批量写入失败: {e}")

    def store_camera_config(self, cam_config: Dict[str, Any]):
        """写入/更新摄像头配置到 ES，用 camera_id 作为文档 ID（upsert）"""
        if not self._es:
            return
        idx = "warehouse-cameras"
        try:
            if not self._es.indices.exists(index=idx):
                self._es.indices.create(index=idx, body={
                    "mappings": {
                        "properties": {
                            "camera_id":  {"type": "keyword"},
                            "name":       {"type": "keyword"},
                            "url":        {"type": "keyword"},
                            "width":      {"type": "integer"},
                            "height":     {"type": "integer"},
                            "fps":        {"type": "integer"},
                            "roi":        {"type": "float"},
                            "tripwires":  {"type": "object", "enabled": False},
                            "rules":      {"type": "object", "enabled": False},
                            "motion":     {"type": "object", "enabled": False},
                            "updated_at": {"type": "date"},
                        }
                    },
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0}
                })
                logger.info(f"ES 索引已创建: {idx}")

            doc = dict(cam_config)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()
            cam_id = doc.get("id", doc.get("camera_id", "unknown"))
            doc["camera_id"] = cam_id

            self._es.index(index=idx, id=cam_id, body=doc)
            logger.info(f"ES 摄像头配置已写入: {cam_id}")
        except Exception as e:
            logger.error(f"ES 写入摄像头配置失败: {e}")

    def delete_camera_config(self, camera_id: str):
        """从 ES 删除摄像头配置"""
        if not self._es:
            return
        try:
            self._es.delete(index="warehouse-cameras", id=camera_id, ignore=[404])
            logger.info(f"ES 摄像头配置已删除: {camera_id}")
        except Exception as e:
            logger.error(f"ES 删除摄像头配置失败: {e}")

    @property
    def connected(self) -> bool:
        return self._es is not None
