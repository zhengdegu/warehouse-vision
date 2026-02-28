"""
线程安全 JSONL 日志系统
按摄像头分文件存储告警事件，支持按摄像头查询历史。
持久化计数和告警统计。
"""

import os
import json
import threading
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class JSONLLogger:
    """线程安全的 JSONL 日志写入器，按摄像头分文件"""

    def __init__(self, output_dir: str = "events"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._counts_path = os.path.join(output_dir, "counts.jsonl")
        self._stats_path = os.path.join(output_dir, "camera_stats.json")
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._counts_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        # camera_id -> {total_in, total_out, alert_count}
        self._camera_stats: Dict[str, Dict[str, int]] = self._load_stats()

        logger.info(f"JSONL 日志目录: {output_dir}")

    def _load_stats(self) -> Dict[str, Dict[str, int]]:
        """从本地加载摄像头统计数据"""
        if os.path.exists(self._stats_path):
            try:
                with open(self._stats_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载统计数据失败: {e}")
        return {}

    def _save_stats(self):
        """持久化统计数据到本地"""
        try:
            with open(self._stats_path, "w", encoding="utf-8") as f:
                json.dump(self._camera_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存统计数据失败: {e}")

    def _ensure_camera_stats(self, camera_id: str):
        if camera_id not in self._camera_stats:
            self._camera_stats[camera_id] = {
                "total_in": 0, "total_out": 0, "alert_count": 0
            }

    def get_camera_stats(self, camera_id: str = None) -> Dict:
        """获取摄像头统计数据"""
        with self._stats_lock:
            if camera_id:
                self._ensure_camera_stats(camera_id)
                return dict(self._camera_stats.get(camera_id, {}))
            return {k: dict(v) for k, v in self._camera_stats.items()}

    def increment_alert(self, camera_id: str):
        """告警计数+1"""
        with self._stats_lock:
            self._ensure_camera_stats(camera_id)
            self._camera_stats[camera_id]["alert_count"] += 1
            self._save_stats()

    def update_counts(self, camera_id: str, total_in: int, total_out: int):
        """更新进出计数"""
        with self._stats_lock:
            self._ensure_camera_stats(camera_id)
            self._camera_stats[camera_id]["total_in"] = total_in
            self._camera_stats[camera_id]["total_out"] = total_out
            self._save_stats()

    def _get_lock(self, camera_id: str) -> threading.Lock:
        with self._global_lock:
            if camera_id not in self._locks:
                self._locks[camera_id] = threading.Lock()
            return self._locks[camera_id]

    def _event_path(self, camera_id: str) -> str:
        return os.path.join(self.output_dir, f"events_{camera_id}.jsonl")

    def log_event(self, event: Dict[str, Any]):
        """写入事件日志，按摄像头分文件"""
        camera_id = event.get("camera_id", "unknown")
        lock = self._get_lock(camera_id)
        with lock:
            try:
                path = self._event_path(camera_id)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"写入事件日志失败: {e}")
        # 更新告警计数
        self.increment_alert(camera_id)

    def log_count(self, count: Dict[str, Any]):
        """写入计数日志"""
        with self._counts_lock:
            try:
                with open(self._counts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(count, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"写入 counts.jsonl 失败: {e}")
        # 持久化进出计数
        camera_id = count.get("camera_id")
        if camera_id:
            self.update_counts(
                camera_id,
                count.get("total_in", 0),
                count.get("total_out", 0)
            )

    def read_events(self, limit: int = 100, camera_id: str = None) -> list:
        """读取事件，可按摄像头过滤"""
        if camera_id:
            return self._read_jsonl(self._event_path(camera_id), limit)
        all_events = []
        try:
            for f in os.listdir(self.output_dir):
                if f.startswith("events_") and f.endswith(".jsonl"):
                    path = os.path.join(self.output_dir, f)
                    all_events.extend(self._read_jsonl(path, limit))
        except Exception as e:
            logger.error(f"读取事件失败: {e}")
        all_events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return all_events[:limit]

    def read_counts(self, limit: int = 100) -> list:
        return self._read_jsonl(self._counts_path, limit)

    def _read_jsonl(self, path: str, limit: int) -> list:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            results = []
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
            return results
        except Exception as e:
            logger.error(f"读取 {path} 失败: {e}")
            return []
