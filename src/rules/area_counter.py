"""
区域内目标计数模块
统计 ROI 区域内当前的人/车数量（实时快照），并记录历史。
"""

import time
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime

from ..vision.detector import Detection
from .geometry import point_in_polygon, Polygon

logger = logging.getLogger(__name__)


class AreaCounter:
    """ROI 区域内目标计数器"""

    def __init__(self, roi: Optional[Polygon] = None, 
                 record_interval: float = 60.0):
        """
        Args:
            roi: ROI 多边形顶点列表，None 表示全画面
            record_interval: 记录间隔（秒），用于持久化历史数据
        """
        self.roi = [(float(p[0]), float(p[1])) for p in roi] if roi else None
        self.record_interval = record_interval
        
        self._current_counts: Dict[str, int] = defaultdict(int)
        self._current_tracks: Dict[int, str] = {}  # track_id -> class_name
        self._last_record_time = time.time()
        
        # 历史记录（最近 1 小时的快照）
        self._history: deque = deque(maxlen=3600)

    def update(self, detections: List[Detection]) -> Dict[str, int]:
        """
        更新区域内目标计数。
        
        Returns:
            当前区域内各类别的数量，如 {"person": 3, "car": 1}
        """
        counts: Dict[str, int] = defaultdict(int)
        tracks: Dict[int, str] = {}

        for det in detections:
            if det.track_id < 0:
                continue

            # ROI 过滤：只统计区域内的目标
            if self.roi and not point_in_polygon(det.foot, self.roi):
                continue

            counts[det.class_name] += 1
            tracks[det.track_id] = det.class_name

        self._current_counts = dict(counts)
        self._current_tracks = tracks
        return self._current_counts

    def get_counts(self) -> Dict[str, Any]:
        """获取当前区域内目标计数"""
        total = sum(self._current_counts.values())
        return {
            "total": total,
            "by_class": dict(self._current_counts),
            "person": self._current_counts.get("person", 0),
            "car": self._current_counts.get("car", 0),
            "truck": self._current_counts.get("truck", 0),
            "motorcycle": self._current_counts.get("motorcycle", 0),
            "bicycle": self._current_counts.get("bicycle", 0),
            "bus": self._current_counts.get("bus", 0),
        }

    def maybe_record(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        定期记录快照，用于持久化。
        
        Returns:
            如果到了记录时间，返回记录数据；否则返回 None
        """
        now = time.time()
        if now - self._last_record_time < self.record_interval:
            return None
        
        self._last_record_time = now
        counts = self.get_counts()
        
        record = {
            "type": "area_count",
            "camera_id": camera_id,
            "timestamp": now,
            "person": counts["person"],
            "car": counts["car"],
            "truck": counts["truck"],
            "motorcycle": counts["motorcycle"],
            "bicycle": counts["bicycle"],
            "bus": counts["bus"],
            "total": counts["total"],
        }
        
        # 保存到内存历史
        self._history.append(record)
        
        return record
