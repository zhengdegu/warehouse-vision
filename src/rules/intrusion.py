"""
入侵检测模块
基于 ROI 多边形判定目标是否入侵禁区，
支持连续 confirm_frames 确认和 per-track cooldown 防抖。
"""

import time
import logging
from typing import List, Dict, Any, Optional
from ..vision.detector import Detection
from .geometry import point_in_polygon, Point, Polygon

logger = logging.getLogger(__name__)


class IntrusionDetector:
    """ROI 入侵检测器"""

    def __init__(self, roi: Polygon, confirm_frames: int = 5,
                 cooldown: float = 30.0):
        """
        Args:
            roi: ROI 多边形顶点列表
            confirm_frames: 连续多少帧确认为入侵
            cooldown: 同一 track 触发后的冷却时间（秒）
        """
        self.roi = [(float(p[0]), float(p[1])) for p in roi]
        self.confirm_frames = confirm_frames
        self.cooldown = cooldown

        # track_id -> 连续在 ROI 内的帧数
        self._confirm_count: Dict[int, int] = {}
        # track_id -> 上次触发时间
        self._last_trigger: Dict[int, float] = {}

    def update(self, detections: List[Detection],
               camera_id: str = "") -> List[Dict[str, Any]]:
        """
        更新检测结果，返回入侵事件列表。
        """
        events = []
        now = time.time()
        active_ids = set()

        for det in detections:
            if det.track_id < 0:
                continue

            active_ids.add(det.track_id)
            # 使用脚点判断是否在 ROI 内
            in_roi = point_in_polygon(det.foot, self.roi)

            if in_roi:
                self._confirm_count[det.track_id] = \
                    self._confirm_count.get(det.track_id, 0) + 1
            else:
                self._confirm_count[det.track_id] = 0
                continue

            # 连续帧确认
            if self._confirm_count[det.track_id] < self.confirm_frames:
                continue

            # cooldown 检查
            last = self._last_trigger.get(det.track_id, 0)
            if now - last < self.cooldown:
                continue

            # 触发入侵事件
            self._last_trigger[det.track_id] = now
            events.append({
                "type": "intrusion",
                "camera_id": camera_id,
                "track_id": det.track_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "center": det.center,
                "timestamp": now,
            })
            logger.info(f"[入侵] cam={camera_id} track={det.track_id} "
                        f"class={det.class_name}")

        # 清理已消失的 track
        for tid in list(self._confirm_count.keys()):
            if tid not in active_ids:
                del self._confirm_count[tid]

        return events
