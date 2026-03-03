"""
人/车辆存在检测模块
检测到指定类别目标（人、车辆等）即触发报警，
支持 per-track cooldown 防止重复报警。
"""

import time
import logging
from typing import List, Dict, Any, Set

from ..vision.detector import Detection
from .geometry import point_in_polygon

logger = logging.getLogger(__name__)

# COCO 类别映射
CLASS_LABELS = {
    0: "人员", 1: "自行车", 2: "汽车", 3: "摩托车", 5: "公交车", 7: "卡车"
}


class PresenceDetector:
    """目标存在检测器 - 检测到人/车辆即报警"""

    def __init__(self, watch_classes: List[int] = None,
                 cooldown: float = 10.0,
                 min_confidence: float = 0.5,
                 roi: List[List[float]] = None):
        self.watch_classes = watch_classes or [0, 2, 3, 5, 7]
        self.cooldown = cooldown
        self.min_confidence = min_confidence
        self._last_trigger: Dict[int, float] = {}
        # ROI 多边形，设置后只检测区域内的目标
        self.roi = [(float(p[0]), float(p[1])) for p in roi] if roi else None


    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        events = []
        now = frame_ts if frame_ts > 0 else time.time()

        for det in detections:
            if det.track_id < 0:
                continue
            if det.class_id not in self.watch_classes:
                continue
            if det.confidence < self.min_confidence:
                continue

            # ROI 过滤：只检测区域内的目标
            if self.roi and not point_in_polygon(det.foot, self.roi):
                continue

            last = self._last_trigger.get(det.track_id, 0)
            if now - last < self.cooldown:
                continue

            self._last_trigger[det.track_id] = now
            label = CLASS_LABELS.get(det.class_id, det.class_name)
            events.append({
                "type": "presence",
                "sub_type": label,
                "camera_id": camera_id,
                "track_id": det.track_id,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": round(det.confidence, 2),
                "bbox": det.bbox,
                "center": det.center,
                "timestamp": now,
                "detail": f"检测到{label} (置信度:{det.confidence:.0%})",
            })
            logger.info(f"[存在检测] cam={camera_id} track={det.track_id} "
                        f"class={label} conf={det.confidence:.2f}")

        # 清理过期 track
        expired = [k for k, v in self._last_trigger.items()
                   if now - v > self.cooldown * 5]
        for k in expired:
            del self._last_trigger[k]

        return events
