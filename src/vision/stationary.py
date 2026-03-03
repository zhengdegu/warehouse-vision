"""
静止目标跟踪器 — 跳过未移动的目标，减少重复推理

参考 Frigate 的 stationary_object_ids 机制：
- 跟踪每个目标的位置历史
- 连续 N 帧未移动超过阈值 → 标记为静止
- 静止目标复用上次检测结果，不参与推理
- 每隔 K 帧重新检测静止目标（防止漏检离开）
- 运动区域与静止目标重叠时，强制重新检测
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detector import Detection
from .motion import MotionBox

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_MOVE_THRESHOLD = 20.0     # 像素，中心点移动小于此值视为未移动
DEFAULT_STATIONARY_FRAMES = 15    # 连续未移动帧数 → 标记静止
DEFAULT_RECHECK_INTERVAL = 30     # 每隔多少帧重新检测静止目标


@dataclass
class TrackedObject:
    """被跟踪目标的状态"""
    track_id: int
    detection: Detection
    last_center: Tuple[float, float]
    still_frames: int = 0           # 连续未移动帧数
    is_stationary: bool = False
    frames_since_recheck: int = 0   # 距上次重新检测的帧数
    last_update_time: float = field(default_factory=time.time)


class StationaryTracker:
    """
    静止目标管理器。
    在 _analyze_loop 中插入：检测后更新状态，下次检测前过滤静止目标。
    """

    def __init__(self,
                 move_threshold: float = DEFAULT_MOVE_THRESHOLD,
                 stationary_frames: int = DEFAULT_STATIONARY_FRAMES,
                 recheck_interval: int = DEFAULT_RECHECK_INTERVAL,
                 expire_seconds: float = 60.0):
        self.move_threshold = move_threshold
        self.stationary_frames = stationary_frames
        self.recheck_interval = recheck_interval
        self.expire_seconds = expire_seconds

        self._objects: Dict[int, TrackedObject] = {}

        # 统计
        self.total_skipped = 0
        self.total_rechecked = 0

    def update(self, detections: List[Detection]) -> None:
        """
        用最新检测结果更新跟踪状态。
        在每次推理完成后调用。
        """
        now = time.time()
        seen_ids = set()

        for det in detections:
            if det.track_id < 0:
                continue
            seen_ids.add(det.track_id)
            cx, cy = det.center

            if det.track_id in self._objects:
                obj = self._objects[det.track_id]
                dx = cx - obj.last_center[0]
                dy = cy - obj.last_center[1]
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < self.move_threshold:
                    obj.still_frames += 1
                    if obj.still_frames >= self.stationary_frames:
                        obj.is_stationary = True
                else:
                    obj.still_frames = 0
                    obj.is_stationary = False
                    obj.frames_since_recheck = 0

                obj.last_center = (cx, cy)
                obj.detection = det
                obj.last_update_time = now
                obj.frames_since_recheck += 1
            else:
                self._objects[det.track_id] = TrackedObject(
                    track_id=det.track_id,
                    detection=det,
                    last_center=(cx, cy),
                    last_update_time=now,
                )

        # 清理消失的目标
        expired = [tid for tid, obj in self._objects.items()
                   if tid not in seen_ids and (now - obj.last_update_time) > self.expire_seconds]
        for tid in expired:
            del self._objects[tid]

    def get_stationary_detections(self, motion_boxes: List[MotionBox]
                                  ) -> Tuple[List[Detection], bool]:
        """
        获取可以跳过推理的静止目标检测结果。

        返回:
            (stationary_detections, need_full_detect)
            - stationary_detections: 可复用的静止目标检测结果
            - need_full_detect: 是否仍需要运行完整推理
              (有非静止目标、或有静止目标需要 recheck)
        """
        stationary_dets = []
        need_full = False

        for obj in self._objects.values():
            if not obj.is_stationary:
                need_full = True
                continue

            # 检查运动区域是否与静止目标重叠 → 强制重新检测
            if self._overlaps_motion(obj, motion_boxes):
                obj.is_stationary = False
                obj.still_frames = 0
                obj.frames_since_recheck = 0
                need_full = True
                continue

            # 定期 recheck
            if obj.frames_since_recheck >= self.recheck_interval:
                obj.frames_since_recheck = 0
                self.total_rechecked += 1
                need_full = True
                continue

            # 可以复用
            stationary_dets.append(obj.detection)
            self.total_skipped += 1

        # 如果没有任何已跟踪目标，也需要完整检测（可能有新目标）
        if not self._objects:
            need_full = True

        return stationary_dets, need_full

    def _overlaps_motion(self, obj: TrackedObject,
                         motion_boxes: List[MotionBox]) -> bool:
        """检查静止目标是否与运动区域重叠"""
        det = obj.detection
        dx1, dy1, dx2, dy2 = det.bbox

        for mb in motion_boxes:
            if (dx1 <= mb.x2 and mb.x1 <= dx2 and
                    dy1 <= mb.y2 and mb.y1 <= dy2):
                return True
        return False

    @property
    def stationary_count(self) -> int:
        return sum(1 for o in self._objects.values() if o.is_stationary)

    @property
    def tracked_count(self) -> int:
        return len(self._objects)
