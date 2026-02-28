"""
越线检测模块
判断目标轨迹是否穿越 Tripwire 线段，支持方向判定。

三层检测策略（由精确到宽松）:
1. 逐帧跟踪: 有 track_id 时，前一帧 foot → 当前 foot
2. 历史回溯: 有 track_id 时，首次出现位置 → 当前位置
3. 侧边翻转: 无 track_id 时，用 bbox IoU 匹配 + 线段侧判断
   解决快速目标 ByteTrack 来不及分配 ID 的问题
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from ..vision.detector import Detection
from .geometry import line_crossing, Point, bbox_center

logger = logging.getLogger(__name__)

_MAX_HISTORY = 30
_TRACK_EXPIRE = 10.0

# COCO 车辆类别 — 这些目标用 center 而非 foot 做越线判定
_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def _side_of_line(point: Point, lp1: Point, lp2: Point) -> int:
    """判断点在线段的哪一侧。返回 +1, -1, 0"""
    cross = ((lp2[0] - lp1[0]) * (point[1] - lp1[1]) -
             (lp2[1] - lp1[1]) * (point[0] - lp1[0]))
    if cross > 0:
        return 1
    elif cross < 0:
        return -1
    return 0


def _bbox_iou(b1: list, b2: list) -> float:
    """计算两个 bbox 的 IoU"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def _bbox_distance(b1: list, b2: list) -> float:
    """两个 bbox 中心点距离"""
    cx1 = (b1[0] + b1[2]) / 2
    cy1 = (b1[1] + b1[3]) / 2
    cx2 = (b2[0] + b2[2]) / 2
    cy2 = (b2[1] + b2[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


class _UntrackEntry:
    """无 track_id 的检测记录"""
    __slots__ = ('bbox', 'foot', 'center', 'class_id', 'class_name',
                 'side', 'center_side', 'timestamp')

    def __init__(self, det: Detection, side: int, center_side: int, ts: float):
        self.bbox = list(det.bbox)
        self.foot = det.foot
        self.center = det.center
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.side = side
        self.center_side = center_side
        self.timestamp = ts



class TripwireDetector:
    """越线检测器 — 三层策略，支持快速移动目标"""

    def __init__(self, tripwire_id: str, name: str,
                 p1: Tuple[float, float], p2: Tuple[float, float],
                 direction: str = "left_to_right",
                 cooldown: float = 2.0):
        self.tripwire_id = tripwire_id
        self.name = name
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.direction = direction
        self.cooldown = cooldown

        # === 有 track_id 的跟踪数据 ===
        self._track_history: Dict[int, deque] = {}
        self._last_trigger: Dict[int, float] = {}
        self._prev_foot: Dict[int, Point] = {}
        self._prev_center: Dict[int, Point] = {}

        # === 无 track_id 的侧边翻转检测 ===
        self._untracked_prev: List[_UntrackEntry] = []
        # 按类别冷却：防止同类目标被无跟踪策略重复触发，但不阻塞其他类别
        self._untracked_cooldown_by_class: Dict[str, float] = {}
        # 无跟踪策略使用更短的冷却时间
        self._untracked_cooldown_time = min(cooldown, 0.8)

        # 侧边翻转防抖：记录每个 untracked 目标的稳定侧边，
        # 需要连续 N 帧在同一侧才算稳定，避免 bbox 抖动导致误触发
        self._untracked_stable_side: Dict[int, Tuple[int, int]] = {}  # class_id -> (side, consecutive_count)
        self._STABLE_FRAMES_REQUIRED = 3  # 需要连续 3 帧在同一侧才算稳定

        self._last_cleanup = time.time()
        self._last_debug = 0.0

    def _cleanup_stale(self, now: float):
        if now - self._last_cleanup < 5.0:
            return
        self._last_cleanup = now
        stale = [tid for tid, hist in self._track_history.items()
                 if hist and now - hist[-1][2] > _TRACK_EXPIRE]
        for tid in stale:
            self._track_history.pop(tid, None)
            self._prev_foot.pop(tid, None)
            self._prev_center.pop(tid, None)
            self._last_trigger.pop(tid, None)
        # 清理过期的无跟踪记录
        self._untracked_prev = [
            e for e in self._untracked_prev if now - e.timestamp < 3.0
        ]

    def _make_event(self, camera_id: str, det: Detection,
                    cross_dir: str, now: float,
                    method: str = "") -> Dict[str, Any]:
        evt = {
            "type": "tripwire",
            "sub_type": cross_dir,
            "camera_id": camera_id,
            "tripwire_id": self.tripwire_id,
            "tripwire_name": self.name,
            "track_id": det.track_id,
            "class_name": det.class_name,
            "crossing_direction": cross_dir,
            "bbox": det.bbox,
            "timestamp": now,
        }
        logger.info(f"[越线] cam={camera_id} tw={self.name} "
                    f"track={det.track_id} cls={det.class_name} "
                    f"dir={cross_dir} method={method}")
        return evt

    def _crossing_to_dir(self, crossing: str) -> str:
        if self.direction == "left_to_right":
            return "in" if crossing == "positive" else "out"
        else:
            return "out" if crossing == "positive" else "in"

    def _side_flip_to_dir(self, old_side: int, new_side: int) -> Optional[str]:
        """侧边翻转 → 方向"""
        if old_side == new_side or old_side == 0 or new_side == 0:
            return None
        # old_side → new_side 等价于 crossing positive/negative
        # positive = 从 -1 侧到 +1 侧
        if old_side == -1 and new_side == 1:
            crossing = "positive"
        else:
            crossing = "negative"
        return self._crossing_to_dir(crossing)

    def update(self, detections: List[Detection],
               camera_id: str = "") -> List[Dict[str, Any]]:
        events = []
        now = time.time()
        self._cleanup_stale(now)

        # 调试日志
        if now - self._last_debug > 30 and len(detections) > 0:
            tracked = [d for d in detections if d.track_id >= 0]
            untracked = [d for d in detections if d.track_id < 0]
            logger.info(f"[TW调试] cam={camera_id} tw={self.name} "
                        f"总={len(detections)} tracked={len(tracked)} "
                        f"untracked={len(untracked)} "
                        f"prev_untracked={len(self._untracked_prev)}")
            self._last_debug = now

        tracked_dets = []
        untracked_dets = []
        for det in detections:
            if det.track_id >= 0:
                tracked_dets.append(det)
            else:
                untracked_dets.append(det)

        # ========== 策略 1 & 2: 有 track_id ==========
        for det in tracked_dets:
            tid = det.track_id
            curr_foot = det.foot
            curr_center = det.center
            # 车辆用 center，行人用 foot
            is_vehicle = det.class_name in _VEHICLE_CLASSES
            curr_ref = curr_center if is_vehicle else curr_foot

            if tid not in self._track_history:
                self._track_history[tid] = deque(maxlen=_MAX_HISTORY)
            self._track_history[tid].append((curr_foot, curr_center, now))

            last = self._last_trigger.get(tid, 0)
            if now - last < self.cooldown:
                self._prev_foot[tid] = curr_foot
                self._prev_center[tid] = curr_center
                # 在 cooldown 期间持续清空历史，防止 cooldown 结束后
                # 策略2用旧的起始点重新触发越线
                self._track_history[tid].clear()
                continue

            crossing = "none"

            # 策略1: 逐帧 — 车辆用 center，行人用 foot
            prev_ref = self._prev_center.get(tid) if is_vehicle else self._prev_foot.get(tid)
            if prev_ref is not None:
                crossing = line_crossing(prev_ref, curr_ref,
                                         self.p1, self.p2)
            # 如果 foot 没检测到，也试 center（兜底）
            if crossing == "none" and not is_vehicle:
                prev_center = self._prev_center.get(tid)
                if prev_center is not None:
                    crossing = line_crossing(prev_center, curr_center,
                                             self.p1, self.p2)

            # 策略2: 历史回溯
            if crossing == "none" and len(self._track_history[tid]) >= 2:
                first_foot, first_center, first_time = \
                    self._track_history[tid][0]
                span = now - first_time
                if 0.03 < span < 8.0:
                    first_ref = first_center if is_vehicle else first_foot
                    crossing = line_crossing(first_ref, curr_ref,
                                             self.p1, self.p2)
                    if crossing == "none":
                        crossing = line_crossing(first_center, curr_center,
                                                 self.p1, self.p2)

            self._prev_foot[tid] = curr_foot
            self._prev_center[tid] = curr_center

            if crossing != "none":
                cross_dir = self._crossing_to_dir(crossing)
                self._last_trigger[tid] = now
                self._track_history[tid].clear()
                # 重置 prev 位置到当前位置，防止 cooldown 后用旧位置重新触发
                self._prev_foot[tid] = curr_foot
                self._prev_center[tid] = curr_center
                # 触发 tracked 越线后，给同类别的 untracked 加冷却
                self._untracked_cooldown_by_class[det.class_name] = now + self.cooldown
                events.append(self._make_event(
                    camera_id, det, cross_dir, now, "tracked"))

        # ========== 策略 3: 无 track_id 的侧边翻转 ==========
        # 车辆速度太快，ByteTrack 来不及分配 ID
        # 用 bbox 相似度匹配 + 线段侧边判断
        # 改进：按类别冷却，允许多辆车同帧触发
        if untracked_dets:
            new_entries: List[_UntrackEntry] = []
            matched_prev: set = set()  # 已匹配的 prev entry 索引

            for det in untracked_dets:
                is_vehicle = det.class_name in _VEHICLE_CLASSES
                # 车辆用 center 判断侧边，行人用 foot
                ref_point = det.center if is_vehicle else det.foot
                side = _side_of_line(ref_point, self.p1, self.p2)
                center_side = _side_of_line(det.center, self.p1, self.p2)
                entry = _UntrackEntry(det, side, center_side, now)
                new_entries.append(entry)

                if side == 0:
                    continue

                # 按类别检查冷却
                cls_cd = self._untracked_cooldown_by_class.get(det.class_name, 0)
                if now < cls_cd:
                    continue

                # 在上一帧的无跟踪记录中找最佳匹配
                best_match = None
                best_match_idx = -1
                best_score = 0
                for idx, prev_e in enumerate(self._untracked_prev):
                    if idx in matched_prev:
                        continue
                    # 同类别优先
                    if prev_e.class_id != det.class_id:
                        continue
                    iou = _bbox_iou(prev_e.bbox, det.bbox)
                    if iou > best_score:
                        best_score = iou
                        best_match = prev_e
                        best_match_idx = idx

                # IoU 太低时用距离兜底（快速移动 bbox 可能不重叠）
                if best_match is None or best_score < 0.01:
                    best_dist = float('inf')
                    for idx, prev_e in enumerate(self._untracked_prev):
                        if idx in matched_prev:
                            continue
                        if prev_e.class_id != det.class_id:
                            continue
                        dist = _bbox_distance(prev_e.bbox, det.bbox)
                        # 允许较大位移（快速车辆）
                        max_dim = max(det.bbox[2] - det.bbox[0],
                                      det.bbox[3] - det.bbox[1], 1)
                        if dist < max_dim * 3 and dist < best_dist:
                            best_dist = dist
                            best_match = prev_e
                            best_match_idx = idx

                if best_match is not None:
                    # 车辆优先用 center_side 比较，行人用 foot side
                    prev_side = best_match.center_side if is_vehicle else best_match.side
                    if prev_side != 0:
                        cross_dir = self._side_flip_to_dir(prev_side, side)
                        if cross_dir is not None:
                            # 防抖：检查前后帧的 bbox 位移是否足够大
                            # 如果目标几乎没动（bbox 抖动），不算越线
                            disp = _bbox_distance(best_match.bbox, list(det.bbox))
                            max_dim = max(det.bbox[2] - det.bbox[0],
                                          det.bbox[3] - det.bbox[1], 1)
                            min_displacement = max_dim * 0.05  # 至少移动 bbox 尺寸的 5%
                            if disp < min_displacement:
                                continue  # bbox 抖动，跳过

                            # 按类别冷却，不阻塞其他类别
                            self._untracked_cooldown_by_class[det.class_name] = \
                                now + self._untracked_cooldown_time
                            events.append(self._make_event(
                                camera_id, det, cross_dir, now, "untracked"))
                            matched_prev.add(best_match_idx)
                            # 不再 break — 允许同帧多辆车触发

            self._untracked_prev = new_entries

        # 即使有 tracked 检测，也更新 untracked 记录
        # （tracked 的也记录侧边信息，作为下一帧 untracked 匹配的候选）
        if tracked_dets and not untracked_dets:
            new_entries = []
            for det in tracked_dets:
                is_vehicle = det.class_name in _VEHICLE_CLASSES
                ref_point = det.center if is_vehicle else det.foot
                side = _side_of_line(ref_point, self.p1, self.p2)
                center_side = _side_of_line(det.center, self.p1, self.p2)
                new_entries.append(_UntrackEntry(det, side, center_side, now))
            self._untracked_prev = new_entries

        return events
