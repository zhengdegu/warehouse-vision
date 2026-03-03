"""
行为异常检测模块（规则型）
实现：dwell（滞留/徘徊）、crowd（聚集）、proximity（人车过近）、
      fight（打架）、fall（跌倒）
扩展：wrong_way（逆行）、speed（超速）
所有规则均支持 confirm_frames + cooldown 防抖 + 时间周期约束。
"""

import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from ..vision.detector import Detection
from .geometry import point_in_polygon, Polygon

logger = logging.getLogger(__name__)


class BaseAnomalyRule:
    """异常规则基类"""

    def __init__(self, rule_name: str, confirm_frames: int = 5,
                 cooldown: float = 60.0):
        self.rule_name = rule_name
        self.confirm_frames = confirm_frames
        self.cooldown = cooldown
        self._confirm_count: Dict[str, int] = {}
        self._last_trigger: Dict[str, float] = {}

    def _check_confirm_and_cooldown(self, key: str, condition: bool,
                                     now: float = 0.0) -> bool:
        """通用的确认帧 + 冷却检查"""
        if now <= 0:
            now = time.time()
        if condition:
            self._confirm_count[key] = self._confirm_count.get(key, 0) + 1
        else:
            self._confirm_count[key] = 0
            return False

        if self._confirm_count[key] < self.confirm_frames:
            return False

        last = self._last_trigger.get(key, 0)
        if now - last < self.cooldown:
            return False

        self._last_trigger[key] = now
        return True

    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DwellRule(BaseAnomalyRule):
    """滞留检测：目标在某区域停留超过阈值时间"""

    def __init__(self, max_seconds: float = 120.0,
                 confirm_frames: int = 5, cooldown: float = 60.0):
        super().__init__("dwell", confirm_frames, cooldown)
        self.max_seconds = max_seconds
        # track_id -> 首次出现时间
        self._first_seen: Dict[int, float] = {}

    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        events = []
        now = frame_ts if frame_ts > 0 else time.time()
        active_ids = set()

        for det in detections:
            if det.track_id < 0:
                continue
            active_ids.add(det.track_id)

            if det.track_id not in self._first_seen:
                self._first_seen[det.track_id] = now

            dwell_time = now - self._first_seen[det.track_id]
            is_dwelling = dwell_time >= self.max_seconds

            key = f"dwell_{det.track_id}"
            if self._check_confirm_and_cooldown(key, is_dwelling, now=now):
                events.append({
                    "type": "anomaly",
                    "sub_type": "dwell",
                    "camera_id": camera_id,
                    "track_id": det.track_id,
                    "class_name": det.class_name,
                    "dwell_seconds": round(dwell_time, 1),
                    "bbox": det.bbox,
                    "timestamp": now,
                })
                logger.info(f"[滞留] cam={camera_id} track={det.track_id} "
                            f"dwell={dwell_time:.1f}s")

        # 清理消失的 track
        for tid in list(self._first_seen.keys()):
            if tid not in active_ids:
                del self._first_seen[tid]

        return events


class CrowdRule(BaseAnomalyRule):
    """聚集检测：某区域内目标数量超过阈值"""

    def __init__(self, max_count: int = 5, radius: float = 200.0,
                 confirm_frames: int = 5, cooldown: float = 60.0):
        super().__init__("crowd", confirm_frames, cooldown)
        self.max_count = max_count
        self.radius = radius

    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        events = []
        now = frame_ts if frame_ts > 0 else time.time()

        # 计算所有人的聚集情况
        person_dets = [d for d in detections
                       if d.track_id >= 0 and d.class_name == "person"]

        if len(person_dets) <= self.max_count:
            self._confirm_count.clear()
            return events

        # 简单方法：检查是否有足够多的人在彼此的 radius 范围内
        # 使用聚类思想：找到密集区域
        for i, det_i in enumerate(person_dets):
            nearby = 1
            for j, det_j in enumerate(person_dets):
                if i == j:
                    continue
                dist = math.dist(det_i.center, det_j.center)
                if dist < self.radius:
                    nearby += 1

            is_crowded = nearby > self.max_count
            key = f"crowd_{det_i.track_id}"

            if self._check_confirm_and_cooldown(key, is_crowded, now=now):
                events.append({
                    "type": "anomaly",
                    "sub_type": "crowd",
                    "camera_id": camera_id,
                    "count": nearby,
                    "center": det_i.center,
                    "timestamp": now,
                })
                logger.info(f"[聚集] cam={camera_id} count={nearby}")
                break  # 一帧只报一次

        return events


class ProximityRule(BaseAnomalyRule):
    """人车过近检测"""

    def __init__(self, min_distance: float = 50.0,
                 confirm_frames: int = 3, cooldown: float = 30.0):
        super().__init__("proximity", confirm_frames, cooldown)
        self.min_distance = min_distance
        self._person_classes = {"person"}
        self._vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}

    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        events = []
        now = frame_ts if frame_ts > 0 else time.time()

        persons = [d for d in detections
                   if d.track_id >= 0 and d.class_name in self._person_classes]
        vehicles = [d for d in detections
                    if d.track_id >= 0 and d.class_name in self._vehicle_classes]

        for p in persons:
            for v in vehicles:
                dist = math.dist(p.foot, v.foot)
                is_close = dist < self.min_distance

                key = f"prox_{p.track_id}_{v.track_id}"
                if self._check_confirm_and_cooldown(key, is_close, now=now):
                    events.append({
                        "type": "anomaly",
                        "sub_type": "proximity",
                        "camera_id": camera_id,
                        "person_track_id": p.track_id,
                        "vehicle_track_id": v.track_id,
                        "vehicle_class": v.class_name,
                        "distance": round(dist, 1),
                        "timestamp": now,
                    })
                    logger.info(f"[过近] cam={camera_id} "
                                f"person={p.track_id} vehicle={v.track_id} "
                                f"dist={dist:.1f}")

        return events


class WrongWayRule(BaseAnomalyRule):
    """逆行检测（扩展接口，预留实现）"""

    def __init__(self, expected_direction: str = "positive",
                 confirm_frames: int = 5, cooldown: float = 60.0):
        super().__init__("wrong_way", confirm_frames, cooldown)
        self.expected_direction = expected_direction
        self._prev_positions: Dict[int, tuple] = {}

    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        # 扩展接口：根据预期方向判断是否逆行
        # 需要配合具体场景的方向定义
        events = []
        now = frame_ts if frame_ts > 0 else time.time()

        for det in detections:
            if det.track_id < 0:
                continue
            prev = self._prev_positions.get(det.track_id)
            self._prev_positions[det.track_id] = det.center

            if prev is None:
                continue

            # 计算运动方向（简单用 x 轴方向）
            dx = det.center[0] - prev[0]
            if self.expected_direction == "positive":
                is_wrong = dx < -5  # 向左运动视为逆行
            else:
                is_wrong = dx > 5

            key = f"wrongway_{det.track_id}"
            if self._check_confirm_and_cooldown(key, is_wrong, now=now):
                events.append({
                    "type": "anomaly",
                    "sub_type": "wrong_way",
                    "camera_id": camera_id,
                    "track_id": det.track_id,
                    "class_name": det.class_name,
                    "bbox": det.bbox,
                    "timestamp": now,
                })

        return events


class SpeedRule(BaseAnomalyRule):
    """超速检测（扩展接口，预留实现）"""

    def __init__(self, max_pixel_speed: float = 100.0,
                 confirm_frames: int = 3, cooldown: float = 30.0):
        super().__init__("speed", confirm_frames, cooldown)
        self.max_pixel_speed = max_pixel_speed
        self._prev_positions: Dict[int, tuple] = {}
        self._prev_times: Dict[int, float] = {}

    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        events = []
        now = frame_ts if frame_ts > 0 else time.time()

        for det in detections:
            if det.track_id < 0:
                continue

            prev_pos = self._prev_positions.get(det.track_id)
            prev_time = self._prev_times.get(det.track_id)
            self._prev_positions[det.track_id] = det.center
            self._prev_times[det.track_id] = now

            if prev_pos is None or prev_time is None:
                continue

            dt = now - prev_time
            if dt <= 0:
                continue

            dist = math.dist(det.center, prev_pos)
            speed = dist / dt  # 像素/秒

            is_fast = speed > self.max_pixel_speed
            key = f"speed_{det.track_id}"
            if self._check_confirm_and_cooldown(key, is_fast, now=now):
                events.append({
                    "type": "anomaly",
                    "sub_type": "speed",
                    "camera_id": camera_id,
                    "track_id": det.track_id,
                    "class_name": det.class_name,
                    "pixel_speed": round(speed, 1),
                    "bbox": det.bbox,
                    "timestamp": now,
                })

        return events


class FightRule(BaseAnomalyRule):
    """
    打架检测（Pose 增强版）：
    - 基础判断：多人近距离 + 高速运动
    - Pose 增强：手腕/肘部运动速度（挥拳动作）大幅提升精度
    """

    def __init__(self, proximity_radius: float = 150.0,
                 min_speed: float = 60.0,
                 min_persons: int = 2,
                 confirm_frames: int = 3,
                 cooldown: float = 30.0):
        super().__init__("fight", confirm_frames, cooldown)
        self.proximity_radius = proximity_radius
        self.min_speed = min_speed  # 像素/秒
        self.min_persons = min_persons
        self._prev_positions: Dict[int, tuple] = {}
        self._prev_times: Dict[int, float] = {}
        # Pose: 上一帧手腕位置
        self._prev_wrists: Dict[int, list] = {}

    def _calc_limb_speed(self, det: 'Detection', now: float) -> float:
        """计算手腕/肘部的运动速度（Pose 增强）"""
        if det.keypoints is None or det.track_id < 0:
            return 0.0

        kp = det.keypoints
        # 取左右手腕 (index 9, 10)，置信度 > 0.3 才用
        wrists = []
        for idx in [9, 10]:
            if kp[idx][2] > 0.3:
                wrists.append((float(kp[idx][0]), float(kp[idx][1])))

        if not wrists:
            return 0.0

        prev = self._prev_wrists.get(det.track_id)
        self._prev_wrists[det.track_id] = wrists

        if prev is None:
            return 0.0

        # 计算手腕最大位移速度
        prev_time = self._prev_times.get(det.track_id, now)
        dt = now - prev_time
        if dt <= 0:
            return 0.0

        max_speed = 0.0
        for w in wrists:
            for pw in prev:
                speed = math.dist(w, pw) / dt
                max_speed = max(max_speed, speed)

        return max_speed

    def update(self, detections: List['Detection'],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        events = []
        now = frame_ts if frame_ts > 0 else time.time()

        person_dets = [d for d in detections
                       if d.track_id >= 0 and d.class_name == "person"]

        # 计算每个人的速度（中心点 + 手腕）
        speeds: Dict[int, float] = {}
        limb_speeds: Dict[int, float] = {}
        for det in person_dets:
            prev_pos = self._prev_positions.get(det.track_id)
            prev_time = self._prev_times.get(det.track_id)
            self._prev_positions[det.track_id] = det.center

            limb_spd = self._calc_limb_speed(det, now)
            limb_speeds[det.track_id] = limb_spd

            self._prev_times[det.track_id] = now

            if prev_pos is not None and prev_time is not None:
                dt = now - prev_time
                if dt > 0:
                    speeds[det.track_id] = math.dist(det.center, prev_pos) / dt

        # 清理已消失的 track
        active_ids = {d.track_id for d in person_dets}
        for tid in list(self._prev_positions.keys()):
            if tid not in active_ids:
                self._prev_positions.pop(tid, None)
                self._prev_times.pop(tid, None)
                self._prev_wrists.pop(tid, None)

        if len(person_dets) < self.min_persons:
            self._confirm_count.clear()
            return events

        for i, det_i in enumerate(person_dets):
            nearby_fast = []
            speed_i = speeds.get(det_i.track_id, 0)
            limb_i = limb_speeds.get(det_i.track_id, 0)
            # Pose 增强：手腕速度权重更高（挥拳特征）
            effective_speed_i = max(speed_i, limb_i * 0.6)

            for j, det_j in enumerate(person_dets):
                if i == j:
                    continue
                dist = math.dist(det_i.center, det_j.center)
                speed_j = speeds.get(det_j.track_id, 0)
                limb_j = limb_speeds.get(det_j.track_id, 0)
                effective_speed_j = max(speed_j, limb_j * 0.6)
                if dist < self.proximity_radius and (effective_speed_i > self.min_speed or effective_speed_j > self.min_speed):
                    nearby_fast.append(det_j.track_id)

            is_fight = len(nearby_fast) >= (self.min_persons - 1) and effective_speed_i > self.min_speed
            key = f"fight_{det_i.track_id}"

            if self._check_confirm_and_cooldown(key, is_fight, now=now):
                involved = [det_i.track_id] + nearby_fast
                avg_speed = sum(speeds.get(t, 0) for t in involved) / len(involved)
                has_pose = any(limb_speeds.get(t, 0) > 0 for t in involved)
                events.append({
                    "type": "anomaly",
                    "sub_type": "fight",
                    "camera_id": camera_id,
                    "track_id": det_i.track_id,
                    "class_name": "person",
                    "involved_count": len(involved),
                    "avg_speed": round(avg_speed, 1),
                    "bbox": det_i.bbox,
                    "detail": f"疑似打架：{len(involved)}人近距离剧烈运动，平均速度{avg_speed:.0f}px/s" + (" [Pose增强]" if has_pose else ""),
                    "timestamp": now,
                })
                logger.info(f"[打架] cam={camera_id} involved={len(involved)} speed={avg_speed:.0f} pose={'Y' if has_pose else 'N'}")
                break

        return events


class FallRule(BaseAnomalyRule):
    """
    跌倒检测（Pose 增强版）：
    - 基础判断：bbox 宽高比突变 + 中心点下移
    - Pose 增强：头部关键点低于臀部 → 跌倒（精度大幅提升）
    """

    def __init__(self, ratio_threshold: float = 1.0,
                 min_ratio_change: float = 0.5,
                 min_y_drop: float = 20.0,
                 confirm_frames: int = 2,
                 cooldown: float = 30.0):
        super().__init__("fall", confirm_frames, cooldown)
        self.ratio_threshold = ratio_threshold
        self.min_ratio_change = min_ratio_change
        self.min_y_drop = min_y_drop
        self._prev_ratios: Dict[int, float] = {}
        self._prev_centers: Dict[int, tuple] = {}

    def _pose_is_fallen(self, kp: 'np.ndarray') -> bool:
        """
        通过关键点判断是否跌倒：
        - 头部(nose/eyes) Y 坐标 > 臀部(hips) Y 坐标 → 倒地
        - 肩膀连线接近水平且躯干角度异常
        """
        import numpy as _np

        # 头部: nose(0), left_eye(1), right_eye(2)
        head_pts = []
        for idx in [0, 1, 2]:
            if kp[idx][2] > 0.3:
                head_pts.append(kp[idx][:2])

        # 臀部: left_hip(11), right_hip(12)
        hip_pts = []
        for idx in [11, 12]:
            if kp[idx][2] > 0.3:
                hip_pts.append(kp[idx][:2])

        if not head_pts or not hip_pts:
            return False

        head_y = _np.mean([p[1] for p in head_pts])
        hip_y = _np.mean([p[1] for p in hip_pts])

        # 头部 Y > 臀部 Y（图像坐标系，Y 向下）→ 头低于臀 → 跌倒
        if head_y > hip_y:
            return True

        # 额外检查：肩膀到踝部的躯干角度
        shoulder_pts = []
        for idx in [5, 6]:
            if kp[idx][2] > 0.3:
                shoulder_pts.append(kp[idx][:2])
        ankle_pts = []
        for idx in [15, 16]:
            if kp[idx][2] > 0.3:
                ankle_pts.append(kp[idx][:2])

        if shoulder_pts and ankle_pts:
            shoulder_y = _np.mean([p[1] for p in shoulder_pts])
            ankle_y = _np.mean([p[1] for p in ankle_pts])
            shoulder_x = _np.mean([p[0] for p in shoulder_pts])
            ankle_x = _np.mean([p[0] for p in ankle_pts])
            dy = abs(ankle_y - shoulder_y)
            dx = abs(ankle_x - shoulder_x)
            # 躯干接近水平（dx > dy）→ 倒地
            if dx > 0 and dy / dx < 0.5:
                return True

        return False

    def update(self, detections: List['Detection'],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        events = []
        now = frame_ts if frame_ts > 0 else time.time()

        person_dets = [d for d in detections
                       if d.track_id >= 0 and d.class_name == "person"]

        active_ids = {d.track_id for d in person_dets}
        for tid in list(self._prev_ratios.keys()):
            if tid not in active_ids:
                self._prev_ratios.pop(tid, None)
                self._prev_centers.pop(tid, None)

        for det in person_dets:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            if h <= 0:
                continue
            ratio = w / h

            prev_ratio = self._prev_ratios.get(det.track_id)
            prev_center = self._prev_centers.get(det.track_id)
            self._prev_ratios[det.track_id] = ratio
            self._prev_centers[det.track_id] = det.center

            # Pose 增强判断
            pose_fallen = False
            if det.keypoints is not None:
                pose_fallen = self._pose_is_fallen(det.keypoints)

            if pose_fallen:
                # Pose 直接判定跌倒，不需要历史帧对比
                is_fall = True
                detail = f"疑似跌倒：姿态异常(头低于臀) [Pose增强]"
            elif prev_ratio is not None and prev_center is not None:
                # 回退到 bbox 规则
                ratio_change = ratio - prev_ratio
                y_drop = det.center[1] - prev_center[1]
                is_fall = (ratio > self.ratio_threshold
                           and ratio_change > self.min_ratio_change
                           and y_drop > self.min_y_drop)
                detail = f"疑似跌倒：宽高比{ratio:.2f}(变化+{ratio_change:.2f})，下移{y_drop:.0f}px"
            else:
                continue

            key = f"fall_{det.track_id}"
            if self._check_confirm_and_cooldown(key, is_fall, now=now):
                events.append({
                    "type": "anomaly",
                    "sub_type": "fall",
                    "camera_id": camera_id,
                    "track_id": det.track_id,
                    "class_name": "person",
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "detail": detail,
                    "timestamp": now,
                })
                logger.info(f"[跌倒] cam={camera_id} track={det.track_id} pose={'Y' if pose_fallen else 'N'}")

        return events


class AnomalyEngine:
    """异常检测引擎，聚合所有规则，支持时间周期约束"""

    def __init__(self, config: dict, roi: Polygon = None):
        self.rules: List[BaseAnomalyRule] = []
        # ROI 多边形，设置后只检测区域内的目标
        self.roi = [(float(p[0]), float(p[1])) for p in roi] if roi else None

        # 时间周期约束
        tp_cfg = config.get("time_period", {})
        self._time_period_enabled = tp_cfg.get("enabled", False)
        self._time_period_start = tp_cfg.get("start", "00:00")
        self._time_period_end = tp_cfg.get("end", "23:59")
        self._time_period_days = tp_cfg.get("days", [0, 1, 2, 3, 4, 5, 6])

        # dwell
        dwell_cfg = config.get("dwell", {})
        if dwell_cfg.get("enabled", False):
            self.rules.append(DwellRule(
                max_seconds=dwell_cfg.get("max_seconds", 120),
                confirm_frames=dwell_cfg.get("confirm_frames", 5),
                cooldown=dwell_cfg.get("cooldown", 60),
            ))

        # crowd
        crowd_cfg = config.get("crowd", {})
        if crowd_cfg.get("enabled", False):
            self.rules.append(CrowdRule(
                max_count=crowd_cfg.get("max_count", 5),
                radius=crowd_cfg.get("radius", 200),
                confirm_frames=crowd_cfg.get("confirm_frames", 5),
                cooldown=crowd_cfg.get("cooldown", 60),
            ))

        # proximity
        prox_cfg = config.get("proximity", {})
        if prox_cfg.get("enabled", False):
            self.rules.append(ProximityRule(
                min_distance=prox_cfg.get("min_distance", 50),
                confirm_frames=prox_cfg.get("confirm_frames", 3),
                cooldown=prox_cfg.get("cooldown", 30),
            ))

        # wrong_way (扩展)
        ww_cfg = config.get("wrong_way", {})
        if ww_cfg.get("enabled", False):
            self.rules.append(WrongWayRule(
                expected_direction=ww_cfg.get("expected_direction", "positive"),
                confirm_frames=ww_cfg.get("confirm_frames", 5),
                cooldown=ww_cfg.get("cooldown", 60),
            ))

        # speed (扩展)
        spd_cfg = config.get("speed", {})
        if spd_cfg.get("enabled", False):
            self.rules.append(SpeedRule(
                max_pixel_speed=spd_cfg.get("max_pixel_speed", 100),
                confirm_frames=spd_cfg.get("confirm_frames", 3),
                cooldown=spd_cfg.get("cooldown", 30),
            ))

        # fight (打架检测)
        fight_cfg = config.get("fight", {})
        if fight_cfg.get("enabled", False):
            self.rules.append(FightRule(
                proximity_radius=fight_cfg.get("proximity_radius", 150),
                min_speed=fight_cfg.get("min_speed", 60),
                min_persons=fight_cfg.get("min_persons", 2),
                confirm_frames=fight_cfg.get("confirm_frames", 3),
                cooldown=fight_cfg.get("cooldown", 30),
            ))

        # fall (跌倒检测)
        fall_cfg = config.get("fall", {})
        if fall_cfg.get("enabled", False):
            self.rules.append(FallRule(
                ratio_threshold=fall_cfg.get("ratio_threshold", 1.0),
                min_ratio_change=fall_cfg.get("min_ratio_change", 0.5),
                min_y_drop=fall_cfg.get("min_y_drop", 20),
                confirm_frames=fall_cfg.get("confirm_frames", 2),
                cooldown=fall_cfg.get("cooldown", 30),
            ))

    def _is_in_active_period(self) -> bool:
        """检查当前时间是否在激活时段内"""
        if not self._time_period_enabled:
            return True  # 未启用时间约束 → 全天候运行

        now = datetime.now()
        # 检查星期几（0=周一 ... 6=周日）
        if now.weekday() not in self._time_period_days:
            return False

        # 检查时间段
        current = now.strftime("%H:%M")
        start = self._time_period_start
        end = self._time_period_end

        if start <= end:
            # 正常时段，如 08:00 ~ 18:00
            return start <= current <= end
        else:
            # 跨午夜时段，如 22:00 ~ 06:00
            return current >= start or current <= end

    def update(self, detections: List[Detection],
               camera_id: str = "",
               frame_ts: float = 0.0) -> List[Dict[str, Any]]:
        """运行所有规则，返回异常事件列表（仅处理 ROI 内的目标，受时间约束）"""
        # 时间周期约束检查
        if not self._is_in_active_period():
            return []

        # ROI 过滤
        if self.roi:
            detections = [d for d in detections
                          if d.track_id < 0 or point_in_polygon(d.foot, self.roi)]
        all_events = []
        for rule in self.rules:
            try:
                events = rule.update(detections, camera_id,
                                     frame_ts=frame_ts)
                all_events.extend(events)
            except Exception as e:
                logger.error(f"规则 {rule.rule_name} 异常: {e}")
        return all_events
