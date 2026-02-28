"""
人车计数 / 流量统计模块
基于越线事件统计 in/out，按时间窗口聚合，输出 counts.jsonl。
使用滑动窗口确保任意时刻查询都能拿到最近 N 秒的统计。
"""

import time
import logging
from typing import Dict, Any, List
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class FlowCounter:
    """流量计数器，滑动窗口聚合"""

    def __init__(self, camera_id: str, window_seconds: int = 60):
        self.camera_id = camera_id
        self.window_seconds = window_seconds

        # 总计数
        self.total_in = 0
        self.total_out = 0

        # 滑动窗口：记录每个事件的 (timestamp, direction)
        self._recent_events: deque = deque()

        # 定期输出窗口聚合（用于持久化到 JSONL/DB）
        self._last_flush = time.time()
        self._flush_in = 0
        self._flush_out = 0
        self._flush_class_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0}
        )

    def _prune_old(self, now: float):
        """清理超出窗口的旧事件"""
        cutoff = now - self.window_seconds
        while self._recent_events and self._recent_events[0][0] < cutoff:
            self._recent_events.popleft()

    def update(self, tripwire_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据越线事件更新计数，返回窗口聚合结果（如果到了 flush 周期）。
        """
        now = time.time()

        for evt in tripwire_events:
            direction = evt.get("crossing_direction", "")
            cls_name = evt.get("class_name", "unknown")

            if direction == "in":
                self.total_in += 1
                self._flush_in += 1
                self._flush_class_counts[cls_name]["in"] += 1
                self._recent_events.append((now, "in"))
            elif direction == "out":
                self.total_out += 1
                self._flush_out += 1
                self._flush_class_counts[cls_name]["out"] += 1
                self._recent_events.append((now, "out"))

        self._prune_old(now)

        # 定期 flush（用于持久化记录）
        results = []
        if now - self._last_flush >= self.window_seconds:
            results.append({
                "type": "count_window",
                "camera_id": self.camera_id,
                "window_start": self._last_flush,
                "window_end": now,
                "window_in": self._flush_in,
                "window_out": self._flush_out,
                "total_in": self.total_in,
                "total_out": self.total_out,
                "by_class": dict(self._flush_class_counts),
                "timestamp": now,
            })
            self._last_flush = now
            self._flush_in = 0
            self._flush_out = 0
            self._flush_class_counts.clear()

        return results

    def get_current_counts(self) -> Dict[str, Any]:
        """获取当前计数快照（滑动窗口）"""
        now = time.time()
        self._prune_old(now)

        window_in = sum(1 for _, d in self._recent_events if d == "in")
        window_out = sum(1 for _, d in self._recent_events if d == "out")

        return {
            "camera_id": self.camera_id,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "window_in": window_in,
            "window_out": window_out,
        }
