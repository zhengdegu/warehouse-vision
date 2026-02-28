"""
运动检测模块 — Frigate 核心优化
只在有运动的区域运行 AI 检测，大幅降低 CPU/GPU 负载。

流程: 帧差法 → 轮廓提取 → 运动区域合并 → 返回运动框列表
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config.schema import MotionConfig

logger = logging.getLogger(__name__)


@dataclass
class MotionBox:
    """运动区域"""
    x1: int
    y1: int
    x2: int
    y2: int
    area: int = 0


class MotionDetector:
    """
    基于背景差分的运动检测器。
    参考 Frigate 的设计：低开销预过滤，只在运动区域跑目标检测。
    """

    def __init__(self, config: MotionConfig = None):
        cfg = config or MotionConfig()
        self.enabled = cfg.enabled
        self.threshold = cfg.threshold
        self.contour_area = cfg.contour_area
        self.frame_alpha = cfg.frame_alpha

        self._bg_model: Optional[np.ndarray] = None
        self._frame_count = 0
        # 运动遮罩（预计算）
        self._mask: Optional[np.ndarray] = None
        self._mask_polygons = cfg.mask

    def _init_mask(self, h: int, w: int):
        """初始化运动遮罩"""
        if not self._mask_polygons:
            self._mask = None
            return
        mask = np.ones((h, w), dtype=np.uint8) * 255
        for poly in self._mask_polygons:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 0)  # 遮罩区域设为 0
        self._mask = mask

    def detect(self, frame: np.ndarray) -> Tuple[List[MotionBox], bool]:
        """
        检测运动区域。
        返回 (motion_boxes, has_motion)
        """
        if not self.enabled:
            # 运动检测禁用时，返回全帧作为运动区域
            h, w = frame.shape[:2]
            return [MotionBox(0, 0, w, h, w * h)], True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        h, w = gray.shape[:2]

        # 初始化背景模型
        if self._bg_model is None:
            self._bg_model = gray.astype(np.float32)
            self._init_mask(h, w)
            self._frame_count = 1
            return [], False

        # 更新背景模型（指数移动平均）
        cv2.accumulateWeighted(gray, self._bg_model, self.frame_alpha)
        self._frame_count += 1

        # 需要几帧来建立稳定背景
        if self._frame_count < 10:
            return [], False

        # 计算帧差
        bg_uint8 = cv2.convertScaleAbs(self._bg_model)
        diff = cv2.absdiff(gray, bg_uint8)

        # 应用遮罩
        if self._mask is not None:
            diff = cv2.bitwise_and(diff, self._mask)

        # 二值化
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.contour_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append(MotionBox(x, y, x + bw, y + bh, area))

        # 合并重叠的运动框
        boxes = self._merge_boxes(boxes)

        return boxes, len(boxes) > 0

    def _merge_boxes(self, boxes: List[MotionBox],
                     margin: int = 50) -> List[MotionBox]:
        """合并相近的运动框"""
        if len(boxes) <= 1:
            return boxes

        merged = True
        while merged:
            merged = False
            new_boxes = []
            used = set()
            for i in range(len(boxes)):
                if i in used:
                    continue
                b1 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if j in used:
                        continue
                    b2 = boxes[j]
                    # 检查是否重叠或相近
                    if (b1.x1 - margin <= b2.x2 and b2.x1 - margin <= b1.x2 and
                            b1.y1 - margin <= b2.y2 and b2.y1 - margin <= b1.y2):
                        b1 = MotionBox(
                            min(b1.x1, b2.x1), min(b1.y1, b2.y1),
                            max(b1.x2, b2.x2), max(b1.y2, b2.y2),
                            b1.area + b2.area,
                        )
                        used.add(j)
                        merged = True
                new_boxes.append(b1)
                used.add(i)
            boxes = new_boxes

        return boxes
