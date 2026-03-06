"""
运动检测模块 — 参考 Frigate ImprovedMotionDetector

核心优化（对标 Frigate）:
1. 接收 YUV420P 帧，直接取 Y 通道（零开销灰度）
2. 缩小到 frame_height（默认 100px）后再做运动检测，像素量减少 20~50 倍
3. 用 INTER_NEAREST 最快插值
4. scipy gaussian_filter(sigma=1, radius=1) 替代 OpenCV GaussianBlur(11,11)
5. 运动框坐标映射回原始分辨率

参考: frigate/frigate/motion/improved_motion.py
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

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
    基于背景差分的运动检测器 — 参考 Frigate ImprovedMotionDetector。

    关键优化:
    - 在缩小帧（默认 100px 高）上做所有运算
    - 直接从 YUV 取 Y 通道，不做颜色转换
    - INTER_NEAREST 缩放 + 轻量高斯模糊
    """

    def __init__(self, config: MotionConfig = None):
        cfg = config or MotionConfig()
        self.enabled = cfg.enabled
        self.threshold = cfg.threshold
        self.contour_area = cfg.contour_area
        self.frame_alpha = cfg.frame_alpha
        self.frame_height = cfg.frame_height  # Frigate 默认 100

        self._bg_model: Optional[np.ndarray] = None
        self._frame_count = 0
        self._calibrating = True

        # 缩放因子（在第一帧时计算）
        self._resize_factor: float = 1.0
        self._motion_frame_size: Optional[Tuple[int, int]] = None  # (h, w)

        # 运动遮罩（在缩小帧上）
        self._mask: Optional[np.ndarray] = None
        self._mask_polygons = cfg.mask

    def _init_sizes(self, orig_h: int, orig_w: int):
        """根据原始帧尺寸计算缩放参数 — 对标 Frigate __init__"""
        self._resize_factor = orig_h / self.frame_height
        motion_w = int(orig_w / self._resize_factor)
        self._motion_frame_size = (self.frame_height, motion_w)
        logger.info(
            f"运动检测帧: {orig_w}x{orig_h} → {motion_w}x{self.frame_height} "
            f"(缩放因子 {self._resize_factor:.1f}x)"
        )

    def _init_mask(self):
        """在缩小帧上初始化运动遮罩"""
        if not self._mask_polygons or self._motion_frame_size is None:
            self._mask = None
            return
        h, w = self._motion_frame_size
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self._mask_polygons:
            # 将原始坐标缩放到运动检测帧
            pts = np.array(poly, dtype=np.float32).reshape((-1, 2))
            pts = pts / self._resize_factor
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
        # mask=True 的区域会被置零（不检测运动）
        self._mask = mask.astype(bool)

    def _extract_gray(self, frame: np.ndarray) -> np.ndarray:
        """
        从帧中提取灰度图。
        - YUV420P 帧: 直接取 Y 通道（前 2/3 高度），零开销
        - BGR 帧: cvtColor 转换（兼容回退）
        """
        if frame.ndim == 2:
            # 已经是灰度
            return frame

        h = frame.shape[0]
        w = frame.shape[1]

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            return frame.squeeze()

        if frame.ndim == 3 and frame.shape[2] == 3:
            # BGR 帧（兼容回退）
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 不应该到这里
        return frame

    def detect(self, frame: np.ndarray,
               frame_height: int = 0,
               frame_width: int = 0) -> Tuple[List[MotionBox], bool]:
        """
        检测运动区域 — 参考 Frigate ImprovedMotionDetector.detect()。

        Args:
            frame: YUV420P 帧（h*3/2, w）或 BGR 帧（h, w, 3）
            frame_height: 原始帧高度（YUV 模式下需要传入）
            frame_width: 原始帧宽度

        返回 (motion_boxes, has_motion)，坐标为原始帧坐标。
        """
        if not self.enabled:
            # 运动检测禁用时，返回全帧作为运动区域
            if frame_height > 0:
                h, w = frame_height, frame_width
            elif frame.ndim == 3:
                h, w = frame.shape[:2]
            else:
                # YUV420P: 实际高度 = shape[0] * 2 / 3
                h = frame.shape[0] * 2 // 3
                w = frame.shape[1]
            return [MotionBox(0, 0, w, h, w * h)], True

        # 提取 Y 通道灰度 — 对标 Frigate: gray = frame[0:height, 0:width]
        if frame_height > 0 and frame.ndim == 2:
            # YUV420P 帧: 直接取 Y 通道（前 height 行）
            gray = frame[0:frame_height, 0:frame_width]
            orig_h, orig_w = frame_height, frame_width
        else:
            gray = self._extract_gray(frame)
            orig_h, orig_w = gray.shape[:2]

        # 首次初始化缩放参数
        if self._motion_frame_size is None:
            self._init_sizes(orig_h, orig_w)
            self._init_mask()

        # 缩小帧 — 对标 Frigate: cv2.resize(..., interpolation=INTER_NEAREST)
        resized = cv2.resize(
            gray,
            dsize=(self._motion_frame_size[1], self._motion_frame_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # 应用遮罩（在缩小帧上）— 对标 Frigate: resized_frame[self.mask] = [0]
        if self._mask is not None:
            resized[self._mask] = 0

        # 轻量高斯模糊 — 对标 Frigate: gaussian_filter(sigma=1, radius=1)
        resized = gaussian_filter(resized, sigma=1, radius=1)

        # 初始化背景模型
        if self._bg_model is None:
            self._bg_model = resized.astype(np.float32)
            self._frame_count = 1
            return [], False

        # 帧差 — 对标 Frigate: cv2.absdiff + cv2.convertScaleAbs
        frame_delta = cv2.absdiff(resized, cv2.convertScaleAbs(self._bg_model))

        # 更新背景模型（指数移动平均）
        cv2.accumulateWeighted(resized, self._bg_model, self.frame_alpha)
        self._frame_count += 1

        # 需要几帧来建立稳定背景
        if self._frame_count < 10:
            self._calibrating = True
            return [], False
        self._calibrating = False

        # 二值化 + 膨胀 — 对标 Frigate
        thresh = cv2.threshold(
            frame_delta, self.threshold, 255, cv2.THRESH_BINARY
        )[1]
        thresh = cv2.dilate(thresh, None, iterations=1)

        # 查找轮廓
        contours = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # OpenCV 版本兼容
        contours = contours[0] if len(contours) == 2 else contours[1]

        boxes = []
        rf = self._resize_factor
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.contour_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # 映射回原始帧坐标 — 对标 Frigate: int(x * self.resize_factor)
            boxes.append(MotionBox(
                int(x * rf), int(y * rf),
                int((x + bw) * rf), int((y + bh) * rf),
                area,
            ))

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
