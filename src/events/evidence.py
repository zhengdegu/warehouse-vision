"""
事件截图与证据保存模块
保存事件截图 JPG，可绘制 bbox / ROI / tripwire 叠加信息。
支持中文文字渲染（通过 PIL）。
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 中文字体 - 尝试常见 Windows/Linux 字体路径
_font_cache = {}

def _get_font(size=16):
    if size in _font_cache:
        return _font_cache[size]
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def put_chinese_text(img, text, pos, color=(255,255,255), size=16, bg_color=None):
    """在 OpenCV 图像上绘制中文文字"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(size)
    x, y = pos
    if bg_color:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill=bg_color)
    # color is BGR in OpenCV, convert to RGB for PIL
    rgb_color = (color[2], color[1], color[0])
    draw.text((x, y), text, font=font, fill=rgb_color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)


class EvidenceSaver:
    """事件截图保存器"""

    def __init__(self, output_dir: str = "events",
                 draw_bbox: bool = True,
                 draw_roi: bool = True,
                 draw_tripwire: bool = True):
        self.output_dir = output_dir
        self.draw_bbox = draw_bbox
        self.draw_roi = draw_roi
        self.draw_tripwire = draw_tripwire
        os.makedirs(output_dir, exist_ok=True)

    def save_screenshot(self, frame: np.ndarray, event: Dict[str, Any],
                        roi: Optional[List] = None,
                        tripwires: Optional[List] = None,
                        detections: Optional[List] = None) -> str:
        """
        保存事件截图，返回文件路径。
        """
        img = frame.copy()

        # 绘制 ROI
        if self.draw_roi and roi:
            pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
            put_chinese_text(img, "禁区", (int(roi[0][0]), int(roi[0][1]) - 22),
                             color=(0, 255, 255), size=18)

        # 绘制 Tripwire
        if self.draw_tripwire and tripwires:
            for tw in tripwires:
                p1 = (int(tw["p1"][0]), int(tw["p1"][1]))
                p2 = (int(tw["p2"][0]), int(tw["p2"][1]))
                cv2.line(img, p1, p2, (255, 0, 255), 2)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                put_chinese_text(img, tw.get("name", "入口线"), (mid[0]+5, mid[1]-20),
                                 color=(255, 0, 255), size=16)

        # 绘制检测框
        if self.draw_bbox and detections:
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                color = (0, 255, 0)
                if event.get("track_id") == det.track_id:
                    color = (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} #{det.track_id}"
                put_chinese_text(img, label, (x1, max(y1 - 20, 0)),
                                 color=color, size=14, bg_color=(0, 0, 0))

        # 绘制事件信息
        event_type = event.get("type", "unknown")
        sub_type = event.get("sub_type", "")
        label = f"事件: {event_type}"
        if sub_type:
            label += f"/{sub_type}"
        put_chinese_text(img, label, (10, 8), color=(0, 0, 255), size=22)

        # 保存文件 - 用纯 ASCII 文件名避免 URL 编码问题
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        cam_id = event.get("camera_id", "unknown")
        event_type = event.get("type", "unknown")
        track_id = event.get("track_id", 0)
        filename = f"{cam_id}_{event_type}_t{track_id}_{ts}_{ms:03d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, img)
        logger.info(f"截图已保存: {filepath}")
        return filepath


def draw_overlay(frame: np.ndarray,
                 detections: list,
                 roi: Optional[List] = None,
                 tripwires: Optional[List] = None) -> np.ndarray:
    """
    在帧上绘制叠加信息（用于 Web 实时展示）。
    使用纯 OpenCV 绘制，避免 PIL 转换开销。
    """
    img = frame.copy()

    # ROI
    if roi:
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        # 半透明填充
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
        _put_label(img, "ROI", (int(roi[0][0]), int(roi[0][1]) - 8), (0, 255, 255))

    # Tripwire
    if tripwires:
        for tw in tripwires:
            p1 = (int(tw["p1"][0]), int(tw["p1"][1]))
            p2 = (int(tw["p2"][0]), int(tw["p2"][1]))
            cv2.line(img, p1, p2, (255, 0, 255), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            _put_label(img, tw.get("name", "line"), (mid[0]+5, mid[1]-8), (255, 0, 255))

    # 检测框
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        if det.class_id == 0:
            color = (0, 255, 0)
        elif det.class_id in (2, 3, 5, 7):
            color = (255, 165, 0)
        else:
            color = (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
        _put_label(img, label, (x1, max(y1 - 8, 0)), color)

    return img



def _put_label(img: np.ndarray, text: str, pos: Tuple[int, int],
               color: Tuple[int, int, int], scale: float = 0.45,
               thickness: int = 1):
    """轻量标签绘制，含非 ASCII 字符时自动走 PIL 路径"""
    # 检测是否包含非 ASCII 字符（中文等）
    if any(ord(c) > 127 for c in text):
        font = _get_font(16)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x, y = pos
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 4), (0, 0, 0), -1)
        put_chinese_text(img, text, (x + 2, y - th - 2), color=color, size=16)
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 2, y - 2), font, scale, color, thickness, cv2.LINE_AA)


