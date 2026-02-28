"""
目标检测与跟踪模块
基于 Ultralytics YOLO，支持可配置模型路径和类别白名单。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

# 自动检测 GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"推理设备: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))


@dataclass
class Detection:
    """检测结果数据类"""
    track_id: int = -1          # 跟踪 ID，-1 表示未跟踪
    class_id: int = 0           # 类别 ID
    class_name: str = ""        # 类别名称
    confidence: float = 0.0     # 置信度
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])  # [x1, y1, x2, y2]
    center: tuple = (0, 0)      # 中心点
    foot: tuple = (0, 0)        # 脚点（bbox 底边中点）
    keypoints: Optional[np.ndarray] = None  # 姿态关键点 (17, 3) [x, y, conf]


class YOLODetector:
    """YOLO 检测器，封装 Ultralytics"""

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.5,
                 allowed_classes: Optional[List[int]] = None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(DEVICE)
        self.confidence = confidence
        self.allowed_classes = allowed_classes
        logger.info(f"YOLO 模型已加载: {model_path}, 设备: {DEVICE}, 白名单: {allowed_classes}")

    def _parse_results(self, results, with_track: bool = False) -> List[Detection]:
        """解析 YOLO 结果为 Detection 列表"""
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())

                # 类别白名单过滤
                if self.allowed_classes and cls_id not in self.allowed_classes:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                fx = cx
                fy = y2  # 脚点

                track_id = -1
                if with_track and boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                cls_name = self.model.names.get(cls_id, str(cls_id))

                detections.append(Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    center=(cx, cy),
                    foot=(fx, fy)
                ))

        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """纯检测，不跟踪"""
        results = self.model(frame, conf=self.confidence, device=DEVICE, verbose=False)
        return self._parse_results(results, with_track=False)

    def track(self, frame: np.ndarray) -> List[Detection]:
        """检测 + 跟踪（ByteTrack）"""
        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker="bytetrack.yaml",
            device=DEVICE,
            verbose=False
        )
        return self._parse_results(results, with_track=True)


# COCO 关键点索引常量
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


class PoseDetector:
    """
    YOLO Pose 检测器 — 输出人体关键点。
    用于增强打架/跌倒等行为检测的精度。
    仅检测 person 类别。
    """

    def __init__(self, model_path: str = "yolo26m-pose.pt",
                 confidence: float = 0.3):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(DEVICE)
        self.confidence = confidence
        logger.info(f"Pose 模型已加载: {model_path}, 设备: {DEVICE}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """纯姿态检测，不跟踪"""
        results = self.model(frame, conf=self.confidence, device=DEVICE, verbose=False)
        return self._parse_results(results, with_track=False)

    def track(self, frame: np.ndarray) -> List[Detection]:
        """姿态检测 + 跟踪"""
        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker="bytetrack.yaml",
            device=DEVICE,
            verbose=False,
        )
        return self._parse_results(results, with_track=True)

    def _parse_results(self, results, with_track: bool = False) -> List[Detection]:
        detections = []
        for r in results:
            boxes = r.boxes
            kps = r.keypoints
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                track_id = -1
                if with_track and boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                # 提取关键点 (17, 3): [x, y, confidence]
                keypoints = None
                if kps is not None and kps.data is not None and i < len(kps.data):
                    keypoints = kps.data[i].cpu().numpy()  # (17, 3)

                detections.append(Detection(
                    track_id=track_id,
                    class_id=0,
                    class_name="person",
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    center=(cx, cy),
                    foot=(cx, y2),
                    keypoints=keypoints,
                ))

        return detections
