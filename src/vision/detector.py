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

class RoboflowDetector:
    """
    Roboflow RF-DETR 检测器 + supervision ByteTrack 跟踪。
    与 YOLODetector 接口一致（detect/track），可无缝替换。

    参数:
        model_id: RF-DETR 模型标识，可选:
            "rfdetr-base", "rfdetr-large", "rfdetr-nano",
            "rfdetr-small", "rfdetr-medium", "rfdetr-seg-preview"
        confidence: 置信度阈值
        allowed_classes: 类别名称白名单（如 ["person", "car"]），None=全部
    """

    # 延迟导入的模型映射
    _MODEL_REGISTRY = None

    @classmethod
    def _get_model_registry(cls):
        if cls._MODEL_REGISTRY is None:
            from rfdetr import RFDETRBase, RFDETRLarge
            try:
                from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
            except ImportError:
                RFDETRNano = RFDETRSmall = RFDETRMedium = None
            try:
                from rfdetr import RFDETRSegPreview
            except ImportError:
                RFDETRSegPreview = None

            registry = {
                "rfdetr-base": RFDETRBase,
                "rfdetr-large": RFDETRLarge,
            }
            if RFDETRNano:
                registry["rfdetr-nano"] = RFDETRNano
            if RFDETRSmall:
                registry["rfdetr-small"] = RFDETRSmall
            if RFDETRMedium:
                registry["rfdetr-medium"] = RFDETRMedium
            if RFDETRSegPreview:
                registry["rfdetr-seg-preview"] = RFDETRSegPreview
            cls._MODEL_REGISTRY = registry
        return cls._MODEL_REGISTRY

    def __init__(self, model_id: str = "rfdetr-base",
                 confidence: float = 0.5,
                 allowed_classes: Optional[List[str]] = None):
        import supervision as sv
        import warnings

        registry = self._get_model_registry()
        if model_id not in registry:
            raise ValueError(
                f"未知 model_id: {model_id}，可选: {list(registry.keys())}")

        logger.info(f"加载 Roboflow RF-DETR 模型: {model_id}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = registry[model_id]()
            try:
                self.model.optimize_for_inference()
            except RuntimeError:
                self.model.optimize_for_inference(compile=False)

        self.confidence = confidence
        self.allowed_classes = allowed_classes
        self.class_names = self.model.class_names  # {id: name}

        # supervision ByteTrack 跟踪器
        self._tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        logger.info(
            f"Roboflow RF-DETR 已加载: {model_id}, "
            f"类别白名单: {allowed_classes}")

    def _sv_to_detections(self, sv_dets, with_track: bool = False) -> List[Detection]:
        """将 supervision.Detections 转换为项目 Detection 列表"""
        detections = []
        if sv_dets.class_id is None or not sv_dets.class_id.size:
            return detections

        for i in range(len(sv_dets)):
            cls_id = int(sv_dets.class_id[i])
            cls_name = self.class_names.get(cls_id, str(cls_id))

            # 类别白名单过滤
            if self.allowed_classes and cls_name not in self.allowed_classes:
                continue

            conf = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else 0.0
            x1, y1, x2, y2 = sv_dets.xyxy[i].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            track_id = -1
            if with_track and sv_dets.tracker_id is not None:
                track_id = int(sv_dets.tracker_id[i])

            detections.append(Detection(
                track_id=track_id,
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                center=(cx, cy),
                foot=(cx, y2),
            ))

        return detections

    def _predict(self, frame: np.ndarray):
        """运行 RF-DETR 推理，返回 supervision.Detections"""
        import supervision as sv

        result = self.model.predict(frame, confidence=self.confidence)
        sv_dets = result[0] if isinstance(result, list) else result

        if sv_dets.class_id is None or not sv_dets.class_id.size:
            return sv.Detections.empty()
        return sv_dets

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """纯检测，不跟踪"""
        sv_dets = self._predict(frame)
        return self._sv_to_detections(sv_dets, with_track=False)

    def track(self, frame: np.ndarray) -> List[Detection]:
        """检测 + ByteTrack 跟踪"""
        sv_dets = self._predict(frame)
        tracked = self._tracker.update_with_detections(sv_dets)
        return self._sv_to_detections(tracked, with_track=True)

