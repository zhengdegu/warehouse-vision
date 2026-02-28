"""标注引擎 - 负责手动标注的保存/查询和自动标注的执行"""

import json
import logging
import os
from typing import Dict, List, Optional

from .models import (
    AnnotationBox,
    AutoAnnotateResult,
    InvalidAnnotationError,
    SampleMeta,
    TrainingError,
)

logger = logging.getLogger(__name__)


class AnnotationEngine:
    """标注引擎，提供手动标注和自动标注功能"""

    def __init__(self, base_dir: str = "data"):
        """初始化标注引擎"""
        self.base_dir = base_dir
        self.labels_dir = os.path.join(base_dir, "labels")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.samples_json = os.path.join(self.metadata_dir, "samples.json")
        self.datasets_json = os.path.join(self.metadata_dir, "datasets.json")

        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    # ── 保存标注 ─────────────────────────────────────────────────────────────

    def save_annotation(
        self,
        sample_id: str,
        annotations: List[AnnotationBox],
        dataset_classes: List[str],
    ) -> None:
        """
        保存标注数据为 YOLO 格式文件。
        - 验证坐标归一化范围 [0, 1]
        - 验证 class_id 在数据集类别列表范围内
        - 写入 data/labels/{dataset_name}/{sample_id}.txt
        - 更新样本元数据中的标注状态
        """
        # 查找样本元数据
        sample = self._find_sample(sample_id)

        # 验证所有标注
        num_classes = len(dataset_classes)
        for box in annotations:
            self._validate_annotation(box, num_classes, dataset_classes)

        # 写入标注文件
        dataset_name = sample.dataset_name
        label_dir = os.path.join(self.labels_dir, dataset_name)
        os.makedirs(label_dir, exist_ok=True)
        label_path = os.path.join(label_dir, f"{sample_id}.txt")

        with open(label_path, "w", encoding="utf-8") as f:
            for box in annotations:
                f.write(box.to_yolo_line() + "\n")

        # 更新样本元数据
        self._update_sample_annotation_status(
            sample_id, annotated=True, annotation_type="manual"
        )

    # ── 读取标注 ─────────────────────────────────────────────────────────────

    def get_annotation(self, sample_id: str) -> List[AnnotationBox]:
        """读取并返回样本的标注数据"""
        sample = self._find_sample(sample_id)
        dataset_name = sample.dataset_name
        label_path = os.path.join(
            self.labels_dir, dataset_name, f"{sample_id}.txt"
        )

        if not os.path.exists(label_path):
            return []

        annotations: List[AnnotationBox] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(AnnotationBox.from_yolo_line(line))
        return annotations

    # ── 自动标注 ─────────────────────────────────────────────────────────────

    def auto_annotate(
        self,
        dataset_name: str,
        model_path: str,
        confidence_threshold: float = 0.5,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> AutoAnnotateResult:
        """
        使用 YOLO 模型对数据集中未标注样本执行自动标注。
        - 加载指定模型执行推理
        - 将模型输出的 class_id 重映射到数据集类别索引
        - 过滤低于置信度阈值的检测结果
        - 保存为 YOLO 格式标注文件，标记为"自动标注"
        - 单张图片推理失败时记录日志并继续
        - 返回处理摘要（成功/失败/跳过数量）
        """
        from ultralytics import YOLO
        import torch

        # 加载模型（自动使用 GPU）
        _device = "0" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)

        # 构建模型 class name → 数据集 class_id 的映射
        # 例如模型 names={0:'person', 2:'car', 7:'truck', ...}
        # 数据集 classes=['person','car'] → person→0, car→1
        # 则映射: 模型class_id 0 → 数据集 0, 模型class_id 2 → 数据集 1
        dataset_classes = self._get_dataset_classes(dataset_name)
        ds_name_to_idx = {name.lower(): idx for idx, name in enumerate(dataset_classes)}
        model_names = model.names  # dict: {int: str}

        # 模型 class_id → 数据集 class_id（仅保留数据集中存在的类别）
        model_to_ds: Dict[int, int] = {}
        if class_mapping is not None:
            model_to_ds = class_mapping
        else:
            for model_cls_id, model_cls_name in model_names.items():
                ds_idx = ds_name_to_idx.get(model_cls_name.lower())
                if ds_idx is not None:
                    model_to_ds[model_cls_id] = ds_idx

        if not model_to_ds:
            logger.warning(
                "自动标注: 模型类别与数据集类别无交集 "
                "(模型: %s, 数据集: %s)",
                list(model_names.values()), dataset_classes,
            )

        # 获取数据集中的所有样本
        samples = self._load_samples_meta()
        ds_samples = [s for s in samples if s.dataset_name == dataset_name]

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for sample in ds_samples:
            # 跳过已标注样本
            if sample.annotated:
                skipped_count += 1
                continue

            try:
                image_path = os.path.join(self.base_dir, sample.file_path)
                results = model(image_path, device=_device)

                annotations: List[AnnotationBox] = []
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i])
                        if conf <= confidence_threshold:
                            continue

                        model_cls_id = int(boxes.cls[i])

                        # 重映射到数据集类别，跳过不在数据集中的类别
                        if model_cls_id not in model_to_ds:
                            continue
                        ds_cls_id = model_to_ds[model_cls_id]

                        # Convert xyxy normalized to center format
                        xyxyn = boxes.xyxyn[i]
                        x1, y1, x2, y2 = (
                            float(xyxyn[0]),
                            float(xyxyn[1]),
                            float(xyxyn[2]),
                            float(xyxyn[3]),
                        )
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        width = x2 - x1
                        height = y2 - y1

                        annotations.append(
                            AnnotationBox(
                                class_id=ds_cls_id,
                                center_x=center_x,
                                center_y=center_y,
                                width=width,
                                height=height,
                            )
                        )

                # 写入标注文件
                label_dir = os.path.join(self.labels_dir, dataset_name)
                os.makedirs(label_dir, exist_ok=True)
                label_path = os.path.join(
                    label_dir, f"{sample.sample_id}.txt"
                )
                with open(label_path, "w", encoding="utf-8") as f:
                    for box in annotations:
                        f.write(box.to_yolo_line() + "\n")

                # 更新样本元数据
                self._update_sample_annotation_status(
                    sample.sample_id, annotated=True, annotation_type="auto"
                )
                success_count += 1

            except Exception as e:
                logger.error(
                    "自动标注失败 sample_id=%s: %s", sample.sample_id, str(e)
                )
                failed_count += 1

        return AutoAnnotateResult(
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
        )

    # ── 内部辅助方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate_annotation(
        box: AnnotationBox, num_classes: int, dataset_classes: List[str]
    ) -> None:
        """验证单个标注框的坐标范围和 class_id"""
        # 坐标归一化范围 [0, 1]
        for attr_name in ("center_x", "center_y", "width", "height"):
            value = getattr(box, attr_name)
            if value < 0.0 or value > 1.0:
                raise InvalidAnnotationError(
                    error="invalid_coordinates",
                    message=f"标注坐标 {attr_name}={value} 超出 [0, 1] 范围",
                    details={"field": attr_name, "value": value, "valid_range": [0, 1]},
                )

        # class_id 有效性
        if box.class_id < 0 or box.class_id >= num_classes:
            raise InvalidAnnotationError(
                error="invalid_class_id",
                message=f"class_id={box.class_id} 不在有效范围内",
                details={
                    "class_id": box.class_id,
                    "valid_range": [0, num_classes - 1],
                    "valid_classes": dataset_classes,
                },
            )

    def _find_sample(self, sample_id: str) -> SampleMeta:
        """根据 sample_id 查找样本元数据"""
        samples = self._load_samples_meta()
        for s in samples:
            if s.sample_id == sample_id:
                return s
        raise TrainingError(
            error="sample_not_found",
            message=f"样本不存在: {sample_id}",
        )

    def _get_dataset_classes(self, dataset_name: str) -> List[str]:
        """获取数据集的类别列表"""
        if not os.path.exists(self.datasets_json):
            return []
        with open(self.datasets_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("name") == dataset_name:
                return item.get("classes", [])
        return []

    def _load_samples_meta(self) -> List[SampleMeta]:
        """从 JSON 文件加载样本元数据"""
        if not os.path.exists(self.samples_json):
            return []
        with open(self.samples_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [SampleMeta(**item) for item in data]

    def _save_samples_meta(self, samples: List[SampleMeta]) -> None:
        """将样本元数据保存到 JSON 文件"""
        data = []
        for s in samples:
            data.append({
                "sample_id": s.sample_id,
                "filename": s.filename,
                "file_path": s.file_path,
                "dataset_name": s.dataset_name,
                "upload_time": s.upload_time,
                "annotated": s.annotated,
                "annotation_type": s.annotation_type,
            })
        with open(self.samples_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _update_sample_annotation_status(
        self, sample_id: str, annotated: bool, annotation_type: str
    ) -> None:
        """更新样本的标注状态"""
        samples = self._load_samples_meta()
        for s in samples:
            if s.sample_id == sample_id:
                s.annotated = annotated
                s.annotation_type = annotation_type
                break
        self._save_samples_meta(samples)
