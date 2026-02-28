"""数据集准备器 - 将数据集转换为 YOLO 训练格式"""

import json
import os
import random
import shutil
from typing import List

import yaml

from .models import SampleMeta, DatasetMeta


class DatasetPreparer:
    """数据集准备器，负责将标注数据组织为 YOLO 训练目录结构"""

    def __init__(self, base_dir: str = "data"):
        """初始化数据集准备器"""
        self.base_dir = base_dir
        self.samples_dir = os.path.join(base_dir, "samples")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.training_dir = os.path.join(base_dir, "training")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.samples_json = os.path.join(self.metadata_dir, "samples.json")
        self.datasets_json = os.path.join(self.metadata_dir, "datasets.json")

    def prepare(
        self,
        dataset_name: str,
        job_id: str,
        train_ratio: float = 0.8,
    ) -> str:
        """
        准备 YOLO 训练目录结构。
        - 创建 data/training/{job_id}/images/train, images/val, labels/train, labels/val
        - 按 train_ratio 随机拆分已标注样本
        - 复制图片和标注文件
        - 生成 data.yaml 配置文件
        - 返回 data.yaml 的路径
        """
        job_dir = os.path.join(self.training_dir, job_id)

        # 创建 YOLO 训练目录结构
        for split in ("train", "val"):
            os.makedirs(os.path.join(job_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(job_dir, "labels", split), exist_ok=True)

        # 加载已标注样本
        samples = self._load_samples_meta()
        annotated = [
            s for s in samples
            if s.dataset_name == dataset_name and s.annotated
        ]

        # 随机拆分
        random.shuffle(annotated)
        split_idx = round(len(annotated) * train_ratio)
        train_samples = annotated[:split_idx]
        val_samples = annotated[split_idx:]

        # 复制文件到训练目录
        self._copy_samples(train_samples, dataset_name, job_dir, "train")
        self._copy_samples(val_samples, dataset_name, job_dir, "val")

        # 加载数据集类别信息并生成 data.yaml
        dataset_meta = self._load_dataset_meta(dataset_name)
        classes = dataset_meta.classes if dataset_meta else []
        yaml_path = self.generate_data_yaml(job_dir, classes)

        return yaml_path

    def generate_data_yaml(self, job_dir: str, classes: List[str]) -> str:
        """
        生成 YOLO data.yaml 配置文件。
        内容包含：path, train, val, nc, names
        """
        abs_job_dir = os.path.abspath(job_dir)
        data = {
            "path": abs_job_dir,
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": classes,
        }

        yaml_path = os.path.join(job_dir, "data.yaml")
        os.makedirs(job_dir, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return yaml_path

    @staticmethod
    def parse_data_yaml(yaml_path: str) -> dict:
        """解析 data.yaml 文件为字典"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ── 内部辅助方法 ─────────────────────────────────────────────────────────

    def _copy_samples(
        self,
        samples: List[SampleMeta],
        dataset_name: str,
        job_dir: str,
        split: str,
    ) -> None:
        """复制图片和标注文件到训练目录的指定拆分子目录"""
        for sample in samples:
            # 复制图片文件
            src_image = os.path.join(self.base_dir, sample.file_path)
            if os.path.exists(src_image):
                dst_image = os.path.join(
                    job_dir, "images", split, os.path.basename(sample.file_path)
                )
                shutil.copy2(src_image, dst_image)

            # 复制标注文件
            label_filename = f"{sample.sample_id}.txt"
            src_label = os.path.join(self.labels_dir, dataset_name, label_filename)
            if os.path.exists(src_label):
                dst_label = os.path.join(job_dir, "labels", split, label_filename)
                shutil.copy2(src_label, dst_label)

    def _load_samples_meta(self) -> List[SampleMeta]:
        """从 JSON 文件加载样本元数据"""
        if not os.path.exists(self.samples_json):
            return []
        with open(self.samples_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [SampleMeta(**item) for item in data]

    def _load_dataset_meta(self, dataset_name: str):
        """加载指定数据集的元数据"""
        if not os.path.exists(self.datasets_json):
            return None
        with open(self.datasets_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            meta = DatasetMeta(**item)
            if meta.name == dataset_name:
                return meta
        return None
