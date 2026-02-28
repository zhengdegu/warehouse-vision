"""样本管理器 - 负责样本图片的存储、元数据管理和数据集组织"""

import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from .models import (
    BatchLimitError,
    DatasetMeta,
    DatasetStats,
    InvalidFormatError,
    PaginatedResult,
    SampleMeta,
    TrainingError,
)

# 支持的图片格式（扩展名 -> MIME 类型前缀）
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SampleManager:
    """样本管理器，负责数据样本的上传、存储、查询和删除"""

    BATCH_LIMIT = 100

    def __init__(self, base_dir: str = "data"):
        """初始化样本管理器，base_dir 为数据根目录"""
        self.base_dir = base_dir
        self.samples_dir = os.path.join(base_dir, "samples")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.samples_json = os.path.join(self.metadata_dir, "samples.json")
        self.datasets_json = os.path.join(self.metadata_dir, "datasets.json")

        # 确保目录存在
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        # 初始化元数据文件
        if not os.path.exists(self.samples_json):
            self._save_samples_meta([])
        if not os.path.exists(self.datasets_json):
            self._save_datasets_meta([])

    # ── 样本上传 ─────────────────────────────────────────────────────────────

    def upload_samples(
        self,
        files: list,
        dataset_name: Optional[str] = None,
    ) -> List[SampleMeta]:
        """
        上传样本图片。
        - 验证文件格式（仅 JPEG/PNG）
        - 验证数量限制（≤100）
        - 生成唯一 sample_id（UUID）
        - 保存到 data/samples/{dataset_name}/{sample_id}.{ext}
        - 写入元数据到 data/metadata/samples.json
        """
        # 数量限制校验
        if len(files) > self.BATCH_LIMIT:
            raise BatchLimitError(
                error="batch_limit_exceeded",
                message=f"单次上传数量不能超过 {self.BATCH_LIMIT} 张",
                details={"limit": self.BATCH_LIMIT, "received": len(files)},
            )

        # 格式校验（先全部校验，再保存，保证原子性）
        for f in files:
            filename = getattr(f, "filename", "")
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise InvalidFormatError(
                    error="invalid_format",
                    message=f"不支持的文件格式: {filename}，仅支持 JPEG/PNG",
                    details={"allowed_formats": [".jpg", ".jpeg", ".png"]},
                )

        # 确定存储子目录
        folder_name = dataset_name if dataset_name else "_unassigned"
        save_dir = os.path.join(self.samples_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        samples_meta = self._load_samples_meta()
        new_samples: List[SampleMeta] = []
        now = datetime.now(timezone.utc).isoformat()

        for f in files:
            filename = getattr(f, "filename", "")
            ext = os.path.splitext(filename)[1].lower()
            sample_id = uuid.uuid4().hex

            # 构建存储路径（相对于 base_dir）
            rel_path = os.path.join("samples", folder_name, f"{sample_id}{ext}")
            abs_path = os.path.join(self.base_dir, rel_path)

            # 写入文件
            content = self._read_file_content(f)
            with open(abs_path, "wb") as out:
                out.write(content)

            meta = SampleMeta(
                sample_id=sample_id,
                filename=filename,
                file_path=rel_path,
                dataset_name=dataset_name or "",
                upload_time=now,
                annotated=False,
                annotation_type=None,
            )
            new_samples.append(meta)

        # 追加到元数据
        samples_meta.extend(new_samples)
        self._save_samples_meta(samples_meta)

        return new_samples

    # ── 样本查询 ─────────────────────────────────────────────────────────────

    def list_samples(
        self,
        dataset_name: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResult:
        """分页查询样本列表，可按数据集筛选"""
        all_samples = self._load_samples_meta()

        if dataset_name is not None:
            all_samples = [s for s in all_samples if s.dataset_name == dataset_name]

        total = len(all_samples)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start = (page - 1) * page_size
        end = start + page_size
        items = all_samples[start:end]

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    # ── 样本删除 ─────────────────────────────────────────────────────────────

    def delete_sample(self, sample_id: str) -> None:
        """删除样本及其关联标注文件"""
        samples_meta = self._load_samples_meta()
        target = None
        for s in samples_meta:
            if s.sample_id == sample_id:
                target = s
                break

        if target is None:
            raise TrainingError(
                error="sample_not_found",
                message=f"样本不存在: {sample_id}",
            )

        # 删除图片文件
        abs_path = os.path.join(self.base_dir, target.file_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)

        # 删除关联标注文件
        self._delete_label_file(target)

        # 更新元数据
        samples_meta = [s for s in samples_meta if s.sample_id != sample_id]
        self._save_samples_meta(samples_meta)

    # ── 数据集管理 ───────────────────────────────────────────────────────────

    def create_dataset(self, dataset_name: str, classes: List[str]) -> DatasetMeta:
        """创建数据集，指定类别列表"""
        datasets = self._load_datasets_meta()

        # 检查重名
        for d in datasets:
            if d.name == dataset_name:
                raise TrainingError(
                    error="dataset_exists",
                    message=f"数据集已存在: {dataset_name}",
                )

        now = datetime.now(timezone.utc).isoformat()
        meta = DatasetMeta(name=dataset_name, classes=classes, created_time=now)
        datasets.append(meta)
        self._save_datasets_meta(datasets)

        # 创建样本和标注子目录
        os.makedirs(os.path.join(self.samples_dir, dataset_name), exist_ok=True)
        os.makedirs(os.path.join(self.labels_dir, dataset_name), exist_ok=True)

        return meta

    def delete_dataset(self, dataset_name: str) -> None:
        """删除数据集及其所有样本和标注"""
        datasets = self._load_datasets_meta()
        found = any(d.name == dataset_name for d in datasets)
        if not found:
            raise TrainingError(
                error="dataset_not_found",
                message=f"数据集不存在: {dataset_name}",
            )

        # 删除该数据集下的所有样本元数据
        samples_meta = self._load_samples_meta()
        samples_meta = [s for s in samples_meta if s.dataset_name != dataset_name]
        self._save_samples_meta(samples_meta)

        # 删除样本文件目录
        samples_path = os.path.join(self.samples_dir, dataset_name)
        if os.path.exists(samples_path):
            shutil.rmtree(samples_path)

        # 删除标注文件目录
        labels_path = os.path.join(self.labels_dir, dataset_name)
        if os.path.exists(labels_path):
            shutil.rmtree(labels_path)

        # 更新数据集元数据
        datasets = [d for d in datasets if d.name != dataset_name]
        self._save_datasets_meta(datasets)

    def get_dataset_stats(self, dataset_name: str) -> DatasetStats:
        """返回数据集统计：样本总数、已标注数、未标注数"""
        datasets = self._load_datasets_meta()
        found = any(d.name == dataset_name for d in datasets)
        if not found:
            raise TrainingError(
                error="dataset_not_found",
                message=f"数据集不存在: {dataset_name}",
            )

        samples = self._load_samples_meta()
        ds_samples = [s for s in samples if s.dataset_name == dataset_name]
        total = len(ds_samples)
        annotated = sum(1 for s in ds_samples if s.annotated)

        return DatasetStats(
            total_samples=total,
            annotated_count=annotated,
            unannotated_count=total - annotated,
        )

    def list_datasets(self) -> List[DatasetMeta]:
        """列出所有数据集"""
        return self._load_datasets_meta()

    # ── 内部辅助方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_file_content(f) -> bytes:
        """从类文件对象中读取内容（兼容 FastAPI UploadFile 和普通对象）"""
        if hasattr(f, "file"):
            # FastAPI UploadFile
            return f.file.read()
        elif hasattr(f, "read"):
            return f.read()
        elif hasattr(f, "content"):
            return f.content if isinstance(f.content, bytes) else f.content.encode()
        return b""

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

    def _load_datasets_meta(self) -> List[DatasetMeta]:
        """从 JSON 文件加载数据集元数据"""
        if not os.path.exists(self.datasets_json):
            return []
        with open(self.datasets_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [DatasetMeta(**item) for item in data]

    def _save_datasets_meta(self, datasets: List[DatasetMeta]) -> None:
        """将数据集元数据保存到 JSON 文件"""
        data = []
        for d in datasets:
            data.append({
                "name": d.name,
                "classes": d.classes,
                "created_time": d.created_time,
            })
        with open(self.datasets_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _delete_label_file(self, sample: SampleMeta) -> None:
        """删除样本关联的标注文件"""
        if not sample.dataset_name:
            return
        label_path = os.path.join(
            self.labels_dir,
            sample.dataset_name,
            f"{sample.sample_id}.txt",
        )
        if os.path.exists(label_path):
            os.remove(label_path)
