"""训练子系统核心数据模型和异常类"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ── 枚举 ──────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── 核心数据类 ────────────────────────────────────────────────────────────────


@dataclass
class SampleMeta:
    """样本元数据"""
    sample_id: str
    filename: str
    file_path: str
    dataset_name: str
    upload_time: str
    annotated: bool = False
    annotation_type: Optional[str] = None


@dataclass
class AnnotationBox:
    """单个标注框（YOLO 格式）"""
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        """转换为 YOLO 格式行：class_id cx cy w h"""
        return f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} {self.width:.6f} {self.height:.6f}"

    @staticmethod
    def from_yolo_line(line: str) -> "AnnotationBox":
        """从 YOLO 格式行解析"""
        parts = line.strip().split()
        return AnnotationBox(
            class_id=int(parts[0]),
            center_x=float(parts[1]),
            center_y=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )


@dataclass
class DatasetMeta:
    """数据集元数据"""
    name: str
    classes: List[str]
    created_time: str


@dataclass
class DatasetStats:
    """数据集统计"""
    total_samples: int
    annotated_count: int
    unannotated_count: int


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    train_loss: float
    map50: float = 0.0
    map50_95: float = 0.0


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    dataset_name: str
    status: JobStatus = JobStatus.PENDING
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640
    base_model: str = "yolov8n.pt"
    parent_model_id: Optional[str] = None
    current_epoch: int = 0
    metrics: List[TrainingMetrics] = field(default_factory=list)
    best_map50: float = 0.0
    error_message: Optional[str] = None
    created_time: str = ""
    completed_time: Optional[str] = None
    output_model_id: Optional[str] = None


@dataclass
class ModelVersion:
    """模型版本"""
    model_id: str
    version: str
    job_id: str
    dataset_name: str
    weights_path: str
    metrics: dict
    training_config: dict
    parent_model_id: Optional[str] = None
    published: bool = False
    created_time: str = ""


@dataclass
class AutoAnnotateResult:
    """自动标注结果摘要"""
    success_count: int
    failed_count: int
    skipped_count: int


@dataclass
class PaginatedResult:
    """通用分页结果"""
    items: List
    total: int
    page: int
    page_size: int
    total_pages: int


@dataclass
class ErrorResponse:
    """统一错误响应结构"""
    error: str
    message: str
    details: dict


# ── 异常类层次结构 ─────────────────────────────────────────────────────────────


class TrainingError(Exception):
    """训练子系统基础异常"""

    def __init__(self, error: str, message: str, details: dict = None):
        self.error = error
        self.message = message
        self.details = details or {}
        super().__init__(message)


class InvalidFormatError(TrainingError):
    """文件格式不支持"""
    pass


class BatchLimitError(TrainingError):
    """上传数量超限"""
    pass


class InvalidAnnotationError(TrainingError):
    """标注数据无效（坐标越界或 class_id 无效）"""
    pass


class InsufficientSamplesError(TrainingError):
    """样本数不足"""
    pass


class ModelNotFoundError(TrainingError):
    """模型版本不存在"""
    pass


class InvalidStateError(TrainingError):
    """无效状态转换"""
    pass


class ModelInUseError(TrainingError):
    """无法删除生产模型"""
    pass
