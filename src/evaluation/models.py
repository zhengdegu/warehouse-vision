"""Data models for the accuracy evaluation framework."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ── Image evaluation models ──


@dataclass
class GroundTruthBox:
    """YOLO 格式真值框（像素坐标）"""
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ImageEvalEntry:
    """单张图片评估结果"""
    filename: str
    gt_count: int
    det_count: int
    tp_count: int


@dataclass
class ClassMetrics:
    """单类别指标"""
    class_id: int
    class_name: str
    ap: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    gt_count: int
    has_samples: bool


@dataclass
class DetectionResult:
    """图片评估完整结果"""
    map50: float
    map50_95: float
    total_precision: float
    total_recall: float
    class_metrics: List[ClassMetrics]
    confusion_matrix: list
    per_image: List[ImageEvalEntry]
    skipped_images: int


@dataclass
class LabelSummary:
    """预标注统计摘要"""
    total_images: int
    skipped: int
    labeled: int
    class_counts: Dict[str, int]


# ── Video evaluation models ──


@dataclass
class VideoAnnotation:
    """视频事件标注"""
    video_path: str
    fps_sample: int
    camera_config: dict
    events: List[dict]


@dataclass
class MatchResult:
    """事件匹配结果"""
    tp: int
    fp: int
    fn: int
    matched_pairs: List[Tuple[dict, dict]]
    false_positives: List[dict]
    false_negatives: List[dict]


@dataclass
class RuleMetrics:
    """单规则类型指标"""
    rule_type: str
    sub_type: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    has_samples: bool
    details: dict


@dataclass
class VideoEvalResult:
    """单视频评估结果"""
    video_name: str
    rule_metrics: List[RuleMetrics]
    errors: List[str]


@dataclass
class RuleResult:
    """视频评估完整结果"""
    per_video: List[VideoEvalResult]
    aggregated: List[RuleMetrics]


@dataclass
class CountingDetail:
    """计数评估详情"""
    class_name: str
    expected_in: int
    expected_out: int
    actual_in: int
    actual_out: int
    abs_error_in: int
    abs_error_out: int
    rel_error_in: float
    rel_error_out: float
