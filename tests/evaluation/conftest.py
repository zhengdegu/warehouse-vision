"""Shared fixtures for the evaluation test suite.

Provides reusable fixtures for:
- Temporary directories with sample image/label data
- Sample YOLO annotation data
- Sample video annotation JSON data
- Sample detection results (DetectionResult, ClassMetrics)
- Sample rule evaluation results (RuleResult, RuleMetrics)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from src.evaluation.models import (
    ClassMetrics,
    CountingDetail,
    DetectionResult,
    GroundTruthBox,
    ImageEvalEntry,
    LabelSummary,
    MatchResult,
    RuleMetrics,
    RuleResult,
    VideoAnnotation,
    VideoEvalResult,
)


# ── Stub Detection (avoids torch/ultralytics dependency) ──


@dataclass
class StubDetection:
    """Lightweight Detection stub for testing without torch dependency."""

    track_id: int = -1
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    center: tuple = (0, 0)
    foot: tuple = (0, 0)
    keypoints: Optional[np.ndarray] = None


# ── Temporary directory fixtures ──


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a temp directory with 3 dummy 640x480 JPEG images."""
    from PIL import Image as PILImage

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ["img_001.jpg", "img_002.jpg", "img_003.jpg"]:
        img = PILImage.new("RGB", (640, 480), color=(0, 0, 0))
        img.save(str(img_dir / name))
    return img_dir


@pytest.fixture
def sample_label_dir(tmp_path):
    """Create a temp directory with YOLO label files matching sample images.

    - img_001.txt: 2 person boxes
    - img_002.txt: 1 car box
    - img_003.txt: no detections (comment only)
    """
    lbl_dir = tmp_path / "labels"
    lbl_dir.mkdir()
    (lbl_dir / "img_001.txt").write_text(
        "# conf=0.90 person\n"
        "0 0.3125 0.2604 0.3125 0.4167\n"
        "# conf=0.80 person\n"
        "0 0.7031 0.5208 0.1563 0.2083\n"
    )
    (lbl_dir / "img_002.txt").write_text(
        "# conf=0.75 car\n"
        "2 0.5000 0.5000 0.4688 0.4167\n"
    )
    (lbl_dir / "img_003.txt").write_text("# no detections\n")
    return lbl_dir


@pytest.fixture
def empty_label_dir(tmp_path):
    """Create an empty label directory."""
    lbl_dir = tmp_path / "empty_labels"
    lbl_dir.mkdir()
    return lbl_dir


# ── Sample YOLO annotation data ──


@pytest.fixture
def sample_gt_boxes():
    """Return a list of sample GroundTruthBox objects for testing."""
    return [
        GroundTruthBox(class_id=0, x1=100.0, y1=25.0, x2=300.0, y2=225.0),
        GroundTruthBox(class_id=0, x1=400.0, y1=200.0, x2=500.0, y2=300.0),
        GroundTruthBox(class_id=2, x1=170.0, y1=140.0, x2=470.0, y2=340.0),
    ]


# ── Sample video annotation data ──


@pytest.fixture
def sample_video_annotation_data():
    """Return a dict representing a valid video annotation JSON."""
    return {
        "video_path": "clip_01.mp4",
        "fps_sample": 5,
        "camera_config": {
            "roi": [[100, 100], [500, 100], [500, 400], [100, 400]],
            "tripwires": [
                {
                    "id": "tw01",
                    "name": "入口线",
                    "p1": [200, 100],
                    "p2": [200, 400],
                    "direction": "left_to_right",
                    "cooldown": 2.0,
                }
            ],
            "intrusion": {"confirm_frames": 5, "cooldown": 30.0},
            "anomaly": {
                "dwell": {"enabled": True, "max_seconds": 120},
                "crowd": {"enabled": True, "max_count": 5, "radius": 200},
                "proximity": {"enabled": True, "min_distance": 50},
                "fight": {"enabled": True},
                "fall": {"enabled": True},
            },
        },
        "events": [
            {
                "type": "intrusion",
                "time_range": [2.0, 8.0],
                "class_name": "person",
            },
            {
                "type": "tripwire",
                "time_sec": 3.5,
                "direction": "in",
                "class_name": "car",
                "tripwire_id": "tw01",
            },
            {
                "type": "counting",
                "expected_in": 2,
                "expected_out": 1,
                "by_class": {
                    "person": {"in": 1, "out": 1},
                    "car": {"in": 1, "out": 0},
                },
            },
            {
                "type": "presence",
                "time_range": [0, 10],
                "class_name": "person",
                "expected_count": 3,
            },
            {
                "type": "anomaly",
                "sub_type": "dwell",
                "time_range": [10.0, 25.0],
                "class_name": "person",
            },
        ],
    }


@pytest.fixture
def sample_video_annotation(sample_video_annotation_data):
    """Return a VideoAnnotation model from sample data."""
    d = sample_video_annotation_data
    return VideoAnnotation(
        video_path=d["video_path"],
        fps_sample=d["fps_sample"],
        camera_config=d["camera_config"],
        events=d["events"],
    )


@pytest.fixture
def sample_video_annotation_json(tmp_path, sample_video_annotation_data):
    """Write sample video annotation to a JSON file and return the path."""
    json_path = tmp_path / "clip_01.json"
    json_path.write_text(json.dumps(sample_video_annotation_data, indent=2))
    return json_path


# ── Sample evaluation results ──


@pytest.fixture
def sample_class_metrics():
    """Return a list of sample ClassMetrics for person and car."""
    return [
        ClassMetrics(
            class_id=0,
            class_name="person",
            ap=0.85,
            precision=0.90,
            recall=0.80,
            tp=8,
            fp=1,
            fn=2,
            gt_count=10,
            has_samples=True,
        ),
        ClassMetrics(
            class_id=2,
            class_name="car",
            ap=0.72,
            precision=0.75,
            recall=0.70,
            tp=7,
            fp=2,
            fn=3,
            gt_count=10,
            has_samples=True,
        ),
    ]


@pytest.fixture
def sample_detection_result(sample_class_metrics):
    """Return a sample DetectionResult with realistic metrics."""
    return DetectionResult(
        map50=0.785,
        map50_95=0.52,
        total_precision=0.83,
        total_recall=0.75,
        class_metrics=sample_class_metrics,
        confusion_matrix=[
            [8, 0, 2],
            [1, 7, 3],
            [1, 2, 0],
        ],
        per_image=[
            ImageEvalEntry(filename="img_001.jpg", gt_count=2, det_count=2, tp_count=2),
            ImageEvalEntry(filename="img_002.jpg", gt_count=1, det_count=1, tp_count=1),
            ImageEvalEntry(filename="img_003.jpg", gt_count=0, det_count=0, tp_count=0),
        ],
        skipped_images=0,
    )


@pytest.fixture
def sample_rule_metrics():
    """Return a list of sample RuleMetrics covering all rule types."""
    return [
        RuleMetrics(
            rule_type="intrusion",
            sub_type="",
            precision=0.80,
            recall=0.75,
            f1=0.77,
            tp=3,
            fp=1,
            fn=1,
            has_samples=True,
            details={
                "false_positives": [{"timestamp": 20.0, "class_name": "person"}],
                "false_negatives": [{"time_range": [30.0, 35.0], "class_name": "person"}],
            },
        ),
        RuleMetrics(
            rule_type="tripwire",
            sub_type="",
            precision=1.0,
            recall=0.67,
            f1=0.80,
            tp=2,
            fp=0,
            fn=1,
            has_samples=True,
            details={"direction_accuracy": 1.0},
        ),
        RuleMetrics(
            rule_type="counting",
            sub_type="",
            precision=0.0,
            recall=0.0,
            f1=0.0,
            tp=0,
            fp=0,
            fn=0,
            has_samples=True,
            details={
                "expected_in": 2,
                "expected_out": 1,
                "actual_in": 2,
                "actual_out": 1,
                "abs_error_in": 0,
                "abs_error_out": 0,
            },
        ),
        RuleMetrics(
            rule_type="presence",
            sub_type="",
            precision=1.0,
            recall=1.0,
            f1=1.0,
            tp=1,
            fp=0,
            fn=0,
            has_samples=True,
            details={},
        ),
        RuleMetrics(
            rule_type="anomaly",
            sub_type="dwell",
            precision=0.50,
            recall=1.0,
            f1=0.67,
            tp=1,
            fp=1,
            fn=0,
            has_samples=True,
            details={},
        ),
    ]


@pytest.fixture
def sample_rule_result(sample_rule_metrics):
    """Return a sample RuleResult with one video evaluation."""
    return RuleResult(
        per_video=[
            VideoEvalResult(
                video_name="clip_01.mp4",
                rule_metrics=sample_rule_metrics,
                errors=[],
            ),
        ],
        aggregated=sample_rule_metrics,
    )
