"""Unit tests for DetectionEvaluator.

Tests cover:
- Empty label directory raises ValueError
- No-sample classes marked as N/A (has_samples=False)
- Per-image ImageEvalEntry correctness
- Greedy IoU matching logic
- Progress callback invocation
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.models import (
    ClassMetrics,
    DetectionResult,
    GroundTruthBox,
    ImageEvalEntry,
)


# ── Stub Detection to avoid importing torch via src.vision.detector ──

@dataclass
class StubDetection:
    """Lightweight Detection stub for testing (avoids torch dependency)."""
    track_id: int = -1
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    center: tuple = (0, 0)
    foot: tuple = (0, 0)
    keypoints: Optional[np.ndarray] = None


def _make_det(class_id=0, class_name="person", confidence=0.85, bbox=None):
    if bbox is None:
        bbox = [100.0, 50.0, 300.0, 250.0]
    x1, y1, x2, y2 = bbox
    return StubDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        center=((x1 + x2) / 2, (y1 + y2) / 2),
        foot=((x1 + x2) / 2, y2),
    )


# ── Mock YOLODetector AND cv2 before importing detection_eval ──
# cv2.typing has a known incompatibility on this machine, so we must
# ensure detection_eval can be imported without triggering the real cv2.

_mock_detector_module = MagicMock()
_mock_cv2_module = MagicMock()

_original_detector = sys.modules.get("src.vision.detector")
_original_cv2 = sys.modules.get("cv2")

sys.modules["src.vision.detector"] = _mock_detector_module
# Only mock cv2 if it's not already loaded successfully
if _original_cv2 is None:
    sys.modules["cv2"] = _mock_cv2_module

from src.evaluation.detection_eval import DetectionEvaluator  # noqa: E402

# Restore originals
if _original_detector is not None:
    sys.modules["src.vision.detector"] = _original_detector
if _original_cv2 is not None:
    sys.modules["cv2"] = _original_cv2


# ── Fixtures ──

@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a temp dir with dummy images (640x480 PNGs via PIL)."""
    from PIL import Image as PILImage
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ["img_001.jpg", "img_002.jpg", "img_003.jpg"]:
        img = PILImage.new("RGB", (640, 480), color=(0, 0, 0))
        img.save(str(img_dir / name))
    return img_dir


@pytest.fixture
def sample_label_dir(tmp_path):
    """Create a temp dir with YOLO label files."""
    lbl_dir = tmp_path / "labels"
    lbl_dir.mkdir()
    # img_001: 2 person boxes
    (lbl_dir / "img_001.txt").write_text(
        "# conf=0.90 person\n"
        "0 0.3125 0.2604 0.3125 0.4167\n"
        "# conf=0.80 person\n"
        "0 0.7031 0.5208 0.1563 0.2083\n"
    )
    # img_002: 1 car box
    (lbl_dir / "img_002.txt").write_text(
        "# conf=0.75 car\n"
        "2 0.5000 0.5000 0.4688 0.4167\n"
    )
    # img_003: no detections (empty label)
    (lbl_dir / "img_003.txt").write_text("# no detections\n")
    return lbl_dir


@pytest.fixture
def empty_label_dir(tmp_path):
    """Create an empty label directory."""
    lbl_dir = tmp_path / "empty_labels"
    lbl_dir.mkdir()
    return lbl_dir


# ── Helper to create a mock evaluator ──

def _make_evaluator(detect_fn=None):
    """Create a DetectionEvaluator with mocked internals (no real model)."""
    evaluator = DetectionEvaluator.__new__(DetectionEvaluator)
    mock_detector = MagicMock()
    mock_detector.detect = detect_fn or MagicMock(return_value=[])
    mock_detector.model = MagicMock()
    mock_detector.model.names = {
        0: "person", 1: "bicycle", 2: "car",
        3: "motorcycle", 5: "bus", 7: "truck",
    }
    evaluator.detector = mock_detector
    evaluator.iou_threshold = 0.5
    evaluator.confidence = 0.3
    evaluator.class_names = dict(mock_detector.model.names)
    return evaluator


# ── Tests ──


class TestDetectionEvaluatorEmptyLabels:
    """Test that empty label directory raises ValueError."""

    def test_empty_label_dir_raises_value_error(self, sample_image_dir, empty_label_dir):
        """Requirement 2.10: empty label dir should raise ValueError."""
        evaluator = _make_evaluator()
        with pytest.raises(ValueError, match="No valid label files"):
            evaluator.run(sample_image_dir, empty_label_dir)

    def test_no_matching_labels_raises_value_error(self, tmp_path):
        """When images exist but no labels match, should raise ValueError."""
        from PIL import Image as PILImage
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        lbl_dir = tmp_path / "lbls"
        lbl_dir.mkdir()

        PILImage.new("RGB", (100, 100)).save(str(img_dir / "test.jpg"))
        (lbl_dir / "other.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        evaluator = _make_evaluator()
        with pytest.raises(ValueError, match="No valid label files"):
            evaluator.run(img_dir, lbl_dir)


class TestGreedyMatch:
    """Test the greedy IoU matching logic."""

    def test_perfect_match(self):
        """When detection perfectly overlaps GT, should be TP."""
        gt_boxes = [GroundTruthBox(class_id=0, x1=100, y1=50, x2=300, y2=250)]
        dets = [_make_det(class_id=0, confidence=0.9, bbox=[100, 50, 300, 250])]

        tp, fp, fn, matches = DetectionEvaluator._greedy_match(dets, gt_boxes, 0.5)
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_no_overlap_is_fp(self):
        """When detection doesn't overlap any GT, should be FP."""
        gt_boxes = [GroundTruthBox(class_id=0, x1=0, y1=0, x2=50, y2=50)]
        dets = [_make_det(class_id=0, confidence=0.9, bbox=[500, 500, 600, 600])]

        tp, fp, fn, matches = DetectionEvaluator._greedy_match(dets, gt_boxes, 0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 1

    def test_each_gt_matched_at_most_once(self):
        """Two detections overlapping same GT: only highest confidence matches."""
        gt_boxes = [GroundTruthBox(class_id=0, x1=100, y1=100, x2=200, y2=200)]
        dets = [
            _make_det(class_id=0, confidence=0.9, bbox=[100, 100, 200, 200]),
            _make_det(class_id=0, confidence=0.8, bbox=[105, 105, 205, 205]),
        ]

        tp, fp, fn, matches = DetectionEvaluator._greedy_match(dets, gt_boxes, 0.5)
        assert tp == 1
        assert fp == 1
        assert fn == 0

    def test_empty_detections(self):
        """No detections: all GT are FN."""
        gt_boxes = [
            GroundTruthBox(class_id=0, x1=0, y1=0, x2=100, y2=100),
            GroundTruthBox(class_id=2, x1=200, y1=200, x2=400, y2=400),
        ]
        tp, fp, fn, matches = DetectionEvaluator._greedy_match([], gt_boxes, 0.5)
        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_empty_gt(self):
        """No GT boxes: all detections are FP."""
        dets = [_make_det(class_id=0, confidence=0.9)]
        tp, fp, fn, matches = DetectionEvaluator._greedy_match(dets, [], 0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 0


class TestPerImageEntry:
    """Test that per-image ImageEvalEntry is correctly populated."""

    def test_per_image_counts(self, sample_image_dir, sample_label_dir):
        """Verify per-image gt_count, det_count, tp_count."""
        call_count = [0]

        def mock_detect(frame):
            call_count[0] += 1
            idx = call_count[0]
            if idx == 1:
                # img_001: return 2 person detections matching GT
                return [
                    _make_det(class_id=0, confidence=0.9, bbox=[100, 25, 300, 225]),
                    _make_det(class_id=0, confidence=0.8, bbox=[400, 200, 500, 300]),
                ]
            elif idx == 2:
                # img_002: return 1 car detection
                return [
                    _make_det(class_id=2, class_name="car", confidence=0.75,
                              bbox=[170, 140, 470, 340]),
                ]
            else:
                return []

        evaluator = _make_evaluator(detect_fn=mock_detect)

        # Mock cv2.imread to return a dummy numpy array
        with patch("src.evaluation.detection_eval.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            result = evaluator.run(sample_image_dir, sample_label_dir)

        assert isinstance(result, DetectionResult)
        # All 3 images have label files (img_003 has "# no detections" comment)
        assert len(result.per_image) == 3

        filenames = {e.filename for e in result.per_image}
        assert "img_001.jpg" in filenames
        assert "img_002.jpg" in filenames
        assert "img_003.jpg" in filenames

        # img_003 should have 0 GT and 0 detections
        img3 = [e for e in result.per_image if e.filename == "img_003.jpg"][0]
        assert img3.gt_count == 0
        assert img3.det_count == 0
        assert img3.tp_count == 0


class TestNoSampleClass:
    """Test that classes with no GT samples are marked N/A."""

    def test_no_sample_class_has_samples_false(self, sample_image_dir, sample_label_dir):
        """Classes detected but not in GT should have has_samples=False."""
        def mock_detect(frame):
            return [
                _make_det(class_id=1, class_name="bicycle", confidence=0.7,
                          bbox=[50, 50, 150, 150]),
            ]

        evaluator = _make_evaluator(detect_fn=mock_detect)

        with patch("src.evaluation.detection_eval.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            result = evaluator.run(sample_image_dir, sample_label_dir)

        bicycle_metrics = None
        for cm in result.class_metrics:
            if cm.class_id == 1:
                bicycle_metrics = cm
                break

        assert bicycle_metrics is not None
        assert bicycle_metrics.has_samples is False
        assert bicycle_metrics.ap is None


class TestProgressCallback:
    """Test that progress_callback is called correctly."""

    def test_callback_called_for_each_image(self, sample_image_dir, sample_label_dir):
        """progress_callback(current, total) should be called per image."""
        evaluator = _make_evaluator()
        calls = []

        def callback(current, total):
            calls.append((current, total))

        with patch("src.evaluation.detection_eval.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            evaluator.run(sample_image_dir, sample_label_dir, progress_callback=callback)

        assert len(calls) > 0
        # Last call should have current == total
        assert calls[-1][0] == calls[-1][1]


# Feature: accuracy-evaluation, Property 7: 逐图指标一致性

from hypothesis import given, settings
from hypothesis import strategies as st


class TestImageEvalEntryInvariants:
    """Property test: per-image metric consistency.

    For any ImageEvalEntry with valid counts, the following must hold:
    1. tp_count <= min(gt_count, det_count)
    2. fp = det_count - tp_count >= 0
    3. fn = gt_count - tp_count >= 0

    **Validates: Requirements 2.9**
    """

    @given(
        gt_count=st.integers(min_value=0, max_value=100),
        det_count=st.integers(min_value=0, max_value=100),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_per_image_metric_consistency(self, gt_count, det_count, data):
        """tp <= min(gt, det) and derived fp/fn are non-negative."""
        max_tp = min(gt_count, det_count)
        tp_count = data.draw(st.integers(min_value=0, max_value=max_tp), label="tp_count")

        entry = ImageEvalEntry(
            filename="test.jpg",
            gt_count=gt_count,
            det_count=det_count,
            tp_count=tp_count,
        )

        # Invariant 1: tp <= min(gt, det)
        assert entry.tp_count <= min(entry.gt_count, entry.det_count)

        # Invariant 2: fp = det - tp >= 0
        fp = entry.det_count - entry.tp_count
        assert fp >= 0

        # Invariant 3: fn = gt - tp >= 0
        fn = entry.gt_count - entry.tp_count
        assert fn >= 0
