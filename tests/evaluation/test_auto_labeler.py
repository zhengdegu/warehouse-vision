"""Unit tests for AutoLabeler."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.models import LabelSummary


# ── Stub Detection to avoid importing torch via src.vision.detector ──

@dataclass
class Detection:
    """Lightweight Detection stub for testing (avoids torch dependency)."""
    track_id: int = -1
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    center: tuple = (0, 0)
    foot: tuple = (0, 0)
    keypoints: Optional[np.ndarray] = None


def _make_detection(class_id=0, class_name="person", confidence=0.85,
                    bbox=None):
    """Helper to create a Detection with sensible defaults."""
    if bbox is None:
        bbox = [100, 50, 300, 250]
    x1, y1, x2, y2 = bbox
    return Detection(
        track_id=-1,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        center=((x1 + x2) / 2, (y1 + y2) / 2),
        foot=((x1 + x2) / 2, y2),
    )


# ── Mock the detector module before importing auto_labeler ──

_mock_detector_module = MagicMock()
_mock_detector_module.YOLODetector = MagicMock()
_mock_detector_module.Detection = Detection
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())
# Patch the vision.detector module so auto_labeler can import
_original = sys.modules.get("src.vision.detector")
sys.modules["src.vision.detector"] = _mock_detector_module

from src.evaluation.auto_labeler import AutoLabeler  # noqa: E402

# Restore original if it existed
if _original is not None:
    sys.modules["src.vision.detector"] = _original


def _create_test_image(path: Path, width=640, height=480):
    """Create a minimal valid image file using cv2."""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestAutoLabelerInit:
    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_creates_detector(self, mock_cls):
        labeler = AutoLabeler("model.pt", confidence=0.4, classes=[0, 2])
        mock_cls.assert_called_once_with(
            model_path="model.pt",
            confidence=0.4,
            allowed_classes=[0, 2],
        )
        assert labeler.detector is mock_cls.return_value


class TestLabelSingle:
    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_no_detections_writes_comment(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_cls.return_value = mock_detector

        labeler = AutoLabeler("model.pt")
        img_path = tmp_path / "img.jpg"
        _create_test_image(img_path)
        label_path = tmp_path / "img.txt"

        counts = labeler._label_single(img_path, label_path)

        assert label_path.exists()
        content = label_path.read_text()
        assert "# no detections" in content
        assert counts == {}

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_single_detection_writes_comment_and_yolo_line(self, mock_cls, tmp_path):
        det = _make_detection(class_id=0, class_name="person",
                              confidence=0.92, bbox=[100, 50, 300, 250])
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [det]
        mock_cls.return_value = mock_detector

        labeler = AutoLabeler("model.pt")
        img_path = tmp_path / "img.jpg"
        _create_test_image(img_path, width=640, height=480)
        label_path = tmp_path / "img.txt"

        counts = labeler._label_single(img_path, label_path)

        content = label_path.read_text()
        lines = [l for l in content.strip().split("\n") if l]
        # First line: comment with conf and class_name
        assert lines[0].startswith("# conf=0.92")
        assert "person" in lines[0]
        # Second line: YOLO format
        parts = lines[1].split()
        assert parts[0] == "0"  # class_id
        assert len(parts) == 5
        assert counts == {"person": 1}

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_multiple_detections(self, mock_cls, tmp_path):
        dets = [
            _make_detection(class_id=0, class_name="person", confidence=0.9,
                            bbox=[10, 10, 100, 200]),
            _make_detection(class_id=2, class_name="car", confidence=0.75,
                            bbox=[200, 100, 400, 300]),
        ]
        mock_detector = MagicMock()
        mock_detector.detect.return_value = dets
        mock_cls.return_value = mock_detector

        labeler = AutoLabeler("model.pt")
        img_path = tmp_path / "img.png"
        _create_test_image(img_path)
        label_path = tmp_path / "img.txt"

        counts = labeler._label_single(img_path, label_path)

        content = label_path.read_text()
        lines = [l for l in content.strip().split("\n") if l]
        # 2 detections × 2 lines each (comment + data)
        assert len(lines) == 4
        assert counts == {"person": 1, "car": 1}

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_unreadable_image_writes_no_detections(self, mock_cls, tmp_path):
        mock_cls.return_value = MagicMock()

        labeler = AutoLabeler("model.pt")
        img_path = tmp_path / "bad.jpg"
        img_path.write_text("not an image")
        label_path = tmp_path / "bad.txt"

        counts = labeler._label_single(img_path, label_path)

        assert "# no detections" in label_path.read_text()
        assert counts == {}


class TestLabelDirectory:
    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_labels_all_images(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            _make_detection(class_id=0, class_name="person", confidence=0.8)
        ]
        mock_cls.return_value = mock_detector

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "labels"

        for name in ["a.jpg", "b.png", "c.jpeg"]:
            _create_test_image(img_dir / name)

        labeler = AutoLabeler("model.pt")
        summary = labeler.label_directory(img_dir, lbl_dir)

        assert isinstance(summary, LabelSummary)
        assert summary.total_images == 3
        assert summary.labeled == 3
        assert summary.skipped == 0
        assert lbl_dir.exists()
        assert (lbl_dir / "a.txt").exists()
        assert (lbl_dir / "b.txt").exists()
        assert (lbl_dir / "c.txt").exists()

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_skips_existing_labels(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            _make_detection()
        ]
        mock_cls.return_value = mock_detector

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "labels"
        lbl_dir.mkdir()

        _create_test_image(img_dir / "a.jpg")
        _create_test_image(img_dir / "b.jpg")
        # Pre-existing label for 'a'
        (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        labeler = AutoLabeler("model.pt")
        summary = labeler.label_directory(img_dir, lbl_dir)

        assert summary.total_images == 2
        assert summary.skipped == 1
        assert summary.labeled == 1
        # 'a.txt' should not be overwritten
        assert "0 0.5 0.5 0.1 0.1" in (lbl_dir / "a.txt").read_text()

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_overwrite_replaces_existing(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_cls.return_value = mock_detector

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "labels"
        lbl_dir.mkdir()

        _create_test_image(img_dir / "a.jpg")
        (lbl_dir / "a.txt").write_text("old content\n")

        labeler = AutoLabeler("model.pt")
        summary = labeler.label_directory(img_dir, lbl_dir, overwrite=True)

        assert summary.skipped == 0
        assert summary.labeled == 1
        assert "# no detections" in (lbl_dir / "a.txt").read_text()

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_creates_label_dir(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_cls.return_value = mock_detector

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "nested" / "labels"

        _create_test_image(img_dir / "a.jpg")

        labeler = AutoLabeler("model.pt")
        labeler.label_directory(img_dir, lbl_dir)

        assert lbl_dir.exists()

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_ignores_non_image_files(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_cls.return_value = mock_detector

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "labels"

        _create_test_image(img_dir / "a.jpg")
        (img_dir / "readme.txt").write_text("not an image")
        (img_dir / "data.csv").write_text("1,2,3")

        labeler = AutoLabeler("model.pt")
        summary = labeler.label_directory(img_dir, lbl_dir)

        assert summary.total_images == 1

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_class_counts_aggregated(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            _make_detection(class_id=0, class_name="person"),
            _make_detection(class_id=2, class_name="car"),
        ]
        mock_cls.return_value = mock_detector

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "labels"

        for name in ["a.jpg", "b.jpg"]:
            _create_test_image(img_dir / name)

        labeler = AutoLabeler("model.pt")
        summary = labeler.label_directory(img_dir, lbl_dir)

        assert summary.class_counts["person"] == 2
        assert summary.class_counts["car"] == 2

    @patch("src.evaluation.auto_labeler.YOLODetector")
    def test_total_equals_skipped_plus_labeled(self, mock_cls, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [_make_detection()]
        mock_cls.return_value = mock_detector

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "labels"
        lbl_dir.mkdir()

        _create_test_image(img_dir / "a.jpg")
        _create_test_image(img_dir / "b.jpg")
        (lbl_dir / "a.txt").write_text("existing\n")

        labeler = AutoLabeler("model.pt")
        summary = labeler.label_directory(img_dir, lbl_dir)

        assert summary.total_images == summary.skipped + summary.labeled


# ── Tests for scripts/auto_label.py entry point ──

from scripts.auto_label import parse_args, main


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.images == "data/samples/warehouse_v1"
        assert args.labels == "data/labels/warehouse_v1"
        assert args.confidence == pytest.approx(0.3)
        assert args.overwrite is False

    def test_custom_args(self):
        args = parse_args([
            "--model", "custom.pt",
            "--confidence", "0.6",
            "--classes", "0,2",
            "--overwrite",
            "my/images",
            "my/labels",
        ])
        assert args.model == "custom.pt"
        assert args.confidence == pytest.approx(0.6)
        assert args.classes == "0,2"
        assert args.overwrite is True
        assert args.images == "my/images"
        assert args.labels == "my/labels"

    def test_partial_positional(self):
        args = parse_args(["some/dir"])
        assert args.images == "some/dir"
        assert args.labels == "data/labels/warehouse_v1"


class TestMainEntryPoint:
    def test_missing_image_dir_returns_1(self, tmp_path):
        rc = main(["--model", "m.pt", str(tmp_path / "nonexistent")])
        assert rc == 1

    @patch("scripts.auto_label.AutoLabeler")
    def test_success_returns_0(self, mock_labeler_cls, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        lbl_dir = tmp_path / "lbls"

        mock_instance = MagicMock()
        mock_instance.label_directory.return_value = LabelSummary(
            total_images=2, skipped=1, labeled=1,
            class_counts={"person": 3},
        )
        mock_labeler_cls.return_value = mock_instance

        rc = main([
            "--model", "m.pt",
            "--confidence", "0.5",
            "--classes", "0,2",
            str(img_dir),
            str(lbl_dir),
        ])
        assert rc == 0
        mock_labeler_cls.assert_called_once_with(
            model_path="m.pt", confidence=0.5, classes=[0, 2],
        )
        mock_instance.label_directory.assert_called_once()


# ── Property-Based Tests ──

import re
from hypothesis import given, settings
from hypothesis import strategies as st


def _detection_strategy():
    """Strategy to generate a random Detection with valid fields."""
    return st.builds(
        Detection,
        track_id=st.just(-1),
        class_id=st.integers(min_value=0, max_value=79),
        class_name=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu")),
            min_size=2,
            max_size=10,
        ),
        confidence=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        bbox=st.tuples(
            st.integers(min_value=0, max_value=500),
            st.integers(min_value=0, max_value=340),
            st.integers(min_value=0, max_value=500),
            st.integers(min_value=0, max_value=340),
        ).map(lambda t: [min(t[0], t[2]), min(t[1], t[3]),
                         max(t[0], t[2]) + 1, max(t[1], t[3]) + 1])
        .filter(lambda b: b[2] <= 640 and b[3] <= 480),
        center=st.just((0, 0)),
        foot=st.just((0, 0)),
        keypoints=st.just(None),
    )


# Feature: accuracy-evaluation, Property 2: 标注文件注释行包含检测元数据
class TestProperty2CommentLineMetadata:
    """Property 2: 标注文件注释行包含检测元数据

    **Validates: Requirements 1.3**
    """

    @given(detections=st.lists(_detection_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_comment_lines_contain_confidence_and_class_name(
        self, detections, tmp_path_factory
    ):
        """For each detection, the output file should contain a comment line
        with conf=X.XX matching the confidence and the class_name string."""
        tmp_path = tmp_path_factory.mktemp("prop2")

        with patch("src.evaluation.auto_labeler.YOLODetector") as mock_cls:
            # Setup mock detector
            mock_detector = MagicMock()
            mock_detector.detect.return_value = detections
            mock_cls.return_value = mock_detector

            # Create a test image
            img_path = tmp_path / "test.jpg"
            _create_test_image(img_path, width=640, height=480)
            label_path = tmp_path / "test.txt"

            labeler = AutoLabeler("model.pt")
            labeler._label_single(img_path, label_path)

        content = label_path.read_text(encoding="utf-8")
        lines = [l for l in content.strip().split("\n") if l]

        # Should have 2 lines per detection (comment + YOLO data)
        assert len(lines) == len(detections) * 2

        for i, det in enumerate(detections):
            comment_line = lines[i * 2]
            # Comment line must start with #
            assert comment_line.startswith("#"), (
                f"Expected comment line, got: {comment_line}"
            )
            # Must contain conf= with the formatted confidence value
            expected_conf = f"conf={det.confidence:.2f}"
            assert expected_conf in comment_line, (
                f"Expected '{expected_conf}' in '{comment_line}'"
            )
            # Must contain the class name
            assert det.class_name in comment_line, (
                f"Expected class_name '{det.class_name}' in '{comment_line}'"
            )


# Feature: accuracy-evaluation, Property 3: 预标注统计摘要一致性
class TestProperty3LabelSummaryConsistency:
    """Property 3: 预标注统计摘要一致性

    For any set of images and detection results, LabelSummary must satisfy:
    - total_images == skipped + labeled
    - sum(class_counts.values()) == total detections across all labeled images

    **Validates: Requirements 1.5**
    """

    @given(
        num_images=st.integers(min_value=1, max_value=10),
        num_pre_labeled=st.integers(min_value=0, max_value=10),
        detections_per_image=st.lists(
            st.lists(
                _detection_strategy(),
                min_size=0,
                max_size=5,
            ),
            min_size=1,
            max_size=10,
        ),
        overwrite=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_summary_consistency(
        self,
        num_images,
        num_pre_labeled,
        detections_per_image,
        overwrite,
        tmp_path_factory,
    ):
        """total_images == skipped + labeled, and class_counts sum equals
        total detections across all newly labeled images."""
        tmp_path = tmp_path_factory.mktemp("prop3")

        # Clamp pre-labeled count to not exceed image count
        num_pre_labeled = min(num_pre_labeled, num_images)

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        lbl_dir = tmp_path / "labels"
        lbl_dir.mkdir()

        # Create image files
        image_names = [f"img_{i:04d}.jpg" for i in range(num_images)]
        for name in image_names:
            _create_test_image(img_dir / name, width=640, height=480)

        # Pre-create some label files to test skip behavior
        pre_labeled_names = image_names[:num_pre_labeled]
        for name in pre_labeled_names:
            stem = Path(name).stem
            (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Build a detection list that cycles through detections_per_image
        # for each image that will actually be labeled
        call_count = [0]

        def mock_detect(frame):
            idx = call_count[0] % len(detections_per_image)
            call_count[0] += 1
            return detections_per_image[idx]

        with patch("src.evaluation.auto_labeler.YOLODetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detect.side_effect = mock_detect
            mock_cls.return_value = mock_detector

            labeler = AutoLabeler("model.pt")
            summary = labeler.label_directory(img_dir, lbl_dir, overwrite=overwrite)

        # Property 1: total_images == skipped + labeled
        assert summary.total_images == summary.skipped + summary.labeled, (
            f"total_images={summary.total_images} != "
            f"skipped={summary.skipped} + labeled={summary.labeled}"
        )

        # Property 2: class_counts sum == total detections across labeled images
        # Calculate expected total detections from mock calls
        expected_total_detections = 0
        num_labeled = summary.labeled
        for i in range(num_labeled):
            idx = i % len(detections_per_image)
            expected_total_detections += len(detections_per_image[idx])

        actual_total = sum(summary.class_counts.values())
        assert actual_total == expected_total_detections, (
            f"class_counts sum={actual_total} != "
            f"expected detections={expected_total_detections}"
        )
