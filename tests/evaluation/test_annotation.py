"""Unit tests for annotation loading and validation."""

import json
from pathlib import Path

import pytest

from src.evaluation.annotation import (
    AnnotationLoader,
    ValidationError,
    match_files,
    parse_yolo_line,
    pixel_to_yolo,
    serialize_yolo_line,
)
from src.evaluation.models import GroundTruthBox


# ── serialize_yolo_line ──


class TestSerializeYoloLine:
    def test_basic(self):
        result = serialize_yolo_line(0, 0.5, 0.5, 0.2, 0.3)
        assert result == "0 0.500000 0.500000 0.200000 0.300000"

    def test_different_class(self):
        result = serialize_yolo_line(3, 0.1, 0.9, 0.05, 0.1)
        assert result.startswith("3 ")


# ── parse_yolo_line ──


class TestParseYoloLine:
    def test_basic_center(self):
        box = parse_yolo_line("0 0.5 0.5 0.2 0.3", 100, 200)
        assert box.class_id == 0
        assert box.x1 == pytest.approx(40.0)
        assert box.y1 == pytest.approx(70.0)
        assert box.x2 == pytest.approx(60.0)
        assert box.y2 == pytest.approx(130.0)

    def test_full_image_box(self):
        box = parse_yolo_line("1 0.5 0.5 1.0 1.0", 640, 480)
        assert box.x1 == pytest.approx(0.0)
        assert box.y1 == pytest.approx(0.0)
        assert box.x2 == pytest.approx(640.0)
        assert box.y2 == pytest.approx(480.0)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Expected 5 values"):
            parse_yolo_line("0 0.5 0.5", 100, 100)


# ── pixel_to_yolo ──


class TestPixelToYolo:
    def test_basic(self):
        result = pixel_to_yolo(0, 10.0, 20.0, 30.0, 60.0, 100, 200)
        # cx = 20/100 = 0.2, cy = 40/200 = 0.2, w = 20/100 = 0.2, h = 40/200 = 0.2
        assert "0 0.200000 0.200000 0.200000 0.200000" == result


# ── match_files ──


class TestMatchFiles:
    def test_matching(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        (img_dir / "img001.jpg").write_text("")
        (img_dir / "img002.png").write_text("")
        (lbl_dir / "img001.txt").write_text("0 0.5 0.5 0.1 0.1")
        # img002 has no label

        matched, unmatched = match_files(img_dir, lbl_dir)
        assert len(matched) == 1
        assert matched[0][0].stem == "img001"
        assert len(unmatched) == 1
        assert unmatched[0].stem == "img002"

    def test_empty_dirs(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        matched, unmatched = match_files(img_dir, lbl_dir)
        assert matched == []
        assert unmatched == []

    def test_non_image_files_ignored(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        (img_dir / "readme.txt").write_text("not an image")
        (lbl_dir / "readme.txt").write_text("0 0.5 0.5 0.1 0.1")

        matched, unmatched = match_files(img_dir, lbl_dir)
        assert matched == []
        assert unmatched == []


# ── AnnotationLoader.load_yolo_labels ──


class TestLoadYoloLabels:
    def _create_test_image(self, path: Path, width: int = 100, height: int = 200):
        """Create a minimal valid image file using PIL."""
        from PIL import Image
        img = Image.new("RGB", (width, height), color="black")
        img.save(path)

    def test_basic_loading(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        self._create_test_image(img_dir / "test.jpg")
        (lbl_dir / "test.txt").write_text("0 0.5 0.5 0.2 0.3\n")

        labels, skipped = AnnotationLoader.load_yolo_labels(lbl_dir, img_dir)
        assert "test" in labels
        assert len(labels["test"]) == 1
        assert labels["test"][0].class_id == 0
        assert skipped == []

    def test_comment_lines_ignored(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        self._create_test_image(img_dir / "test.jpg")
        (lbl_dir / "test.txt").write_text(
            "# conf=0.85 person\n0 0.5 0.5 0.2 0.3\n# no detections\n"
        )

        labels, skipped = AnnotationLoader.load_yolo_labels(lbl_dir, img_dir)
        assert len(labels["test"]) == 1

    def test_empty_label_file(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        self._create_test_image(img_dir / "test.jpg")
        (lbl_dir / "test.txt").write_text("# no detections\n")

        labels, skipped = AnnotationLoader.load_yolo_labels(lbl_dir, img_dir)
        assert "test" in labels
        assert labels["test"] == []

    def test_skipped_images(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        self._create_test_image(img_dir / "has_label.jpg")
        self._create_test_image(img_dir / "no_label.jpg")
        (lbl_dir / "has_label.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        labels, skipped = AnnotationLoader.load_yolo_labels(lbl_dir, img_dir)
        assert "has_label" in labels
        assert "no_label.jpg" in skipped


# ── AnnotationLoader.validate_video_annotation ──


class TestValidateVideoAnnotation:
    def test_valid_annotation(self):
        data = {
            "video_path": "clip.mp4",
            "events": [
                {"type": "intrusion", "time_range": [1.0, 5.0]},
            ],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert errors == []

    def test_missing_video_path(self):
        data = {"events": []}
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("video_path" in e for e in errors)

    def test_missing_events(self):
        data = {"video_path": "clip.mp4"}
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("events" in e for e in errors)

    def test_invalid_event_type(self):
        data = {
            "video_path": "clip.mp4",
            "events": [{"type": "unknown_type"}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("invalid event type" in e for e in errors)

    def test_anomaly_missing_sub_type(self):
        data = {
            "video_path": "clip.mp4",
            "events": [{"type": "anomaly"}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("sub_type" in e for e in errors)

    def test_anomaly_invalid_sub_type(self):
        data = {
            "video_path": "clip.mp4",
            "events": [{"type": "anomaly", "sub_type": "invalid"}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("invalid anomaly sub_type" in e for e in errors)

    def test_tripwire_missing_id(self):
        data = {
            "video_path": "clip.mp4",
            "events": [{"type": "tripwire"}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("tripwire_id" in e for e in errors)

    def test_tripwire_invalid_id_reference(self):
        data = {
            "video_path": "clip.mp4",
            "camera_config": {
                "tripwires": [{"id": "tw01", "p1": [0, 0], "p2": [100, 100]}],
            },
            "events": [{"type": "tripwire", "tripwire_id": "tw99"}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("tw99" in e and "not found" in e for e in errors)

    def test_tripwire_valid_id_reference(self):
        data = {
            "video_path": "clip.mp4",
            "camera_config": {
                "tripwires": [{"id": "tw01", "p1": [0, 0], "p2": [100, 100]}],
            },
            "events": [{"type": "tripwire", "tripwire_id": "tw01"}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert errors == []

    def test_time_range_start_ge_end(self):
        data = {
            "video_path": "clip.mp4",
            "events": [{"type": "intrusion", "time_range": [10.0, 5.0]}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("start" in e and "less than" in e for e in errors)

    def test_time_range_equal(self):
        data = {
            "video_path": "clip.mp4",
            "events": [{"type": "intrusion", "time_range": [5.0, 5.0]}],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert any("start" in e and "less than" in e for e in errors)

    def test_multiple_errors(self):
        data = {
            "events": [
                {"type": "bad_type"},
                {"type": "anomaly"},
                {"type": "intrusion", "time_range": [10.0, 2.0]},
            ],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        # Missing video_path + invalid type + missing sub_type + bad time_range
        assert len(errors) >= 3


# ── AnnotationLoader.load_video_annotation ──


class TestLoadVideoAnnotation:
    def test_valid_json(self, tmp_path):
        data = {
            "video_path": "clip.mp4",
            "fps_sample": 10,
            "camera_config": {"roi": [[0, 0], [100, 100]]},
            "events": [
                {"type": "intrusion", "time_range": [1.0, 5.0]},
            ],
        }
        json_path = tmp_path / "clip.json"
        json_path.write_text(json.dumps(data))

        annotation = AnnotationLoader.load_video_annotation(json_path)
        assert annotation.video_path == "clip.mp4"
        assert annotation.fps_sample == 10
        assert len(annotation.events) == 1

    def test_invalid_json_raises_validation_error(self, tmp_path):
        data = {"events": [{"type": "bad"}]}
        json_path = tmp_path / "bad.json"
        json_path.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as exc_info:
            AnnotationLoader.load_video_annotation(json_path)
        assert len(exc_info.value.errors) > 0

    def test_defaults(self, tmp_path):
        data = {
            "video_path": "clip.mp4",
            "events": [],
        }
        json_path = tmp_path / "clip.json"
        json_path.write_text(json.dumps(data))

        annotation = AnnotationLoader.load_video_annotation(json_path)
        assert annotation.fps_sample == 5
        assert annotation.camera_config == {}


# Feature: accuracy-evaluation, Property 1: YOLO 标注格式往返一致性
# Validates: Requirements 1.2, 2.2

from hypothesis import given, settings
from hypothesis import strategies as st


class TestYoloRoundTripProperty:
    """Property test: pixel_to_yolo → parse_yolo_line round-trip consistency.

    For any valid bounding box in pixel coordinates, serializing to YOLO format
    and parsing back should yield approximately the same pixel coordinates
    (within floating-point tolerance due to 6-decimal precision).

    **Validates: Requirements 1.2, 2.2**
    """

    @given(
        class_id=st.integers(min_value=0, max_value=79),
        img_width=st.integers(min_value=10, max_value=2000),
        img_height=st.integers(min_value=10, max_value=2000),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_yolo_round_trip(self, class_id, img_width, img_height, data):
        """pixel_to_yolo → parse_yolo_line should recover original pixel coords."""
        # Generate x1 < x2 within [0, img_width], y1 < y2 within [0, img_height]
        # Ensure minimum 1-pixel box to avoid degenerate cases
        x1 = data.draw(st.floats(min_value=0.0, max_value=float(img_width - 1),
                                 allow_nan=False, allow_infinity=False))
        x2 = data.draw(st.floats(min_value=x1 + 1.0, max_value=float(img_width),
                                 allow_nan=False, allow_infinity=False))
        y1 = data.draw(st.floats(min_value=0.0, max_value=float(img_height - 1),
                                 allow_nan=False, allow_infinity=False))
        y2 = data.draw(st.floats(min_value=y1 + 1.0, max_value=float(img_height),
                                 allow_nan=False, allow_infinity=False))

        # Serialize to YOLO format
        yolo_line = pixel_to_yolo(class_id, x1, y1, x2, y2, img_width, img_height)

        # Parse back to pixel coordinates
        box = parse_yolo_line(yolo_line, img_width, img_height)

        # Verify round-trip consistency within tolerance
        # 6-decimal YOLO precision → max error ≈ 0.5e-6 * max_dim ≈ 0.001 px
        tol = 0.01
        assert box.class_id == class_id
        assert abs(box.x1 - x1) < tol, f"x1: {box.x1} vs {x1}"
        assert abs(box.y1 - y1) < tol, f"y1: {box.y1} vs {y1}"
        assert abs(box.x2 - x2) < tol, f"x2: {box.x2} vs {x2}"
        assert abs(box.y2 - y2) < tol, f"y2: {box.y2} vs {y2}"


# Feature: accuracy-evaluation, Property 4: 文件名匹配正确性
# Validates: Requirements 2.1, 4.1

import tempfile


class TestFileMatchProperty:
    """Property test: match_files correctness.

    For any set of image and label files created from random filename stems,
    match_files should satisfy:
    1. Every matched pair has the same filename stem
    2. Unmatched images have no corresponding label file
    3. matched_count + unmatched_count = total_image_count (conservation)

    **Validates: Requirements 2.1, 4.1**
    """

    # Strategy: unique alphanumeric stems, 3-10 chars
    stem_strategy = st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
        min_size=3,
        max_size=10,
    )

    image_ext_strategy = st.sampled_from([".jpg", ".png", ".bmp"])

    # For each stem: 'image_only', 'label_only', or 'both'
    presence_strategy = st.sampled_from(["image_only", "label_only", "both"])

    @given(
        stems=st.lists(
            stem_strategy,
            min_size=0,
            max_size=20,
            unique=True,
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_file_match_correctness(self, stems, data):
        """match_files returns correct matched pairs and unmatched images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            img_dir = base / "images"
            lbl_dir = base / "labels"
            img_dir.mkdir()
            lbl_dir.mkdir()

            for stem in stems:
                presence = data.draw(self.presence_strategy)
                ext = data.draw(self.image_ext_strategy)

                if presence in ("image_only", "both"):
                    (img_dir / f"{stem}{ext}").write_text("")

                if presence in ("label_only", "both"):
                    (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1")

            matched, unmatched = match_files(img_dir, lbl_dir)

            matched_stems = {img.stem for img, lbl in matched}
            unmatched_stems = {img.stem for img in unmatched}

            # Property 1: Every matched pair has the same filename stem
            for img_path, lbl_path in matched:
                assert img_path.stem == lbl_path.stem, (
                    f"Matched pair stems differ: {img_path.stem} vs {lbl_path.stem}"
                )

            # Property 2: Unmatched images have no corresponding label file
            for img_path in unmatched:
                label_candidate = lbl_dir / f"{img_path.stem}.txt"
                assert not label_candidate.is_file(), (
                    f"Unmatched image {img_path.name} has a label file {label_candidate}"
                )

            # Property 3: matched_count + unmatched_count = total_image_count
            total_images = len(list(
                p for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ))
            assert len(matched) + len(unmatched) == total_images, (
                f"Conservation violated: {len(matched)} matched + {len(unmatched)} "
                f"unmatched != {total_images} total images"
            )

            # Additional: no overlap between matched and unmatched stems
            assert matched_stems.isdisjoint(unmatched_stems), (
                f"Overlap between matched and unmatched: {matched_stems & unmatched_stems}"
            )


# Feature: accuracy-evaluation, Property 8: 视频标注校验完备性
# Validates: Requirements 3.6


class TestVideoAnnotationValidationProperty:
    """Property test: validate_video_annotation completeness.

    For any video annotation JSON containing invalid fields (missing required
    fields, invalid event types, invalid anomaly sub_types, invalid tripwire_id
    references, or time_range with start >= end), the validator must return a
    non-empty error list.

    **Validates: Requirements 3.6**
    """

    VALID_EVENT_TYPES = ["intrusion", "tripwire", "counting", "presence", "anomaly"]
    VALID_ANOMALY_SUB_TYPES = [
        "dwell", "crowd", "proximity", "fight", "fall", "wrong_way", "speed",
    ]

    # Strategy: random strings that are NOT valid event types
    invalid_event_type_st = st.text(min_size=1, max_size=30).filter(
        lambda s: s not in {"intrusion", "tripwire", "counting", "presence", "anomaly"}
    )

    # Strategy: random strings that are NOT valid anomaly sub_types
    invalid_sub_type_st = st.text(min_size=1, max_size=30).filter(
        lambda s: s not in {
            "dwell", "crowd", "proximity", "fight", "fall", "wrong_way", "speed",
        }
    )

    @given(
        fps_sample=st.integers(min_value=1, max_value=30),
        events=st.lists(
            st.fixed_dictionaries({
                "type": st.sampled_from(["intrusion", "counting", "presence"]),
            }),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_missing_video_path(self, fps_sample, events):
        """Data without 'video_path' must produce non-empty errors."""
        data = {
            "fps_sample": fps_sample,
            "events": events,
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert len(errors) > 0
        assert any("video_path" in e for e in errors)

    @given(
        invalid_type=invalid_event_type_st,
        num_valid=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=100)
    def test_invalid_event_type(self, invalid_type, num_valid):
        """Events with non-valid type strings must produce errors."""
        events = [
            {"type": "intrusion", "time_range": [0.0, 5.0]}
            for _ in range(num_valid)
        ]
        events.append({"type": invalid_type})
        data = {
            "video_path": "test.mp4",
            "events": events,
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert len(errors) > 0
        assert any("invalid event type" in e or "type" in e for e in errors)

    @given(
        invalid_sub=invalid_sub_type_st,
        num_valid=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=100)
    def test_invalid_anomaly_sub_type(self, invalid_sub, num_valid):
        """Anomaly events with non-valid sub_type must produce errors."""
        events = [
            {"type": "intrusion", "time_range": [0.0, 5.0]}
            for _ in range(num_valid)
        ]
        events.append({"type": "anomaly", "sub_type": invalid_sub})
        data = {
            "video_path": "test.mp4",
            "events": events,
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert len(errors) > 0
        assert any("sub_type" in e for e in errors)

    @given(
        bad_id=st.text(min_size=1, max_size=20).filter(
            lambda s: s not in {"tw01", "tw02"}
        ),
    )
    @settings(max_examples=100)
    def test_invalid_tripwire_id_reference(self, bad_id):
        """Tripwire events referencing non-existent IDs must produce errors."""
        data = {
            "video_path": "test.mp4",
            "camera_config": {
                "tripwires": [
                    {"id": "tw01", "p1": [0, 0], "p2": [100, 100]},
                    {"id": "tw02", "p1": [0, 0], "p2": [200, 200]},
                ],
            },
            "events": [
                {"type": "tripwire", "tripwire_id": bad_id},
            ],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert len(errors) > 0
        assert any("not found" in e or "tripwire_id" in e for e in errors)

    @given(
        start=st.floats(min_value=0.0, max_value=1000.0,
                        allow_nan=False, allow_infinity=False),
        delta=st.floats(min_value=0.0, max_value=500.0,
                        allow_nan=False, allow_infinity=False),
        event_type=st.sampled_from(["intrusion", "presence"]),
    )
    @settings(max_examples=100)
    def test_invalid_time_range_start_ge_end(self, start, delta, event_type):
        """Events with time_range where start >= end must produce errors."""
        end = start - delta  # end <= start, so start >= end
        data = {
            "video_path": "test.mp4",
            "events": [
                {"type": event_type, "time_range": [start, end]},
            ],
        }
        errors = AnnotationLoader.validate_video_annotation(data)
        assert len(errors) > 0
        assert any("start" in e and "less than" in e for e in errors)
