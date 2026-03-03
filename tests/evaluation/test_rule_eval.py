"""Unit tests for RuleEvaluator.

Covers:
- _match_events greedy matching logic (instant events, duration events, one-to-one)
- Video that can't open is skipped with error
- No events of a type → N/A (has_samples=False)
- All detectors are mocked to avoid needing real models
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.evaluation.rule_eval import RuleEvaluator
from src.evaluation.models import (
    MatchResult,
    RuleMetrics,
    VideoAnnotation,
    VideoEvalResult,
)


# ── Helpers ──


def _make_evaluator() -> RuleEvaluator:
    """Create a RuleEvaluator without loading real models."""
    evaluator = RuleEvaluator(
        model_path="fake_model.pt",
        pose_model_path=None,
        confidence=0.3,
        time_tolerance=2.0,
    )
    return evaluator


# ── Tests for _match_events ──


class TestMatchEventsInstant:
    """Test greedy matching for instant events (time_sec)."""

    def test_perfect_match_single(self):
        evaluator = _make_evaluator()
        pred = [{"timestamp": 3.0}]
        gt = [{"time_sec": 3.5}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 0
        assert len(result.matched_pairs) == 1

    def test_no_match_outside_tolerance(self):
        evaluator = _make_evaluator()
        pred = [{"timestamp": 10.0}]
        gt = [{"time_sec": 3.0}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 0
        assert result.fp == 1
        assert result.fn == 1

    def test_multiple_predictions_one_gt(self):
        """Only one prediction should match each GT (one-to-one)."""
        evaluator = _make_evaluator()
        pred = [{"timestamp": 3.0}, {"timestamp": 3.5}, {"timestamp": 4.0}]
        gt = [{"time_sec": 3.2}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 1
        assert result.fp == 2
        assert result.fn == 0

    def test_multiple_gt_one_prediction(self):
        """One prediction can only match one GT."""
        evaluator = _make_evaluator()
        pred = [{"timestamp": 5.0}]
        gt = [{"time_sec": 4.5}, {"time_sec": 5.5}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 1

    def test_greedy_closest_first(self):
        """Greedy matching should prefer closest time pairs."""
        evaluator = _make_evaluator()
        pred = [{"timestamp": 1.0}, {"timestamp": 5.0}]
        gt = [{"time_sec": 1.1}, {"time_sec": 4.9}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 2
        assert result.fp == 0
        assert result.fn == 0
        # Verify closest pairs matched
        times = [(p["timestamp"], g["time_sec"]) for p, g in result.matched_pairs]
        assert (1.0, 1.1) in times
        assert (5.0, 4.9) in times

    def test_empty_predictions(self):
        evaluator = _make_evaluator()
        result = evaluator._match_events([], [{"time_sec": 1.0}], tolerance=2.0)
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 1

    def test_empty_ground_truth(self):
        evaluator = _make_evaluator()
        result = evaluator._match_events([{"timestamp": 1.0}], [], tolerance=2.0)
        assert result.tp == 0
        assert result.fp == 1
        assert result.fn == 0

    def test_both_empty(self):
        evaluator = _make_evaluator()
        result = evaluator._match_events([], [], tolerance=2.0)
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 0


class TestMatchEventsDuration:
    """Test greedy matching for duration events (time_range)."""

    def test_pred_within_range(self):
        evaluator = _make_evaluator()
        pred = [{"timestamp": 5.0}]
        gt = [{"time_range": [3.0, 8.0]}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 0

    def test_pred_within_tolerance_before_range(self):
        """Prediction just before range start, within tolerance."""
        evaluator = _make_evaluator()
        pred = [{"timestamp": 1.5}]
        gt = [{"time_range": [3.0, 8.0]}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 1

    def test_pred_within_tolerance_after_range(self):
        """Prediction just after range end, within tolerance."""
        evaluator = _make_evaluator()
        pred = [{"timestamp": 9.5}]
        gt = [{"time_range": [3.0, 8.0]}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 1

    def test_pred_outside_tolerance(self):
        evaluator = _make_evaluator()
        pred = [{"timestamp": 15.0}]
        gt = [{"time_range": [3.0, 8.0]}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 0
        assert result.fp == 1
        assert result.fn == 1

    def test_mixed_instant_and_duration(self):
        """Mix of instant and duration GT events."""
        evaluator = _make_evaluator()
        pred = [{"timestamp": 5.0}, {"timestamp": 12.0}]
        gt = [
            {"time_range": [4.0, 7.0]},  # duration
            {"time_sec": 12.5},            # instant
        ]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 2
        assert result.fp == 0
        assert result.fn == 0


class TestMatchEventsOneToOne:
    """Test one-to-one constraint in matching."""

    def test_one_to_one_constraint(self):
        """Each GT and each prediction can only be matched once."""
        evaluator = _make_evaluator()
        # Two predictions very close to same GT
        pred = [{"timestamp": 3.0}, {"timestamp": 3.1}]
        gt = [{"time_sec": 3.05}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp == 1
        assert result.fp == 1
        assert result.fn == 0

    def test_tp_fp_fn_consistency(self):
        """tp + fp = len(predicted), tp + fn = len(ground_truth)."""
        evaluator = _make_evaluator()
        pred = [{"timestamp": 1.0}, {"timestamp": 5.0}, {"timestamp": 20.0}]
        gt = [{"time_sec": 1.2}, {"time_sec": 10.0}]
        result = evaluator._match_events(pred, gt, tolerance=2.0)
        assert result.tp + result.fp == len(pred)
        assert result.tp + result.fn == len(gt)


# ── Tests for video that can't open ──


class TestVideoCannotOpen:
    """Test that videos that can't be opened are skipped with error."""

    @patch("src.evaluation.rule_eval.cv2.VideoCapture")
    def test_video_cannot_open_skipped(self, mock_cap_cls):
        """When video can't be opened, result has error and no metrics."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        evaluator = _make_evaluator()
        annotation = VideoAnnotation(
            video_path="test.mp4",
            fps_sample=5,
            camera_config={"roi": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            events=[{"type": "intrusion", "time_range": [1.0, 5.0]}],
        )

        result = evaluator._evaluate_video(Path("test.mp4"), annotation)

        assert len(result.errors) > 0
        assert "Cannot open video" in result.errors[0]
        assert result.rule_metrics == []
        mock_cap.release.assert_called_once()


# ── Tests for no events of a type → N/A ──


class TestNoEventsNA:
    """Test that missing event types produce has_samples=False."""

    def test_no_intrusion_events_na(self):
        evaluator = _make_evaluator()
        gt_events = [{"type": "tripwire", "time_sec": 3.0, "tripwire_id": "tw01"}]
        metrics = evaluator._evaluate_intrusion([], gt_events)
        assert len(metrics) == 1
        assert metrics[0].has_samples is False
        assert metrics[0].rule_type == "intrusion"

    def test_no_tripwire_events_na(self):
        evaluator = _make_evaluator()
        gt_events = [{"type": "intrusion", "time_range": [1.0, 5.0]}]
        metrics = evaluator._evaluate_tripwire([], gt_events, {})
        assert len(metrics) == 1
        assert metrics[0].has_samples is False
        assert metrics[0].rule_type == "tripwire"

    def test_no_counting_events_na(self):
        evaluator = _make_evaluator()
        gt_events = [{"type": "intrusion", "time_range": [1.0, 5.0]}]
        metrics = evaluator._evaluate_counting(None, gt_events)
        assert len(metrics) == 1
        assert metrics[0].has_samples is False
        assert metrics[0].rule_type == "counting"

    def test_no_presence_events_na(self):
        evaluator = _make_evaluator()
        gt_events = [{"type": "intrusion", "time_range": [1.0, 5.0]}]
        metrics = evaluator._evaluate_presence([], gt_events)
        assert len(metrics) == 1
        assert metrics[0].has_samples is False
        assert metrics[0].rule_type == "presence"

    def test_no_anomaly_events_na(self):
        evaluator = _make_evaluator()
        gt_events = [{"type": "intrusion", "time_range": [1.0, 5.0]}]
        metrics = evaluator._evaluate_anomaly([], gt_events)
        assert len(metrics) == 1
        assert metrics[0].has_samples is False
        assert metrics[0].rule_type == "anomaly"


# ── Tests for rule-specific evaluation logic ──


class TestEvaluateIntrusion:
    """Test intrusion evaluation with mock events."""

    def test_perfect_intrusion_match(self):
        evaluator = _make_evaluator()
        pred = [{"type": "intrusion", "timestamp": 3.0, "class_name": "person"}]
        gt = [{"type": "intrusion", "time_range": [2.0, 5.0], "class_name": "person"}]
        metrics = evaluator._evaluate_intrusion(pred, gt)
        assert metrics[0].tp == 1
        assert metrics[0].precision == 1.0
        assert metrics[0].recall == 1.0

    def test_intrusion_false_positive(self):
        evaluator = _make_evaluator()
        pred = [
            {"type": "intrusion", "timestamp": 3.0, "class_name": "person"},
            {"type": "intrusion", "timestamp": 20.0, "class_name": "person"},
        ]
        gt = [{"type": "intrusion", "time_range": [2.0, 5.0], "class_name": "person"}]
        metrics = evaluator._evaluate_intrusion(pred, gt)
        assert metrics[0].tp == 1
        assert metrics[0].fp == 1
        assert len(metrics[0].details["false_positives"]) == 1


class TestEvaluateAnomaly:
    """Test anomaly evaluation grouped by sub_type."""

    def test_anomaly_by_subtype(self):
        evaluator = _make_evaluator()
        pred = [
            {"type": "anomaly", "sub_type": "dwell", "timestamp": 15.0},
            {"type": "anomaly", "sub_type": "crowd", "timestamp": 8.0},
        ]
        gt = [
            {"type": "anomaly", "sub_type": "dwell", "time_range": [10.0, 25.0]},
            {"type": "anomaly", "sub_type": "crowd", "time_range": [5.0, 15.0]},
        ]
        metrics = evaluator._evaluate_anomaly(pred, gt)
        # Should have 2 metrics (one per sub_type)
        assert len(metrics) == 2
        sub_types = {m.sub_type for m in metrics}
        assert sub_types == {"crowd", "dwell"}
        for m in metrics:
            assert m.tp == 1
            assert m.has_samples is True


class TestEvaluateCounting:
    """Test counting evaluation."""

    def test_counting_with_no_flow_counter(self):
        evaluator = _make_evaluator()
        gt = [{"type": "counting", "expected_in": 5, "expected_out": 3}]
        metrics = evaluator._evaluate_counting(None, gt)
        assert metrics[0].has_samples is True
        assert metrics[0].details["expected_in"] == 5
        assert metrics[0].details["actual_in"] == 0
        assert metrics[0].details["abs_error_in"] == 5

    def test_counting_with_mock_flow_counter(self):
        evaluator = _make_evaluator()
        mock_counter = MagicMock()
        mock_counter.get_current_counts.return_value = {
            "total_in": 4, "total_out": 3,
        }
        gt = [{"type": "counting", "expected_in": 5, "expected_out": 3}]
        metrics = evaluator._evaluate_counting(mock_counter, gt)
        assert metrics[0].details["actual_in"] == 4
        assert metrics[0].details["actual_out"] == 3
        assert metrics[0].details["abs_error_in"] == 1
        assert metrics[0].details["abs_error_out"] == 0


class TestEvaluatePresence:
    """Test presence evaluation."""

    def test_presence_match(self):
        evaluator = _make_evaluator()
        pred = [{"type": "presence", "timestamp": 5.0, "class_name": "person"}]
        gt = [{"type": "presence", "time_range": [0, 10], "class_name": "person"}]
        metrics = evaluator._evaluate_presence(pred, gt)
        assert metrics[0].tp == 1
        assert metrics[0].recall == 1.0


# ── Tests for aggregation ──


class TestAggregateMetrics:
    """Test cross-video metric aggregation."""

    def test_aggregate_sums_tp_fp_fn(self):
        evaluator = _make_evaluator()
        v1 = VideoEvalResult(
            video_name="v1.mp4",
            rule_metrics=[
                RuleMetrics("intrusion", "", 1.0, 0.5, 0.67, 1, 0, 1, True, {}),
            ],
            errors=[],
        )
        v2 = VideoEvalResult(
            video_name="v2.mp4",
            rule_metrics=[
                RuleMetrics("intrusion", "", 0.5, 1.0, 0.67, 2, 2, 0, True, {}),
            ],
            errors=[],
        )
        agg = evaluator._aggregate_metrics([v1, v2])
        intrusion_agg = [m for m in agg if m.rule_type == "intrusion"]
        assert len(intrusion_agg) == 1
        assert intrusion_agg[0].tp == 3
        assert intrusion_agg[0].fp == 2
        assert intrusion_agg[0].fn == 1

    def test_aggregate_na_stays_na(self):
        evaluator = _make_evaluator()
        v1 = VideoEvalResult(
            video_name="v1.mp4",
            rule_metrics=[
                RuleMetrics("anomaly", "dwell", 0.0, 0.0, 0.0, 0, 0, 0, False, {}),
            ],
            errors=[],
        )
        agg = evaluator._aggregate_metrics([v1])
        assert agg[0].has_samples is False


# ── Tests for run() with mocked filesystem ──


class TestRunScanDirectory:
    """Test run() scans directory and matches videos to annotations."""

    def test_run_no_videos(self, tmp_path):
        evaluator = _make_evaluator()
        result = evaluator.run(tmp_path)
        assert result.per_video == []
        assert result.aggregated == []

    def test_run_video_without_annotation_skipped(self, tmp_path):
        """Video without matching JSON is skipped."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")
        evaluator = _make_evaluator()
        result = evaluator.run(tmp_path)
        # No annotation → skipped entirely (not in per_video)
        assert len(result.per_video) == 0


# ── Property-Based Tests ──

# Feature: accuracy-evaluation, Property 9: 事件匹配一对一约束
# Validates: Requirements 4.4

from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def event_lists(draw):
    """Generate random predicted events, ground truth events, and tolerance."""
    n_pred = draw(st.integers(min_value=0, max_value=15))
    n_gt = draw(st.integers(min_value=0, max_value=15))
    tolerance = draw(st.floats(min_value=0.1, max_value=10.0))

    predicted = [
        {"timestamp": draw(st.floats(min_value=0.0, max_value=100.0))}
        for _ in range(n_pred)
    ]

    # Ground truth can be instant (time_sec) or duration (time_range)
    ground_truth = []
    for _ in range(n_gt):
        use_range = draw(st.booleans())
        if use_range:
            start = draw(st.floats(min_value=0.0, max_value=90.0))
            duration = draw(st.floats(min_value=0.1, max_value=10.0))
            ground_truth.append({"time_range": [start, start + duration]})
        else:
            ground_truth.append(
                {"time_sec": draw(st.floats(min_value=0.0, max_value=100.0))}
            )

    return predicted, ground_truth, tolerance


class TestMatchEventsProperty:
    """Property 9: 事件匹配一对一约束"""

    @given(data=event_lists())
    @settings(max_examples=100)
    def test_one_to_one_and_counts(self, data):
        """
        **Validates: Requirements 4.4**

        For any predicted and ground truth event lists, the greedy matching
        algorithm must satisfy:
        1. Each ground truth event matches at most one predicted event
        2. Each predicted event matches at most one ground truth event
        3. All matched pairs have time difference within tolerance
        4. tp + fp == len(predicted), tp + fn == len(ground_truth)
        """
        predicted, ground_truth, tolerance = data
        evaluator = _make_evaluator()
        result = evaluator._match_events(predicted, ground_truth, tolerance)

        # Property 4: tp + fp == len(predicted), tp + fn == len(ground_truth)
        assert result.tp + result.fp == len(predicted), (
            f"tp({result.tp}) + fp({result.fp}) != len(predicted)({len(predicted)})"
        )
        assert result.tp + result.fn == len(ground_truth), (
            f"tp({result.tp}) + fn({result.fn}) != len(ground_truth)({len(ground_truth)})"
        )

        # Property 1 & 2: one-to-one constraint
        matched_pred_ids = set()
        matched_gt_ids = set()
        for pred_evt, gt_evt in result.matched_pairs:
            pred_idx = id(pred_evt)
            gt_idx = id(gt_evt)
            assert pred_idx not in matched_pred_ids, "Predicted event matched more than once"
            assert gt_idx not in matched_gt_ids, "Ground truth event matched more than once"
            matched_pred_ids.add(pred_idx)
            matched_gt_ids.add(gt_idx)

        assert len(result.matched_pairs) == result.tp

        # Property 3: all matched pairs within tolerance
        for pred_evt, gt_evt in result.matched_pairs:
            pred_time = pred_evt["timestamp"]
            time_range = gt_evt.get("time_range")
            gt_time = gt_evt.get("time_sec")

            if time_range is not None:
                start, end = time_range[0], time_range[1]
                assert start - tolerance <= pred_time <= end + tolerance, (
                    f"Matched pair outside tolerance: pred={pred_time}, "
                    f"range=[{start}, {end}], tol={tolerance}"
                )
            else:
                assert abs(pred_time - gt_time) <= tolerance, (
                    f"Matched pair outside tolerance: pred={pred_time}, "
                    f"gt={gt_time}, tol={tolerance}"
                )
