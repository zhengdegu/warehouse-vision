"""Property-based tests for metrics computation utilities.

# Feature: accuracy-evaluation, Property 15: IoU 计算正确性
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.metrics import compute_iou


# Strategy: generate a valid bbox where x1 < x2 and y1 < y2, coords in [0, 1000]
def valid_bbox():
    """Generate a valid bbox [x1, y1, x2, y2] with x1 < x2 and y1 < y2."""
    return st.tuples(
        st.floats(min_value=0, max_value=999, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=999, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=999, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=999, allow_nan=False, allow_infinity=False),
    ).map(lambda t: [min(t[0], t[2]), min(t[1], t[3]),
                     max(t[0], t[2]) + 1, max(t[1], t[3]) + 1])


class TestComputeIouProperty:
    """Property 15: IoU 计算正确性

    **Validates: Requirements 2.4**
    """

    @given(box1=valid_bbox(), box2=valid_bbox())
    @settings(max_examples=100)
    def test_iou_bounded_between_0_and_1(self, box1, box2):
        """For any two valid bboxes, 0 <= IoU <= 1."""
        iou = compute_iou(box1, box2)
        assert 0.0 <= iou <= 1.0, f"IoU={iou} out of [0, 1] for {box1}, {box2}"

    @given(box=valid_bbox())
    @settings(max_examples=100)
    def test_iou_identical_boxes_equals_1(self, box):
        """For identical bboxes, IoU = 1.0."""
        iou = compute_iou(box, box)
        assert iou == 1.0, f"IoU={iou} != 1.0 for identical box {box}"

    @given(
        x1=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        y1=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        w1=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        h1=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        h2=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        gap=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_iou_non_overlapping_boxes_equals_0(self, x1, y1, w1, h1, w2, h2, gap):
        """For non-overlapping bboxes (guaranteed no intersection), IoU = 0.0."""
        # Place box2 to the right of box1 with a guaranteed gap
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x1 + w1 + gap, y1, x1 + w1 + gap + w2, y1 + h2]
        iou = compute_iou(box1, box2)
        assert iou == 0.0, f"IoU={iou} != 0.0 for non-overlapping {box1}, {box2}"

    @given(box1=valid_bbox(), box2=valid_bbox())
    @settings(max_examples=100)
    def test_iou_symmetry(self, box1, box2):
        """IoU(A, B) == IoU(B, A) (symmetry)."""
        iou_ab = compute_iou(box1, box2)
        iou_ba = compute_iou(box2, box1)
        assert iou_ab == iou_ba, f"IoU(A,B)={iou_ab} != IoU(B,A)={iou_ba}"


# Feature: accuracy-evaluation, Property 5: AP 计算单调性

import numpy as np

from src.evaluation.metrics import compute_ap


def _sorted_recalls(n):
    """Strategy: generate a non-decreasing recall list of length n in [0, 1].

    Recalls represent cumulative recall as detections are processed in
    confidence-descending order, so they can only increase or stay the same.
    """
    return (
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
        .map(sorted)
    )


def pr_curve(min_size=1, max_size=50):
    """Strategy: generate a valid (precisions, recalls) pair.

    - recalls: non-decreasing list of floats in [0, 1]
    - precisions: list of floats in [0, 1] (same length as recalls)
    """
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.tuples(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ),
            _sorted_recalls(n),
        )
    )


AP_METHODS = st.sampled_from(["interp11", "all_points"])


class TestComputeApProperty:
    """Property 5: AP 计算单调性

    **Validates: Requirements 2.5**
    """

    @given(data=pr_curve(), method=AP_METHODS)
    @settings(max_examples=100)
    def test_ap_bounded_between_0_and_1(self, data, method):
        """For any valid precision-recall curve, 0 <= AP <= 1."""
        precisions, recalls = data
        ap = compute_ap(precisions, recalls, method)
        assert 0.0 <= ap <= 1.0, (
            f"AP={ap} out of [0, 1] for method={method}, "
            f"precisions={precisions}, recalls={recalls}"
        )

    @given(
        n=st.integers(min_value=1, max_value=50),
        method=AP_METHODS,
    )
    @settings(max_examples=100)
    def test_ap_all_tp_equals_1(self, n, method):
        """When all predictions are TP (precision=1.0, recall increasing to 1.0), AP=1.0."""
        # Simulate n detections that are all TP: precision stays 1.0,
        # recall increases linearly from 1/n to n/n = 1.0.
        precisions = [1.0] * n
        recalls = [(i + 1) / n for i in range(n)]
        ap = compute_ap(precisions, recalls, method)
        assert abs(ap - 1.0) < 1e-9, (
            f"AP={ap} != 1.0 for all-TP curve with n={n}, method={method}"
        )

    @given(
        n=st.integers(min_value=1, max_value=50),
        method=AP_METHODS,
    )
    @settings(max_examples=100)
    def test_ap_all_fp_equals_0(self, n, method):
        """When all predictions are FP (precision=0.0, recall=0.0), AP=0.0."""
        # All detections are false positives: precision is always 0, recall is always 0.
        precisions = [0.0] * n
        recalls = [0.0] * n
        ap = compute_ap(precisions, recalls, method)
        assert ap == 0.0, (
            f"AP={ap} != 0.0 for all-FP curve with n={n}, method={method}"
        )


# Feature: accuracy-evaluation, Property 6: 混淆矩阵行列和一致性

from src.evaluation.metrics import compute_confusion_matrix


def class_names_strategy():
    """Strategy: generate a list of 1-10 unique class names."""
    return st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L",)),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=10,
        unique=True,
    )


def matches_strategy(class_names):
    """Strategy: generate a list of match dicts for given class_names.

    Each match has gt_class and pred_class that are either a valid class
    index (0..len(class_names)-1) or None (representing background).
    """
    n = len(class_names)
    class_or_none = st.one_of(st.integers(min_value=0, max_value=n - 1), st.none())
    return st.lists(
        st.fixed_dictionaries(
            {"gt_class": class_or_none, "pred_class": class_or_none}
        ),
        min_size=0,
        max_size=50,
    )


class TestConfusionMatrixProperty:
    """Property 6: 混淆矩阵行列和一致性

    **Validates: Requirements 2.6**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_matrix_dimensions(self, data):
        """Matrix dimensions are (num_classes+1) x (num_classes+1)."""
        names = data.draw(class_names_strategy(), label="class_names")
        matches = data.draw(matches_strategy(names), label="matches")

        matrix = compute_confusion_matrix(matches, names)
        expected_size = len(names) + 1

        assert len(matrix) == expected_size, (
            f"Expected {expected_size} rows, got {len(matrix)}"
        )
        for i, row in enumerate(matrix):
            assert len(row) == expected_size, (
                f"Row {i}: expected {expected_size} cols, got {len(row)}"
            )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_matrix_total_equals_match_count(self, data):
        """Sum of all elements in the matrix equals the number of matches."""
        names = data.draw(class_names_strategy(), label="class_names")
        matches = data.draw(matches_strategy(names), label="matches")

        matrix = compute_confusion_matrix(matches, names)
        total = sum(cell for row in matrix for cell in row)

        assert total == len(matches), (
            f"Matrix total {total} != match count {len(matches)}"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_matrix_all_non_negative(self, data):
        """All elements in the matrix are non-negative."""
        names = data.draw(class_names_strategy(), label="class_names")
        matches = data.draw(matches_strategy(names), label="matches")

        matrix = compute_confusion_matrix(matches, names)

        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                assert cell >= 0, (
                    f"Negative value {cell} at matrix[{i}][{j}]"
                )


# Feature: accuracy-evaluation, Property 10: P/R/F1 计算正确性

from src.evaluation.metrics import compute_prf


class TestComputePrfProperty:
    """Property 10: Precision/Recall/F1 计算正确性

    **Validates: Requirements 4.5, 4.6, 4.8, 4.9**
    """

    @given(
        tp=st.integers(min_value=0, max_value=10000),
        fp=st.integers(min_value=0, max_value=10000),
        fn=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_prf_bounded_between_0_and_1(self, tp, fp, fn):
        """For any non-negative tp, fp, fn: 0 <= precision, recall, f1 <= 1."""
        precision, recall, f1 = compute_prf(tp, fp, fn)
        assert 0.0 <= precision <= 1.0, f"precision={precision} out of [0, 1]"
        assert 0.0 <= recall <= 1.0, f"recall={recall} out of [0, 1]"
        assert 0.0 <= f1 <= 1.0, f"f1={f1} out of [0, 1]"

    @given(
        tp=st.integers(min_value=1, max_value=10000),
        fp=st.integers(min_value=0, max_value=10000),
        fn=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_precision_formula(self, tp, fp, fn):
        """When tp > 0, precision = tp / (tp + fp)."""
        precision, _, _ = compute_prf(tp, fp, fn)
        expected = tp / (tp + fp)
        assert abs(precision - expected) < 1e-9, (
            f"precision={precision} != expected={expected} for tp={tp}, fp={fp}"
        )

    @given(
        tp=st.integers(min_value=1, max_value=10000),
        fp=st.integers(min_value=0, max_value=10000),
        fn=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_recall_formula(self, tp, fp, fn):
        """When tp > 0, recall = tp / (tp + fn)."""
        _, recall, _ = compute_prf(tp, fp, fn)
        expected = tp / (tp + fn)
        assert abs(recall - expected) < 1e-9, (
            f"recall={recall} != expected={expected} for tp={tp}, fn={fn}"
        )

    @given(
        tp=st.integers(min_value=1, max_value=10000),
        fp=st.integers(min_value=0, max_value=10000),
        fn=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_f1_formula(self, tp, fp, fn):
        """When tp > 0, f1 = 2 * precision * recall / (precision + recall)."""
        precision, recall, f1 = compute_prf(tp, fp, fn)
        if precision + recall > 0:
            expected_f1 = 2.0 * precision * recall / (precision + recall)
            assert abs(f1 - expected_f1) < 1e-9, (
                f"f1={f1} != expected={expected_f1}"
            )

    @given(
        fp=st.integers(min_value=0, max_value=10000),
        fn=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_tp_zero_returns_all_zeros(self, fp, fn):
        """When tp=0, precision, recall, and f1 are all 0.0."""
        precision, recall, f1 = compute_prf(0, fp, fn)
        assert precision == 0.0, f"precision={precision} != 0.0 when tp=0"
        assert recall == 0.0, f"recall={recall} != 0.0 when tp=0"
        assert f1 == 0.0, f"f1={f1} != 0.0 when tp=0"


# Feature: accuracy-evaluation, Property 11: 计数误差计算正确性

import math

from src.evaluation.metrics import compute_counting_error


class TestComputeCountingErrorProperty:
    """Property 11: 计数误差计算正确性

    **Validates: Requirements 4.7**
    """

    @given(
        expected=st.integers(min_value=0, max_value=10000),
        actual=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_abs_error_formula(self, expected, actual):
        """abs_error = |actual - expected| for any non-negative expected, actual."""
        abs_error, _ = compute_counting_error(expected, actual)
        assert abs_error == abs(actual - expected), (
            f"abs_error={abs_error} != |{actual} - {expected}| = {abs(actual - expected)}"
        )

    @given(
        expected=st.integers(min_value=1, max_value=10000),
        actual=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_rel_error_when_expected_positive(self, expected, actual):
        """When expected > 0, rel_error = abs_error / expected."""
        abs_error, rel_error = compute_counting_error(expected, actual)
        expected_rel = abs_error / expected
        assert abs(rel_error - expected_rel) < 1e-9, (
            f"rel_error={rel_error} != {expected_rel} for expected={expected}, actual={actual}"
        )

    @given(
        actual=st.just(0),
    )
    @settings(max_examples=1)
    def test_rel_error_when_both_zero(self, actual):
        """When expected == 0 and actual == 0, rel_error = 0.0."""
        abs_error, rel_error = compute_counting_error(0, actual)
        assert abs_error == 0, f"abs_error={abs_error} != 0"
        assert rel_error == 0.0, f"rel_error={rel_error} != 0.0"

    @given(
        actual=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_rel_error_when_expected_zero_actual_positive(self, actual):
        """When expected == 0 and actual > 0, rel_error = float('inf')."""
        abs_error, rel_error = compute_counting_error(0, actual)
        assert abs_error == actual, f"abs_error={abs_error} != {actual}"
        assert math.isinf(rel_error) and rel_error > 0, (
            f"rel_error={rel_error} is not positive infinity"
        )
