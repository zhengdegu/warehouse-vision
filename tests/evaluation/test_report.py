"""Property-based tests for report generation.

# Feature: accuracy-evaluation, Property 12: HTML 报告自包含性
# Feature: accuracy-evaluation, Property 13: JSON 报告数据往返一致性
"""

import json
import re
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation.models import (
    ClassMetrics,
    DetectionResult,
    ImageEvalEntry,
    RuleMetrics,
    RuleResult,
    VideoEvalResult,
)
from src.evaluation.report import ReportGenerator


# ── Strategies ──


def class_metrics_strategy():
    """Generate a valid ClassMetrics with bounded floats and non-negative ints."""
    return st.builds(
        ClassMetrics,
        class_id=st.integers(min_value=0, max_value=20),
        class_name=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=10,
        ),
        ap=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        precision=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        recall=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        tp=st.integers(min_value=0, max_value=1000),
        fp=st.integers(min_value=0, max_value=1000),
        fn=st.integers(min_value=0, max_value=1000),
        gt_count=st.integers(min_value=0, max_value=1000),
        has_samples=st.booleans(),
    )


def image_eval_entry_strategy():
    """Generate a valid ImageEvalEntry with tp <= min(gt, det)."""
    return st.integers(min_value=0, max_value=100).flatmap(
        lambda gt: st.integers(min_value=0, max_value=100).flatmap(
            lambda det: st.integers(min_value=0, max_value=min(gt, det)).map(
                lambda tp: ImageEvalEntry(
                    filename=f"img_{gt}_{det}_{tp}.jpg",
                    gt_count=gt,
                    det_count=det,
                    tp_count=tp,
                )
            )
        )
    )


def confusion_matrix_strategy(num_classes):
    """Generate a 2D confusion matrix of shape (num_classes+1) x (num_classes+1)."""
    size = num_classes + 1
    return st.lists(
        st.lists(
            st.integers(min_value=0, max_value=50),
            min_size=size,
            max_size=size,
        ),
        min_size=size,
        max_size=size,
    )


def detection_result_strategy():
    """Generate a valid DetectionResult with consistent dimensions."""
    return st.integers(min_value=1, max_value=5).flatmap(
        lambda n_classes: st.tuples(
            st.lists(class_metrics_strategy(), min_size=n_classes, max_size=n_classes),
            confusion_matrix_strategy(n_classes),
            st.lists(image_eval_entry_strategy(), min_size=0, max_size=5),
        ).flatmap(
            lambda t: st.tuples(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.integers(min_value=0, max_value=10),
            ).map(
                lambda m: DetectionResult(
                    map50=m[0],
                    map50_95=m[1],
                    total_precision=m[2],
                    total_recall=m[3],
                    class_metrics=t[0],
                    confusion_matrix=t[1],
                    per_image=t[2],
                    skipped_images=m[4],
                )
            )
        )
    )


KNOWN_RULE_TYPES = ["intrusion", "tripwire", "counting", "presence", "anomaly"]


def rule_metrics_strategy(rule_type=None):
    """Generate a valid RuleMetrics with known rule_type."""
    rt = st.just(rule_type) if rule_type else st.sampled_from(KNOWN_RULE_TYPES)
    return st.builds(
        RuleMetrics,
        rule_type=rt,
        sub_type=st.text(
            alphabet=st.characters(whitelist_categories=("L",)),
            min_size=0,
            max_size=8,
        ),
        precision=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        recall=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        f1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        tp=st.integers(min_value=0, max_value=500),
        fp=st.integers(min_value=0, max_value=500),
        fn=st.integers(min_value=0, max_value=500),
        has_samples=st.just(True),
        details=st.just({}),
    )


def video_eval_result_strategy():
    """Generate a valid VideoEvalResult."""
    return st.builds(
        VideoEvalResult,
        video_name=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=12,
        ),
        rule_metrics=st.lists(rule_metrics_strategy(), min_size=0, max_size=3),
        errors=st.just([]),
    )


def rule_result_strategy():
    """Generate a valid RuleResult with one aggregated metric per known rule type."""
    aggregated = st.tuples(
        *[rule_metrics_strategy(rt) for rt in KNOWN_RULE_TYPES]
    ).map(list)
    return st.builds(
        RuleResult,
        per_video=st.lists(video_eval_result_strategy(), min_size=0, max_size=3),
        aggregated=aggregated,
    )



# ── Property 12: HTML 报告自包含性 ──

# Regex patterns for external resource references
_EXTERNAL_CSS_RE = re.compile(r'<link\b[^>]*\bhref\s*=\s*["\']https?://', re.IGNORECASE)
_EXTERNAL_JS_RE = re.compile(r'<script\b[^>]*\bsrc\s*=\s*["\']https?://', re.IGNORECASE)

EXPECTED_SECTION_IDS = [
    "section-overview",
    "section-detection",
    "section-intrusion",
    "section-tripwire",
    "section-counting",
    "section-presence",
    "section-anomaly",
]


def _generate_report(gen, detection_result=None, rule_result=None):
    """Generate report in a fresh temp directory and return (html_path, output_dir)."""
    output_dir = Path(tempfile.mkdtemp())
    html_path = gen.generate(
        detection_result=detection_result,
        rule_result=rule_result,
        output_dir=output_dir,
    )
    return html_path, output_dir


class TestHtmlSelfContainedProperty:
    """Property 12: HTML 报告自包含性

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    @given(det=detection_result_strategy(), rules=rule_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_no_external_resources_both(self, det, rules):
        """HTML with both detection and rules has no external CSS/JS."""
        gen = ReportGenerator()
        html_path, _ = _generate_report(gen, detection_result=det, rule_result=rules)
        html = html_path.read_text(encoding="utf-8")

        assert not _EXTERNAL_CSS_RE.search(html), (
            "Found external CSS <link> in HTML report"
        )
        assert not _EXTERNAL_JS_RE.search(html), (
            "Found external <script src> in HTML report"
        )

    @given(det=detection_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_no_external_resources_detection_only(self, det):
        """HTML with detection only has no external CSS/JS."""
        gen = ReportGenerator()
        html_path, _ = _generate_report(gen, detection_result=det, rule_result=None)
        html = html_path.read_text(encoding="utf-8")

        assert not _EXTERNAL_CSS_RE.search(html), (
            "Found external CSS <link> in HTML report"
        )
        assert not _EXTERNAL_JS_RE.search(html), (
            "Found external <script src> in HTML report"
        )

    @given(rules=rule_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_no_external_resources_rules_only(self, rules):
        """HTML with rules only has no external CSS/JS."""
        gen = ReportGenerator()
        html_path, _ = _generate_report(gen, detection_result=None, rule_result=rules)
        html = html_path.read_text(encoding="utf-8")

        assert not _EXTERNAL_CSS_RE.search(html), (
            "Found external CSS <link> in HTML report"
        )
        assert not _EXTERNAL_JS_RE.search(html), (
            "Found external <script src> in HTML report"
        )

    @settings(max_examples=1, deadline=None)
    @given(data=st.data())
    def test_no_external_resources_neither(self, data):
        """HTML with no data has no external CSS/JS."""
        gen = ReportGenerator()
        html_path, _ = _generate_report(gen)
        html = html_path.read_text(encoding="utf-8")

        assert not _EXTERNAL_CSS_RE.search(html), (
            "Found external CSS <link> in HTML report"
        )
        assert not _EXTERNAL_JS_RE.search(html), (
            "Found external <script src> in HTML report"
        )

    @given(det=detection_result_strategy(), rules=rule_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_all_section_ids_present(self, det, rules):
        """HTML contains all required section identifiers."""
        gen = ReportGenerator()
        html_path, _ = _generate_report(gen, detection_result=det, rule_result=rules)
        html = html_path.read_text(encoding="utf-8")

        for section_id in EXPECTED_SECTION_IDS:
            assert section_id in html, (
                f"Missing section id \'{section_id}\' in HTML report"
            )

    @settings(max_examples=1, deadline=None)
    @given(data=st.data())
    def test_all_section_ids_present_empty(self, data):
        """HTML with no data still contains all section identifiers."""
        gen = ReportGenerator()
        html_path, _ = _generate_report(gen)
        html = html_path.read_text(encoding="utf-8")

        for section_id in EXPECTED_SECTION_IDS:
            assert section_id in html, (
                f"Missing section id \'{section_id}\' in empty HTML report"
            )


# ── Property 13: JSON 报告数据往返一致性 ──


class TestJsonRoundTripProperty:
    """Property 13: JSON 报告数据往返一致性

    **Validates: Requirements 5.5**
    """

    @given(det=detection_result_strategy(), rules=rule_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_detection_metrics_round_trip(self, det, rules):
        """Detection metrics survive JSON serialization within float tolerance."""
        gen = ReportGenerator()
        _, output_dir = _generate_report(gen, detection_result=det, rule_result=rules)

        json_files = list(output_dir.glob("eval_*.json"))
        assert len(json_files) == 1, f"Expected 1 JSON file, found {len(json_files)}"

        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        d = data["detection"]

        assert d is not None, "detection key missing from JSON"
        assert abs(d["map50"] - det.map50) < 1e-6, (
            f"map50 mismatch: {d['map50']} vs {det.map50}"
        )
        assert abs(d["map50_95"] - det.map50_95) < 1e-6, (
            f"map50_95 mismatch: {d['map50_95']} vs {det.map50_95}"
        )
        assert abs(d["total_precision"] - det.total_precision) < 1e-6, (
            f"total_precision mismatch: {d['total_precision']} vs {det.total_precision}"
        )
        assert abs(d["total_recall"] - det.total_recall) < 1e-6, (
            f"total_recall mismatch: {d['total_recall']} vs {det.total_recall}"
        )

    @given(det=detection_result_strategy(), rules=rule_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_class_metrics_round_trip(self, det, rules):
        """Per-class metrics survive JSON serialization within float tolerance."""
        gen = ReportGenerator()
        _, output_dir = _generate_report(gen, detection_result=det, rule_result=rules)

        json_files = list(output_dir.glob("eval_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text(encoding="utf-8"))

        loaded_classes = data["detection"]["class_metrics"]
        assert len(loaded_classes) == len(det.class_metrics), (
            f"class_metrics length mismatch: {len(loaded_classes)} vs {len(det.class_metrics)}"
        )

        for orig, loaded in zip(det.class_metrics, loaded_classes):
            assert abs(loaded["ap"] - orig.ap) < 1e-6, (
                f"ap mismatch for {orig.class_name}"
            )
            assert abs(loaded["precision"] - orig.precision) < 1e-6, (
                f"precision mismatch for {orig.class_name}"
            )
            assert abs(loaded["recall"] - orig.recall) < 1e-6, (
                f"recall mismatch for {orig.class_name}"
            )
            assert loaded["tp"] == orig.tp
            assert loaded["fp"] == orig.fp
            assert loaded["fn"] == orig.fn

    @given(det=detection_result_strategy(), rules=rule_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rule_metrics_round_trip(self, det, rules):
        """Aggregated rule metrics survive JSON serialization within float tolerance."""
        gen = ReportGenerator()
        _, output_dir = _generate_report(gen, detection_result=det, rule_result=rules)

        json_files = list(output_dir.glob("eval_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text(encoding="utf-8"))

        loaded_agg = data["rules"]["aggregated"]
        assert len(loaded_agg) == len(rules.aggregated), (
            f"aggregated length mismatch: {len(loaded_agg)} vs {len(rules.aggregated)}"
        )

        for orig, loaded in zip(rules.aggregated, loaded_agg):
            assert loaded["rule_type"] == orig.rule_type, (
                f"rule_type mismatch: {loaded['rule_type']} vs {orig.rule_type}"
            )
            assert abs(loaded["precision"] - orig.precision) < 1e-6, (
                f"precision mismatch for {orig.rule_type}"
            )
            assert abs(loaded["recall"] - orig.recall) < 1e-6, (
                f"recall mismatch for {orig.rule_type}"
            )
            assert abs(loaded["f1"] - orig.f1) < 1e-6, (
                f"f1 mismatch for {orig.rule_type}"
            )
            assert loaded["tp"] == orig.tp
            assert loaded["fp"] == orig.fp
            assert loaded["fn"] == orig.fn

    @given(det=detection_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_detection_only_round_trip(self, det):
        """JSON round-trip works when only detection result is provided."""
        gen = ReportGenerator()
        _, output_dir = _generate_report(gen, detection_result=det, rule_result=None)

        json_files = list(output_dir.glob("eval_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text(encoding="utf-8"))

        assert data["detection"] is not None
        assert data["rules"] is None
        assert abs(data["detection"]["map50"] - det.map50) < 1e-6

    @given(rules=rule_result_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rules_only_round_trip(self, rules):
        """JSON round-trip works when only rule result is provided."""
        gen = ReportGenerator()
        _, output_dir = _generate_report(gen, detection_result=None, rule_result=rules)

        json_files = list(output_dir.glob("eval_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text(encoding="utf-8"))

        assert data["detection"] is None
        assert data["rules"] is not None
        assert len(data["rules"]["aggregated"]) == len(rules.aggregated)
