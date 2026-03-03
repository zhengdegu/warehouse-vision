"""Report generator for accuracy evaluation framework.

Generates self-contained HTML + JSON reports with inline CSS/JS,
tab navigation, and confusion matrix heatmap.
"""

import json
import html as html_mod
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.evaluation.models import (
    DetectionResult,
    RuleResult,
    RuleMetrics,
    CountingDetail,
)


class ReportGenerator:
    """Generates HTML + JSON evaluation reports."""

    # Section identifiers used in tab navigation
    SECTION_IDS = [
        "overview",
        "detection",
        "intrusion",
        "tripwire",
        "counting",
        "presence",
        "anomaly",
    ]

    SECTION_LABELS = {
        "overview": "总览仪表盘",
        "detection": "目标检测",
        "intrusion": "入侵检测",
        "tripwire": "越线检测",
        "counting": "计数统计",
        "presence": "存在检测",
        "anomaly": "异常检测",
    }

    def generate(
        self,
        detection_result: DetectionResult = None,
        rule_result: RuleResult = None,
        metadata: dict = None,
        output_dir: Path = Path("runs/eval"),
    ) -> Path:
        """Generate HTML + JSON report. Returns HTML file path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = output_dir / f"eval_{timestamp}.html"
        json_path = output_dir / f"eval_{timestamp}.json"

        # Build report data dict
        data = self._build_data(detection_result, rule_result, metadata)

        # Write JSON
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        # Write HTML
        html_content = self._render_html(data)
        html_path.write_text(html_content, encoding="utf-8")

        # Print terminal summary
        self._print_summary(data, html_path)

        return html_path

    # ── Data assembly ──

    def _build_data(
        self,
        detection_result: Optional[DetectionResult],
        rule_result: Optional[RuleResult],
        metadata: Optional[dict],
    ) -> dict:
        """Assemble all report data into a single dict."""
        data: dict = {
            "metadata": metadata or {},
            "generated_at": datetime.now().isoformat(),
            "detection": None,
            "rules": None,
        }

        if detection_result is not None:
            data["detection"] = asdict(detection_result)

        if rule_result is not None:
            data["rules"] = asdict(rule_result)

        return data

    # ── HTML rendering ──

    def _render_html(self, data: dict) -> str:
        """Render HTML string (inline CSS/JS, Tab navigation)."""
        meta = data.get("metadata", {})
        detection = data.get("detection")
        rules = data.get("rules")

        sections_html = []
        sections_html.append(self._render_overview(detection, rules, meta))
        sections_html.append(self._render_detection_section(detection))
        sections_html.append(self._render_rule_section("intrusion", "入侵检测", rules))
        sections_html.append(self._render_rule_section("tripwire", "越线检测", rules))
        sections_html.append(self._render_counting_section(rules))
        sections_html.append(self._render_rule_section("presence", "存在检测", rules))
        sections_html.append(self._render_anomaly_section(rules))

        tabs_html = self._render_tabs()
        meta_html = self._render_metadata(meta, data.get("generated_at", ""))

        return self._wrap_page(meta_html, tabs_html, "\n".join(sections_html))

    def _render_tabs(self) -> str:
        """Render tab navigation bar."""
        buttons = []
        for i, sid in enumerate(self.SECTION_IDS):
            active = ' class="active"' if i == 0 else ""
            label = self.SECTION_LABELS[sid]
            buttons.append(f'<button{active} onclick="showTab(\'{sid}\')">{label}</button>')
        return '<div class="tabs">' + "".join(buttons) + "</div>"

    def _render_metadata(self, meta: dict, generated_at: str) -> str:
        """Render report header with evaluation metadata."""
        rows = []
        rows.append(f"<tr><td>评估时间</td><td>{html_mod.escape(str(generated_at))}</td></tr>")
        field_labels = {
            "model": "模型",
            "confidence": "置信度阈值",
            "iou_threshold": "IoU 阈值",
            "time_tolerance": "时间容差",
            "image_count": "图片样本数",
            "video_count": "视频片段数",
        }
        for key, label in field_labels.items():
            if key in meta:
                rows.append(f"<tr><td>{label}</td><td>{html_mod.escape(str(meta[key]))}</td></tr>")
        return '<div class="metadata"><h2>评估元数据</h2><table>' + "".join(rows) + "</table></div>"

    # ── Overview dashboard ──

    def _render_overview(self, detection: dict, rules: dict, meta: dict) -> str:
        """Render overview dashboard section."""
        cards = []

        # Detection status
        if detection is not None:
            cards.append(self._status_card(
                "目标检测", "✅ 已评估",
                f"mAP50: {detection['map50']:.3f} | P: {detection['total_precision']:.3f} | R: {detection['total_recall']:.3f}",
            ))
        else:
            cards.append(self._status_card("目标检测", "⏭ 未评估", ""))

        # Rule types
        rule_types = ["intrusion", "tripwire", "counting", "presence", "anomaly"]
        rule_labels = {
            "intrusion": "入侵检测", "tripwire": "越线检测",
            "counting": "计数统计", "presence": "存在检测", "anomaly": "异常检测",
        }
        for rt in rule_types:
            metric = self._find_rule_metric(rules, rt)
            if metric is None:
                cards.append(self._status_card(rule_labels[rt], "⏭ 未评估", ""))
            elif not metric.get("has_samples", True):
                cards.append(self._status_card(rule_labels[rt], "⚠ 无样本", "N/A"))
            else:
                summary = f"P: {metric['precision']:.3f} | R: {metric['recall']:.3f} | F1: {metric['f1']:.3f}"
                cards.append(self._status_card(rule_labels[rt], "✅ 已评估", summary))

        content = '<div class="cards">' + "".join(cards) + "</div>"
        return self._section_wrapper("overview", content)

    def _status_card(self, title: str, status: str, detail: str) -> str:
        return (
            f'<div class="card"><h3>{html_mod.escape(title)}</h3>'
            f'<div class="status">{html_mod.escape(status)}</div>'
            f'<div class="detail">{html_mod.escape(detail)}</div></div>'
        )

    # ── Detection section ──

    def _render_detection_section(self, detection: dict) -> str:
        if detection is None:
            return self._section_wrapper("detection", self._no_data_msg())

        parts = []

        # Metric cards
        cards = (
            f'<div class="cards">'
            f'<div class="card"><h3>mAP50</h3><div class="value">{detection["map50"]:.4f}</div></div>'
            f'<div class="card"><h3>mAP50-95</h3><div class="value">{detection["map50_95"]:.4f}</div></div>'
            f'<div class="card"><h3>Precision</h3><div class="value">{detection["total_precision"]:.4f}</div></div>'
            f'<div class="card"><h3>Recall</h3><div class="value">{detection["total_recall"]:.4f}</div></div>'
            f'</div>'
        )
        parts.append(cards)

        # Per-class table
        class_metrics = detection.get("class_metrics", [])
        if class_metrics:
            rows = []
            for cm in class_metrics:
                na = "" if cm.get("has_samples", True) else " (N/A)"
                rows.append(
                    f"<tr><td>{html_mod.escape(str(cm['class_name']))}{na}</td>"
                    f"<td>{cm['ap']:.4f}</td><td>{cm['precision']:.4f}</td>"
                    f"<td>{cm['recall']:.4f}</td><td>{cm['tp']}</td>"
                    f"<td>{cm['fp']}</td><td>{cm['fn']}</td>"
                    f"<td>{cm['gt_count']}</td></tr>"
                )
            parts.append(
                '<h3>各类别指标</h3><table class="data-table">'
                "<tr><th>类别</th><th>AP</th><th>Precision</th><th>Recall</th>"
                "<th>TP</th><th>FP</th><th>FN</th><th>GT数</th></tr>"
                + "".join(rows) + "</table>"
            )

        # Confusion matrix
        cm_data = detection.get("confusion_matrix")
        if cm_data:
            matrix = np.array(cm_data)
            labels = [str(c.get("class_name", c.get("class_id", "?"))) for c in class_metrics]
            labels.append("背景")
            parts.append("<h3>混淆矩阵</h3>" + self._render_confusion_matrix(matrix, labels))

        # Per-image table
        per_image = detection.get("per_image", [])
        if per_image:
            rows = []
            for img in per_image:
                rows.append(
                    f"<tr><td>{html_mod.escape(str(img['filename']))}</td>"
                    f"<td>{img['gt_count']}</td><td>{img['det_count']}</td>"
                    f"<td>{img['tp_count']}</td></tr>"
                )
            parts.append(
                '<h3>逐图检测对比</h3><table class="data-table sortable">'
                "<tr><th>文件名</th><th>GT数</th><th>检测数</th><th>TP数</th></tr>"
                + "".join(rows) + "</table>"
            )

        return self._section_wrapper("detection", "\n".join(parts))

    # ── Confusion matrix heatmap ──

    def _render_confusion_matrix(self, matrix: np.ndarray, labels: List[str]) -> str:
        """Render confusion matrix heatmap HTML fragment."""
        if matrix.size == 0:
            return "<p>无混淆矩阵数据</p>"

        max_val = float(np.max(matrix)) if np.max(matrix) > 0 else 1.0
        rows_html = []

        # Header row
        header_cells = "<th></th>" + "".join(
            f"<th>{html_mod.escape(str(l))}</th>" for l in labels
        )
        rows_html.append(f"<tr>{header_cells}</tr>")

        # Data rows
        for i, row_label in enumerate(labels):
            cells = [f"<th>{html_mod.escape(str(row_label))}</th>"]
            for j in range(matrix.shape[1]):
                val = int(matrix[i][j])
                intensity = val / max_val if max_val > 0 else 0
                # Blue heatmap: higher values = darker
                r = int(255 * (1 - intensity * 0.8))
                g = int(255 * (1 - intensity * 0.8))
                b = 255
                color = f"#{r:02x}{g:02x}{b:02x}"
                text_color = "#000" if intensity < 0.6 else "#fff"
                cells.append(
                    f'<td style="background:{color};color:{text_color};text-align:center;padding:8px;">{val}</td>'
                )
            rows_html.append("<tr>" + "".join(cells) + "</tr>")

        return (
            '<table class="confusion-matrix">'
            + "".join(rows_html)
            + "</table>"
        )

    # ── Rule sections ──

    def _render_rule_section(self, rule_type: str, title: str, rules: dict) -> str:
        """Render a generic rule evaluation section (intrusion, tripwire, presence)."""
        metric = self._find_rule_metric(rules, rule_type)
        if metric is None:
            return self._section_wrapper(rule_type, self._no_data_msg())

        if not metric.get("has_samples", True):
            return self._section_wrapper(rule_type, self._no_data_msg())

        parts = []

        # Metric cards
        cards = (
            f'<div class="cards">'
            f'<div class="card"><h3>Precision</h3><div class="value">{metric["precision"]:.4f}</div></div>'
            f'<div class="card"><h3>Recall</h3><div class="value">{metric["recall"]:.4f}</div></div>'
            f'<div class="card"><h3>F1-Score</h3><div class="value">{metric["f1"]:.4f}</div></div>'
            f'<div class="card"><h3>TP / FP / FN</h3><div class="value">{metric["tp"]} / {metric["fp"]} / {metric["fn"]}</div></div>'
            f'</div>'
        )
        parts.append(cards)

        # Details (direction accuracy for tripwire, etc.)
        details = metric.get("details", {})
        if details:
            detail_rows = []
            for k, v in details.items():
                if isinstance(v, list):
                    # Render list items as sub-table (e.g., false positives/negatives)
                    items = "".join(f"<li>{html_mod.escape(str(item))}</li>" for item in v[:20])
                    detail_rows.append(f"<tr><td>{html_mod.escape(str(k))}</td><td><ul>{items}</ul></td></tr>")
                elif isinstance(v, dict):
                    detail_rows.append(f"<tr><td>{html_mod.escape(str(k))}</td><td>{html_mod.escape(json.dumps(v, ensure_ascii=False))}</td></tr>")
                else:
                    detail_rows.append(f"<tr><td>{html_mod.escape(str(k))}</td><td>{html_mod.escape(str(v))}</td></tr>")
            if detail_rows:
                parts.append(
                    '<h3>详情</h3><table class="data-table">'
                    + "".join(detail_rows) + "</table>"
                )

        return self._section_wrapper(rule_type, "\n".join(parts))

    def _render_counting_section(self, rules: dict) -> str:
        """Render counting statistics section."""
        metric = self._find_rule_metric(rules, "counting")
        if metric is None:
            return self._section_wrapper("counting", self._no_data_msg())

        if not metric.get("has_samples", True):
            return self._section_wrapper("counting", self._no_data_msg())

        parts = []
        details = metric.get("details", {})
        counting_details = details.get("counting_details", [])

        if counting_details:
            rows = []
            for cd in counting_details:
                if isinstance(cd, dict):
                    rows.append(
                        f"<tr><td>{html_mod.escape(str(cd.get('class_name', '')))}</td>"
                        f"<td>{cd.get('expected_in', 0)}</td><td>{cd.get('actual_in', 0)}</td>"
                        f"<td>{cd.get('abs_error_in', 0)}</td><td>{cd.get('rel_error_in', 0):.2%}</td>"
                        f"<td>{cd.get('expected_out', 0)}</td><td>{cd.get('actual_out', 0)}</td>"
                        f"<td>{cd.get('abs_error_out', 0)}</td><td>{cd.get('rel_error_out', 0):.2%}</td></tr>"
                    )
            if rows:
                parts.append(
                    '<table class="data-table">'
                    "<tr><th>类别</th><th>预期IN</th><th>实际IN</th><th>绝对误差IN</th><th>相对误差IN</th>"
                    "<th>预期OUT</th><th>实际OUT</th><th>绝对误差OUT</th><th>相对误差OUT</th></tr>"
                    + "".join(rows) + "</table>"
                )
        else:
            # Fallback: show P/R/F1 cards like other rule sections
            cards = (
                f'<div class="cards">'
                f'<div class="card"><h3>Precision</h3><div class="value">{metric["precision"]:.4f}</div></div>'
                f'<div class="card"><h3>Recall</h3><div class="value">{metric["recall"]:.4f}</div></div>'
                f'<div class="card"><h3>F1-Score</h3><div class="value">{metric["f1"]:.4f}</div></div>'
                f'</div>'
            )
            parts.append(cards)

        if not parts:
            return self._section_wrapper("counting", self._no_data_msg())

        return self._section_wrapper("counting", "\n".join(parts))

    def _render_anomaly_section(self, rules: dict) -> str:
        """Render anomaly detection section with per-subtype breakdown."""
        if rules is None:
            return self._section_wrapper("anomaly", self._no_data_msg())

        aggregated = rules.get("aggregated", [])
        anomaly_metrics = [m for m in aggregated if m.get("rule_type") == "anomaly"]

        if not anomaly_metrics:
            return self._section_wrapper("anomaly", self._no_data_msg())

        parts = []
        rows = []
        for m in anomaly_metrics:
            if not m.get("has_samples", True):
                continue
            sub = m.get("sub_type", "")
            rows.append(
                f"<tr><td>{html_mod.escape(sub)}</td>"
                f"<td>{m['precision']:.4f}</td><td>{m['recall']:.4f}</td>"
                f"<td>{m['f1']:.4f}</td><td>{m['tp']}</td>"
                f"<td>{m['fp']}</td><td>{m['fn']}</td></tr>"
            )

        if rows:
            parts.append(
                '<table class="data-table">'
                "<tr><th>子类型</th><th>Precision</th><th>Recall</th>"
                "<th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr>"
                + "".join(rows) + "</table>"
            )
        else:
            parts.append(self._no_data_msg())

        return self._section_wrapper("anomaly", "\n".join(parts))

    # ── Helpers ──

    def _find_rule_metric(self, rules: dict, rule_type: str) -> Optional[dict]:
        """Find aggregated metric for a rule type."""
        if rules is None:
            return None
        aggregated = rules.get("aggregated", [])
        for m in aggregated:
            if m.get("rule_type") == rule_type:
                return m
        return None

    def _no_data_msg(self) -> str:
        return '<div class="no-data">未评估 — 无可用数据</div>'

    def _section_wrapper(self, section_id: str, content: str) -> str:
        display = "block" if section_id == "overview" else "none"
        return (
            f'<div id="section-{section_id}" class="section" style="display:{display};">'
            f"<h2>{self.SECTION_LABELS.get(section_id, section_id)}</h2>"
            f"{content}</div>"
        )

    def _wrap_page(self, meta_html: str, tabs_html: str, sections_html: str) -> str:
        """Wrap all content in a full HTML page with inline CSS/JS."""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>准确率评估报告</title>
<style>
{self._inline_css()}
</style>
</head>
<body>
<h1>准确率评估报告</h1>
{meta_html}
{tabs_html}
{sections_html}
<script>
{self._inline_js()}
</script>
</body>
</html>"""

    @staticmethod
    def _inline_css() -> str:
        return """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 20px; background: #f5f5f5; color: #333; }
h1 { text-align: center; margin-bottom: 10px; }
h2 { border-bottom: 2px solid #4a90d9; padding-bottom: 6px; margin-top: 20px; }
h3 { margin-top: 16px; }
.metadata { background: #fff; padding: 12px 16px; border-radius: 6px;
            margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.metadata table { width: 100%; border-collapse: collapse; }
.metadata td { padding: 4px 8px; border-bottom: 1px solid #eee; }
.metadata td:first-child { font-weight: bold; width: 140px; }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 16px; }
.tabs button { padding: 8px 16px; border: 1px solid #ccc; background: #fff;
               cursor: pointer; border-radius: 4px 4px 0 0; font-size: 14px; }
.tabs button.active { background: #4a90d9; color: #fff; border-color: #4a90d9; }
.section { background: #fff; padding: 16px; border-radius: 0 6px 6px 6px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
.card { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px;
        padding: 12px 16px; min-width: 140px; flex: 1; text-align: center; }
.card h3 { margin: 0 0 6px 0; font-size: 13px; color: #666; }
.card .value { font-size: 22px; font-weight: bold; color: #2c3e50; }
.card .status { font-size: 16px; margin: 4px 0; }
.card .detail { font-size: 12px; color: #666; }
.data-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 14px; }
.data-table th, .data-table td { padding: 6px 10px; border: 1px solid #ddd; text-align: left; }
.data-table th { background: #4a90d9; color: #fff; }
.data-table tr:nth-child(even) { background: #f8f9fa; }
.confusion-matrix { border-collapse: collapse; margin: 10px 0; }
.confusion-matrix th { background: #4a90d9; color: #fff; padding: 8px; font-size: 13px; }
.no-data { padding: 24px; text-align: center; color: #999; font-size: 16px;
           background: #fafafa; border-radius: 6px; border: 1px dashed #ddd; }
ul { margin: 4px 0; padding-left: 20px; }
"""

    @staticmethod
    def _inline_js() -> str:
        return """
function showTab(id) {
    var sections = document.querySelectorAll('.section');
    for (var i = 0; i < sections.length; i++) {
        sections[i].style.display = 'none';
    }
    var target = document.getElementById('section-' + id);
    if (target) target.style.display = 'block';
    var buttons = document.querySelectorAll('.tabs button');
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].classList.remove('active');
        if (buttons[i].textContent === getLabel(id)) {
            buttons[i].classList.add('active');
        }
    }
}
function getLabel(id) {
    var map = {
        'overview': '总览仪表盘', 'detection': '目标检测',
        'intrusion': '入侵检测', 'tripwire': '越线检测',
        'counting': '计数统计', 'presence': '存在检测',
        'anomaly': '异常检测'
    };
    return map[id] || id;
}
"""

    # ── Terminal summary ──

    def _print_summary(self, data: dict, html_path: Path) -> None:
        """Print brief summary to terminal."""
        print("\n" + "=" * 60)
        print("  评估报告摘要")
        print("=" * 60)

        detection = data.get("detection")
        if detection:
            print(f"  目标检测: mAP50={detection['map50']:.4f}  "
                  f"P={detection['total_precision']:.4f}  "
                  f"R={detection['total_recall']:.4f}")
        else:
            print("  目标检测: 未评估")

        rules = data.get("rules")
        if rules:
            for m in rules.get("aggregated", []):
                rt = m.get("rule_type", "")
                st = m.get("sub_type", "")
                label = f"{rt}/{st}" if st else rt
                if m.get("has_samples", True):
                    print(f"  {label}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
                else:
                    print(f"  {label}: N/A (无样本)")
        else:
            print("  规则评估: 未评估")

        print("-" * 60)
        print(f"  报告路径: {html_path}")
        print("=" * 60 + "\n")
