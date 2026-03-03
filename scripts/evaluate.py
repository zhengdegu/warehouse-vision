#!/usr/bin/env python3
"""Evaluation orchestration script.

Runs image detection evaluation, video rule evaluation, or both,
then generates a unified HTML + JSON report.

Usage:
    python scripts/evaluate.py --detection
    python scripts/evaluate.py --rules
    python scripts/evaluate.py --all
    python scripts/evaluate.py  # auto-detect available data
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_CONFIG = Path("configs/cameras.yaml")
DEFAULT_SAMPLES = "data/samples/warehouse_v1"
DEFAULT_LABELS = "data/labels/warehouse_v1"
DEFAULT_VIDEOS = "data/eval_videos"
DEFAULT_OUTPUT = "runs/eval"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.5
DEFAULT_TIME_TOLERANCE = 2.0


def _load_config_defaults() -> dict:
    """Load defaults from cameras.yaml if available."""
    defaults = {
        "model": "yolo26m.pt",
        "pose_model": None,
        "confidence": DEFAULT_CONFIDENCE,
    }
    if DEFAULT_CONFIG.exists():
        try:
            with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            model_cfg = cfg.get("model", {})
            if "path" in model_cfg:
                defaults["model"] = model_cfg["path"]
            if "confidence" in model_cfg:
                defaults["confidence"] = float(model_cfg["confidence"])
            pose_cfg = model_cfg.get("pose", {})
            if "path" in pose_cfg:
                defaults["pose_model"] = pose_cfg["path"]
        except Exception:
            pass
    return defaults


def parse_args(argv: list = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    cfg = _load_config_defaults()

    parser = argparse.ArgumentParser(
        description="Run accuracy evaluation for detection and/or rule engines.",
    )
    parser.add_argument(
        "--detection", action="store_true",
        help="Run image detection evaluation only",
    )
    parser.add_argument(
        "--rules", action="store_true",
        help="Run video rule engine evaluation only",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both detection and rule evaluation",
    )
    parser.add_argument(
        "--model", default=cfg["model"],
        help=f"Detection model path (default: {cfg['model']})",
    )
    parser.add_argument(
        "--pose-model", default=cfg["pose_model"],
        help="Pose model path",
    )
    parser.add_argument(
        "--confidence", type=float, default=cfg["confidence"],
        help=f"Confidence threshold (default: {cfg['confidence']})",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"IoU threshold for detection eval (default: {DEFAULT_IOU})",
    )
    parser.add_argument(
        "--time-tolerance", type=float, default=DEFAULT_TIME_TOLERANCE,
        help=f"Time tolerance in seconds for rule eval (default: {DEFAULT_TIME_TOLERANCE})",
    )
    parser.add_argument(
        "--samples", default=DEFAULT_SAMPLES,
        help=f"Image samples directory (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABELS,
        help=f"Image labels directory (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--videos", default=DEFAULT_VIDEOS,
        help=f"Video evaluation directory (default: {DEFAULT_VIDEOS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Report output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Do not auto-open the HTML report",
    )
    return parser.parse_args(argv)


def validate_paths(
    run_detection: bool,
    run_rules: bool,
    model_path: str,
    pose_model_path: Optional[str],
    samples_dir: str,
    labels_dir: str,
    videos_dir: str,
) -> List[str]:
    """Validate that required paths exist before starting evaluation.

    Returns a list of error messages. An empty list means all paths are valid.
    """
    errors: List[str] = []

    # Model file validation
    if run_detection or run_rules:
        if not Path(model_path).exists():
            errors.append(f"Model file not found: {model_path}")

    if run_rules and pose_model_path:
        if not Path(pose_model_path).exists():
            errors.append(f"Pose model file not found: {pose_model_path}")

    # Detection data validation
    if run_detection:
        samples_path = Path(samples_dir)
        labels_path = Path(labels_dir)

        if not samples_path.exists():
            errors.append(f"Samples directory not found: {samples_dir}")
        elif not any(samples_path.iterdir()):
            errors.append(f"Samples directory is empty: {samples_dir}")

        if not labels_path.exists():
            errors.append(f"Labels directory not found: {labels_dir}")
        elif not any(labels_path.iterdir()):
            errors.append(f"Labels directory is empty: {labels_dir}")

    # Rule evaluation data validation
    if run_rules:
        videos_path = Path(videos_dir)
        if not videos_path.exists():
            errors.append(f"Videos directory not found: {videos_dir}")
        elif not any(videos_path.iterdir()):
            errors.append(f"Videos directory is empty: {videos_dir}")

    return errors


def _detect_available_data(samples_dir: str, labels_dir: str, videos_dir: str):
    """Auto-detect which evaluations can run based on available data."""
    run_detection = False
    run_rules = False

    samples_path = Path(samples_dir)
    labels_path = Path(labels_dir)
    if samples_path.exists() and labels_path.exists():
        has_images = any(
            f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            for f in samples_path.iterdir() if f.is_file()
        )
        has_labels = any(
            f.suffix.lower() == ".txt"
            for f in labels_path.iterdir() if f.is_file()
        )
        if has_images and has_labels:
            run_detection = True

    videos_path = Path(videos_dir)
    if videos_path.exists():
        has_videos = any(
            f.suffix.lower() in (".mp4", ".avi", ".mkv")
            for f in videos_path.iterdir() if f.is_file()
        )
        has_annotations = any(
            f.suffix.lower() == ".json" and f.stem != "annotation_template"
            for f in videos_path.iterdir() if f.is_file()
        )
        if has_videos and has_annotations:
            run_rules = True

    return run_detection, run_rules


def main(argv: list = None) -> int:
    """Entry point for the evaluation script."""
    args = parse_args(argv)

    # Determine what to run
    run_detection = args.detection or args.all
    run_rules = args.rules or args.all

    # Auto-detect if no explicit flags
    if not (args.detection or args.rules or args.all):
        run_detection, run_rules = _detect_available_data(
            args.samples, args.labels, args.videos,
        )
        if not run_detection and not run_rules:
            print("No evaluation data found. Provide --detection, --rules, or --all.")
            return 1

    # Validate paths
    errors = validate_paths(
        run_detection=run_detection,
        run_rules=run_rules,
        model_path=args.model,
        pose_model_path=args.pose_model,
        samples_dir=args.samples,
        labels_dir=args.labels,
        videos_dir=args.videos,
    )
    if errors:
        print("Validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # Lazy imports to avoid loading heavy dependencies when just validating
    from src.evaluation.detection_eval import DetectionEvaluator
    from src.evaluation.report import ReportGenerator
    from src.evaluation.rule_eval import RuleEvaluator

    detection_result = None
    rule_result = None

    # Run detection evaluation
    if run_detection:
        print("=== Image Detection Evaluation ===")
        evaluator = DetectionEvaluator(
            model_path=args.model,
            confidence=args.confidence,
            iou_threshold=args.iou,
        )
        total_images = len(list(Path(args.samples).iterdir()))

        def det_progress(idx, name):
            print(f"  [{idx}/{total_images}] processing {name} ...")

        detection_result = evaluator.run(
            image_dir=Path(args.samples),
            label_dir=Path(args.labels),
            progress_callback=det_progress,
        )
        print()

    # Run rule evaluation
    if run_rules:
        print("=== Video Rule Engine Evaluation ===")
        evaluator = RuleEvaluator(
            model_path=args.model,
            pose_model_path=args.pose_model,
            confidence=args.confidence,
            time_tolerance=args.time_tolerance,
        )
        rule_result = evaluator.run(
            videos_dir=Path(args.videos),
        )
        print()

    # Generate report
    print("=== Generating Report ===")
    metadata = {
        "model": args.model,
        "confidence": args.confidence,
        "iou_threshold": args.iou,
        "time_tolerance": args.time_tolerance,
    }
    generator = ReportGenerator()
    html_path = generator.generate(
        detection_result=detection_result,
        rule_result=rule_result,
        metadata=metadata,
        output_dir=Path(args.output),
    )

    # Auto-open report
    if not args.no_open:
        webbrowser.open(str(html_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
