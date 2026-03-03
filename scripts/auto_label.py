#!/usr/bin/env python3
"""Batch pre-annotation script using YOLODetector.

Scans an image directory, runs detection, and writes YOLO format label files.
Designed for generating initial annotations that are then manually reviewed.

Usage:
    python scripts/auto_label.py
    python scripts/auto_label.py --model custom.pt --confidence 0.5
    python scripts/auto_label.py --classes 0,2,7 --overwrite data/images data/labels
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.evaluation.auto_labeler import AutoLabeler

DEFAULT_CONFIG = Path("configs/cameras.yaml")
DEFAULT_IMAGES = "data/samples/warehouse_v1"
DEFAULT_LABELS = "data/labels/warehouse_v1"
DEFAULT_CLASSES = "0,1,2,3,5,7"
DEFAULT_CONFIDENCE = 0.3


def _load_config_defaults() -> dict:
    """Load defaults from cameras.yaml if available."""
    defaults = {
        "model": "yolo26m.pt",
        "confidence": DEFAULT_CONFIDENCE,
        "classes": DEFAULT_CLASSES,
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
            if "classes" in model_cfg:
                defaults["classes"] = ",".join(str(c) for c in model_cfg["classes"])
        except Exception:
            pass
    return defaults


def parse_args(argv: list = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    cfg = _load_config_defaults()

    parser = argparse.ArgumentParser(
        description="Generate YOLO format pre-annotations using YOLODetector.",
    )
    parser.add_argument(
        "--model",
        default=cfg["model"],
        help=f"Path to YOLO model weights (default: {cfg['model']})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=cfg["confidence"],
        help=f"Detection confidence threshold (default: {cfg['confidence']})",
    )
    parser.add_argument(
        "--classes",
        default=cfg["classes"],
        help=f"Comma-separated class IDs to detect (default: {cfg['classes']})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files",
    )
    parser.add_argument(
        "images",
        nargs="?",
        default=DEFAULT_IMAGES,
        help=f"Image directory (default: {DEFAULT_IMAGES})",
    )
    parser.add_argument(
        "labels",
        nargs="?",
        default=DEFAULT_LABELS,
        help=f"Label output directory (default: {DEFAULT_LABELS})",
    )
    return parser.parse_args(argv)


def main(argv: list = None) -> int:
    """Entry point for the auto-label script."""
    args = parse_args(argv)

    image_dir = Path(args.images)
    label_dir = Path(args.labels)

    if not image_dir.exists():
        print(f"Error: image directory not found: {image_dir}", file=sys.stderr)
        return 1

    classes = [int(c.strip()) for c in args.classes.split(",")]

    print(f"Model:      {args.model}")
    print(f"Confidence: {args.confidence}")
    print(f"Classes:    {classes}")
    print(f"Images:     {image_dir}")
    print(f"Labels:     {label_dir}")
    print(f"Overwrite:  {args.overwrite}")
    print()

    labeler = AutoLabeler(
        model_path=args.model,
        confidence=args.confidence,
        classes=classes,
    )

    summary = labeler.label_directory(
        image_dir=image_dir,
        label_dir=label_dir,
        overwrite=args.overwrite,
    )

    print("=== Summary ===")
    print(f"Total images: {summary.total_images}")
    print(f"Skipped:      {summary.skipped}")
    print(f"Labeled:      {summary.labeled}")
    if summary.class_counts:
        print("Class counts:")
        for cls_name, count in sorted(summary.class_counts.items()):
            print(f"  {cls_name}: {count}")
    else:
        print("Class counts: (none)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
