"""Annotation loading and validation for the accuracy evaluation framework."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from src.evaluation.models import GroundTruthBox, VideoAnnotation

logger = logging.getLogger(__name__)

# ── Constants ──

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VALID_EVENT_TYPES = {"intrusion", "tripwire", "counting", "presence", "anomaly"}
VALID_ANOMALY_SUB_TYPES = {
    "dwell", "crowd", "proximity", "fight", "fall", "wrong_way", "speed",
}


# ── Exceptions ──


class ValidationError(Exception):
    """Raised when video annotation validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed with {len(errors)} error(s): " + "; ".join(errors))


# ── Helper functions (exposed for property testing) ──


def serialize_yolo_line(class_id: int, cx_norm: float, cy_norm: float,
                        w_norm: float, h_norm: float) -> str:
    """Serialize a YOLO detection to a single annotation line.

    Args:
        class_id: Class identifier.
        cx_norm: Normalized center x (0-1).
        cy_norm: Normalized center y (0-1).
        w_norm: Normalized width (0-1).
        h_norm: Normalized height (0-1).

    Returns:
        YOLO format string: "class_id cx cy w h"
    """
    return f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"


def parse_yolo_line(line: str, img_width: int, img_height: int) -> GroundTruthBox:
    """Parse a single YOLO annotation line into a GroundTruthBox with pixel coords.

    Args:
        line: YOLO format line "class_id cx cy w h" (normalized).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        GroundTruthBox with pixel coordinates (x1, y1, x2, y2).

    Raises:
        ValueError: If the line format is invalid.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 values, got {len(parts)}: {line!r}")

    class_id = int(parts[0])
    cx_norm = float(parts[1])
    cy_norm = float(parts[2])
    w_norm = float(parts[3])
    h_norm = float(parts[4])

    # Convert normalized center coords to pixel corner coords
    cx_px = cx_norm * img_width
    cy_px = cy_norm * img_height
    w_px = w_norm * img_width
    h_px = h_norm * img_height

    x1 = cx_px - w_px / 2.0
    y1 = cy_px - h_px / 2.0
    x2 = cx_px + w_px / 2.0
    y2 = cy_px + h_px / 2.0

    return GroundTruthBox(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2)


def pixel_to_yolo(class_id: int, x1: float, y1: float, x2: float, y2: float,
                  img_width: int, img_height: int) -> str:
    """Convert pixel coordinates to YOLO format string (for round-trip testing).

    Args:
        class_id: Class identifier.
        x1, y1, x2, y2: Pixel coordinates of the bounding box.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        YOLO format string: "class_id cx cy w h" (normalized).
    """
    w_px = x2 - x1
    h_px = y2 - y1
    cx_px = x1 + w_px / 2.0
    cy_px = y1 + h_px / 2.0

    cx_norm = cx_px / img_width
    cy_norm = cy_px / img_height
    w_norm = w_px / img_width
    h_norm = h_px / img_height

    return serialize_yolo_line(class_id, cx_norm, cy_norm, w_norm, h_norm)


def match_files(image_dir: Path, label_dir: Path
                ) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """Match image files to label files by filename stem.

    Args:
        image_dir: Directory containing image files.
        label_dir: Directory containing YOLO label .txt files.

    Returns:
        Tuple of (matched_pairs, unmatched_images) where:
        - matched_pairs: list of (image_path, label_path) tuples
        - unmatched_images: list of image paths with no matching label
    """
    matched = []
    unmatched = []

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.is_file():
            matched.append((img_path, label_path))
        else:
            unmatched.append(img_path)

    return matched, unmatched


# ── AnnotationLoader ──


class AnnotationLoader:
    """Loads and validates annotation files for evaluation."""

    @staticmethod
    def load_yolo_labels(label_dir: Path, image_dir: Path
                         ) -> Tuple[Dict[str, List[GroundTruthBox]], List[str]]:
        """Load YOLO format labels, matching by filename to images.

        Scans image_dir for image files, matches them to .txt label files in
        label_dir by filename stem. Parses YOLO format lines (ignoring comment
        lines starting with #) and converts normalized coords to pixel coords.

        Args:
            label_dir: Directory containing YOLO .txt label files.
            image_dir: Directory containing image files.

        Returns:
            Tuple of:
            - dict mapping filename stem to list of GroundTruthBox
            - list of skipped image filenames (no matching label)
        """
        matched, unmatched = match_files(image_dir, label_dir)
        skipped = [p.name for p in unmatched]

        labels: Dict[str, List[GroundTruthBox]] = {}

        for img_path, label_path in matched:
            stem = img_path.stem
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
                img.close()
            except Exception as e:
                logger.warning("Failed to read image %s: %s", img_path, e)
                skipped.append(img_path.name)
                continue

            boxes: List[GroundTruthBox] = []
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        try:
                            box = parse_yolo_line(line, img_width, img_height)
                            boxes.append(box)
                        except ValueError as e:
                            logger.warning(
                                "Skipping invalid line in %s: %s", label_path, e
                            )
            except Exception as e:
                logger.warning("Failed to read label file %s: %s", label_path, e)
                skipped.append(img_path.name)
                continue

            labels[stem] = boxes

        return labels, skipped

    @staticmethod
    def load_video_annotation(json_path: Path) -> VideoAnnotation:
        """Load and validate a video event annotation JSON file.

        Args:
            json_path: Path to the JSON annotation file.

        Returns:
            VideoAnnotation dataclass instance.

        Raises:
            ValidationError: If validation fails (contains all error details).
            FileNotFoundError: If the JSON file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        errors = AnnotationLoader.validate_video_annotation(data)
        if errors:
            raise ValidationError(errors)

        return VideoAnnotation(
            video_path=data["video_path"],
            fps_sample=data.get("fps_sample", 5),
            camera_config=data.get("camera_config", {}),
            events=data.get("events", []),
        )

    @staticmethod
    def validate_video_annotation(data: dict) -> List[str]:
        """Validate video annotation data structure.

        Checks:
        - Required fields: video_path, events
        - Event type validity (intrusion/tripwire/counting/presence/anomaly)
        - Anomaly sub_type validity
        - Tripwire events reference valid tripwire_id from camera_config
        - time_range start < end

        Args:
            data: Parsed JSON annotation data.

        Returns:
            List of error strings (empty if valid).
        """
        errors: List[str] = []

        # Check required fields
        if "video_path" not in data:
            errors.append("Missing required field: 'video_path'")
        if "events" not in data:
            errors.append("Missing required field: 'events'")
            return errors  # Can't validate events if missing

        events = data.get("events", [])
        if not isinstance(events, list):
            errors.append("Field 'events' must be a list")
            return errors

        # Collect valid tripwire IDs from camera_config
        valid_tripwire_ids = set()
        camera_config = data.get("camera_config", {})
        if isinstance(camera_config, dict):
            tripwires = camera_config.get("tripwires", [])
            if isinstance(tripwires, list):
                for tw in tripwires:
                    if isinstance(tw, dict) and "id" in tw:
                        valid_tripwire_ids.add(tw["id"])

        # Validate each event
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                errors.append(f"events[{i}]: event must be a dict")
                continue

            event_type = event.get("type")
            if event_type is None:
                errors.append(f"events[{i}]: missing required field 'type'")
                continue

            if event_type not in VALID_EVENT_TYPES:
                errors.append(
                    f"events[{i}]: invalid event type '{event_type}', "
                    f"must be one of {sorted(VALID_EVENT_TYPES)}"
                )
                continue

            # Anomaly sub_type validation
            if event_type == "anomaly":
                sub_type = event.get("sub_type")
                if sub_type is None:
                    errors.append(
                        f"events[{i}]: anomaly event missing required field 'sub_type'"
                    )
                elif sub_type not in VALID_ANOMALY_SUB_TYPES:
                    errors.append(
                        f"events[{i}]: invalid anomaly sub_type '{sub_type}', "
                        f"must be one of {sorted(VALID_ANOMALY_SUB_TYPES)}"
                    )

            # Tripwire tripwire_id validation
            if event_type == "tripwire":
                tripwire_id = event.get("tripwire_id")
                if tripwire_id is None:
                    errors.append(
                        f"events[{i}]: tripwire event missing required field 'tripwire_id'"
                    )
                elif valid_tripwire_ids and tripwire_id not in valid_tripwire_ids:
                    errors.append(
                        f"events[{i}]: tripwire_id '{tripwire_id}' not found in "
                        f"camera_config.tripwires (valid: {sorted(valid_tripwire_ids)})"
                    )

            # time_range validation
            time_range = event.get("time_range")
            if time_range is not None:
                if (isinstance(time_range, (list, tuple))
                        and len(time_range) == 2):
                    try:
                        start, end = float(time_range[0]), float(time_range[1])
                        if start >= end:
                            errors.append(
                                f"events[{i}]: time_range start ({start}) "
                                f"must be less than end ({end})"
                            )
                    except (TypeError, ValueError):
                        errors.append(
                            f"events[{i}]: time_range values must be numeric"
                        )
                else:
                    errors.append(
                        f"events[{i}]: time_range must be a list of [start, end]"
                    )

        return errors
