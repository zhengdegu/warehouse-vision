"""Auto-labeling module for generating YOLO format annotations using YOLODetector."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from src.evaluation.annotation import IMAGE_EXTENSIONS
from src.evaluation.models import LabelSummary
from src.vision.detector import YOLODetector

logger = logging.getLogger(__name__)


class AutoLabeler:
    """Batch pre-annotation generator using YOLODetector.

    Scans an image directory, runs detection on each image, and writes
    YOLO format label files with confidence/class metadata comments.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        classes: Optional[List[int]] = None,
    ):
        """Initialize with a YOLODetector instance.

        Args:
            model_path: Path to the YOLO model weights.
            confidence: Detection confidence threshold.
            classes: List of allowed class IDs (None = all classes).
        """
        self.detector = YOLODetector(
            model_path=model_path,
            confidence=confidence,
            allowed_classes=classes,
        )

    def label_directory(
        self,
        image_dir: Path,
        label_dir: Path,
        overwrite: bool = False,
    ) -> LabelSummary:
        """Scan image_dir for images and generate YOLO format labels.

        Args:
            image_dir: Directory containing source images.
            label_dir: Directory to write label .txt files.
            overwrite: If True, overwrite existing label files.

        Returns:
            LabelSummary with statistics about the labeling run.
        """
        label_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_images = len(image_files)
        skipped = 0
        labeled = 0
        class_counts: Dict[str, int] = {}

        for img_path in image_files:
            label_path = label_dir / (img_path.stem + ".txt")

            if label_path.exists() and not overwrite:
                skipped += 1
                continue

            counts = self._label_single(img_path, label_path)
            labeled += 1
            for cls_name, cnt in counts.items():
                class_counts[cls_name] = class_counts.get(cls_name, 0) + cnt

        return LabelSummary(
            total_images=total_images,
            skipped=skipped,
            labeled=labeled,
            class_counts=class_counts,
        )

    def _label_single(self, image_path: Path, label_path: Path) -> Dict[str, int]:
        """Generate a YOLO label file for a single image.

        Each detection line is preceded by a comment with confidence and class name.
        If no detections are found, writes a single '# no detections' comment.

        Args:
            image_path: Path to the source image.
            label_path: Path to write the label .txt file.

        Returns:
            Dict mapping class_name to count for this image.
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning("Failed to read image: %s", image_path)
            label_path.write_text("# no detections\n", encoding="utf-8")
            return {}

        img_h, img_w = frame.shape[:2]
        detections = self.detector.detect(frame)

        class_counts: Dict[str, int] = {}

        if not detections:
            label_path.write_text("# no detections\n", encoding="utf-8")
            return class_counts

        lines: List[str] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Convert pixel coords to YOLO normalized format
            w_px = x2 - x1
            h_px = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            cx_norm = cx / img_w
            cy_norm = cy / img_h
            w_norm = w_px / img_w
            h_norm = h_px / img_h

            # Comment line with metadata
            lines.append(f"# conf={det.confidence:.2f} {det.class_name}")
            # YOLO format line
            lines.append(
                f"{det.class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return class_counts
