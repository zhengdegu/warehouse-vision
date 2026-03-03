"""Detection evaluator for image-level object detection accuracy (mAP).

Loads ground-truth YOLO labels, runs YOLODetector.detect() on each image,
performs greedy IoU matching (by confidence descending), and computes
per-class and global metrics including confusion matrix.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from src.evaluation.annotation import AnnotationLoader
from src.evaluation.metrics import (
    compute_ap,
    compute_confusion_matrix,
    compute_iou,
    compute_map,
    compute_prf,
)
from src.evaluation.models import (
    ClassMetrics,
    DetectionResult,
    GroundTruthBox,
    ImageEvalEntry,
)
from src.vision.detector import YOLODetector

logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """Evaluates YOLODetector detection accuracy against ground-truth labels."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        iou_threshold: float = 0.5,
        classes: Optional[List[int]] = None,
    ):
        """Initialize YOLODetector in detect mode.

        Args:
            model_path: Path to the YOLO model weights.
            confidence: Confidence threshold for detection.
            iou_threshold: IoU threshold for matching (default 0.5).
            classes: Allowed class IDs whitelist (None = all classes).
        """
        self.detector = YOLODetector(
            model_path=model_path,
            confidence=confidence,
            allowed_classes=classes,
        )
        self.iou_threshold = iou_threshold
        self.confidence = confidence
        # Store class names from model for confusion matrix / reporting
        self.class_names: Dict[int, str] = dict(self.detector.model.names)

    def run(
        self,
        image_dir: Path,
        label_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> DetectionResult:
        """Execute the full image detection evaluation pipeline.

        1. Load labels via AnnotationLoader
        2. For each image with labels: run detect(), greedy IoU matching
        3. Compute per-class AP, Precision, Recall
        4. Compute confusion matrix
        5. Return DetectionResult

        Args:
            image_dir: Directory containing evaluation images.
            label_dir: Directory containing YOLO format label files.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            DetectionResult with all metrics.

        Raises:
            ValueError: If label_dir is empty or has no valid label files.
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        # Load ground-truth labels
        labels, skipped_names = AnnotationLoader.load_yolo_labels(label_dir, image_dir)

        if not labels:
            raise ValueError(
                f"No valid label files found in '{label_dir}'. "
                "Cannot run detection evaluation without ground-truth annotations."
            )

        # Collect all class IDs present in GT
        all_gt_class_ids = set()
        for boxes in labels.values():
            for box in boxes:
                all_gt_class_ids.add(box.class_id)

        # Per-class accumulators: class_id -> list of (confidence, is_tp)
        # for AP computation across all images
        per_class_dets: Dict[int, List[tuple]] = {}
        per_class_gt_count: Dict[int, int] = {}

        # Confusion matrix matches
        cm_matches: List[dict] = []

        # Per-image results
        per_image: List[ImageEvalEntry] = []

        # Global TP/FP/FN counters
        total_tp = 0
        total_fp = 0
        total_fn = 0

        image_stems = sorted(labels.keys())
        total_images = len(image_stems)

        for idx, stem in enumerate(image_stems):
            gt_boxes = labels[stem]

            # Find the actual image file
            img_path = self._find_image(image_dir, stem)
            if img_path is None:
                logger.warning("Image file not found for stem '%s', skipping", stem)
                continue

            # Load image and run detection
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Failed to read image '%s', skipping", img_path)
                continue

            detections = self.detector.detect(frame)

            # Greedy IoU matching at self.iou_threshold
            img_tp, img_fp, img_fn, img_matches = self._greedy_match(
                detections, gt_boxes, self.iou_threshold
            )

            total_tp += img_tp
            total_fp += img_fp
            total_fn += img_fn

            # Record per-image entry
            per_image.append(ImageEvalEntry(
                filename=img_path.name,
                gt_count=len(gt_boxes),
                det_count=len(detections),
                tp_count=img_tp,
            ))

            # Accumulate per-class detection records for AP
            self._accumulate_per_class(
                detections, gt_boxes, img_matches,
                per_class_dets, per_class_gt_count,
            )

            # Accumulate confusion matrix entries
            cm_matches.extend(img_matches)

            if progress_callback:
                progress_callback(idx + 1, total_images)

        # Compute per-class metrics
        class_metrics = self._compute_class_metrics(
            per_class_dets, per_class_gt_count, all_gt_class_ids
        )

        # Compute mAP50
        ap_per_class_50: Dict[int, Optional[float]] = {}
        for cm in class_metrics:
            ap_per_class_50[cm.class_id] = cm.ap if cm.has_samples else None
        map50 = compute_map(ap_per_class_50)

        # Compute mAP50_95
        map50_95 = self._compute_map50_95(
            image_dir, labels, image_stems
        )

        # Compute global precision/recall
        total_p, total_r, _ = compute_prf(total_tp, total_fp, total_fn)

        # Build class name list for confusion matrix
        # Use sorted class IDs that appear in GT or detections
        all_class_ids = sorted(all_gt_class_ids)
        cm_class_names = [self.class_names.get(cid, str(cid)) for cid in all_class_ids]

        # Remap confusion matrix matches to use index within all_class_ids
        class_id_to_idx = {cid: i for i, cid in enumerate(all_class_ids)}
        remapped_matches = []
        for m in cm_matches:
            gt_cls = m.get("gt_class")
            pred_cls = m.get("pred_class")
            remapped_matches.append({
                "gt_class": class_id_to_idx.get(gt_cls) if gt_cls is not None else None,
                "pred_class": class_id_to_idx.get(pred_cls) if pred_cls is not None else None,
            })

        confusion_matrix = compute_confusion_matrix(remapped_matches, cm_class_names)

        return DetectionResult(
            map50=map50,
            map50_95=map50_95,
            total_precision=total_p,
            total_recall=total_r,
            class_metrics=class_metrics,
            confusion_matrix=confusion_matrix,
            per_image=per_image,
            skipped_images=len(skipped_names),
        )

    @staticmethod
    def _find_image(image_dir: Path, stem: str) -> Optional[Path]:
        """Find an image file by stem in the image directory."""
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = image_dir / (stem + ext)
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _greedy_match(
        detections, gt_boxes: List[GroundTruthBox], iou_threshold: float
    ) -> tuple:
        """Greedy IoU matching: sort detections by confidence desc, match to GT.

        Each GT box can be matched at most once.

        Returns:
            (tp, fp, fn, matches) where matches is a list of dicts with
            gt_class and pred_class keys for confusion matrix.
        """
        # Sort detections by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        matched_gt = set()  # indices of matched GT boxes
        tp = 0
        fp = 0
        matches = []

        for det in sorted_dets:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(
                    det.bbox,
                    [gt.x1, gt.y1, gt.x2, gt.y2],
                )
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                # True positive
                matched_gt.add(best_gt_idx)
                tp += 1
                matches.append({
                    "gt_class": gt_boxes[best_gt_idx].class_id,
                    "pred_class": det.class_id,
                })
            else:
                # False positive
                fp += 1
                matches.append({
                    "gt_class": None,
                    "pred_class": det.class_id,
                })

        # False negatives: unmatched GT boxes
        fn = len(gt_boxes) - len(matched_gt)
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                matches.append({
                    "gt_class": gt.class_id,
                    "pred_class": None,
                })

        return tp, fp, fn, matches

    @staticmethod
    def _accumulate_per_class(
        detections, gt_boxes, matches,
        per_class_dets: Dict[int, List[tuple]],
        per_class_gt_count: Dict[int, int],
    ):
        """Accumulate per-class detection records for AP computation.

        For each detection, record (confidence, is_tp) keyed by class_id.
        For each GT box, increment the GT count for that class.
        """
        # Count GT per class
        for gt in gt_boxes:
            per_class_gt_count[gt.class_id] = (
                per_class_gt_count.get(gt.class_id, 0) + 1
            )

        # Sort detections by confidence desc for this image
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        matched_gt = set()

        for det in sorted_dets:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if gt.class_id != det.class_id:
                    continue
                iou = compute_iou(
                    det.bbox,
                    [gt.x1, gt.y1, gt.x2, gt.y2],
                )
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            is_tp = best_gt_idx >= 0
            if is_tp:
                matched_gt.add(best_gt_idx)

            if det.class_id not in per_class_dets:
                per_class_dets[det.class_id] = []
            per_class_dets[det.class_id].append((det.confidence, is_tp))

    def _compute_class_metrics(
        self,
        per_class_dets: Dict[int, List[tuple]],
        per_class_gt_count: Dict[int, int],
        all_gt_class_ids: set,
    ) -> List[ClassMetrics]:
        """Compute per-class AP, Precision, Recall metrics."""
        # All class IDs: union of GT classes and detected classes
        all_class_ids = sorted(all_gt_class_ids | set(per_class_dets.keys()))

        class_metrics = []
        for class_id in all_class_ids:
            class_name = self.class_names.get(class_id, str(class_id))
            gt_count = per_class_gt_count.get(class_id, 0)
            has_samples = gt_count > 0

            if not has_samples:
                # No GT samples for this class
                det_records = per_class_dets.get(class_id, [])
                fp_count = len(det_records)
                class_metrics.append(ClassMetrics(
                    class_id=class_id,
                    class_name=class_name,
                    ap=None,
                    precision=0.0,
                    recall=0.0,
                    tp=0,
                    fp=fp_count,
                    fn=0,
                    gt_count=0,
                    has_samples=False,
                ))
                continue

            # Sort all detections for this class by confidence desc
            det_records = per_class_dets.get(class_id, [])
            det_records.sort(key=lambda x: x[0], reverse=True)

            # Compute cumulative precision/recall curve
            cum_tp = 0
            cum_fp = 0
            precisions = []
            recalls = []

            for conf, is_tp in det_records:
                if is_tp:
                    cum_tp += 1
                else:
                    cum_fp += 1
                precisions.append(cum_tp / (cum_tp + cum_fp))
                recalls.append(cum_tp / gt_count)

            ap = compute_ap(precisions, recalls)

            # Final TP/FP/FN
            total_tp_cls = sum(1 for _, is_tp in det_records if is_tp)
            total_fp_cls = len(det_records) - total_tp_cls
            total_fn_cls = gt_count - total_tp_cls

            p, r, _ = compute_prf(total_tp_cls, total_fp_cls, total_fn_cls)

            class_metrics.append(ClassMetrics(
                class_id=class_id,
                class_name=class_name,
                ap=ap,
                precision=p,
                recall=r,
                tp=total_tp_cls,
                fp=total_fp_cls,
                fn=total_fn_cls,
                gt_count=gt_count,
                has_samples=True,
            ))

        return class_metrics

    def _compute_map50_95(
        self,
        image_dir: Path,
        labels: Dict[str, List[GroundTruthBox]],
        image_stems: List[str],
    ) -> float:
        """Compute mAP at IoU thresholds [0.5, 0.55, ..., 0.95] and average.

        Re-uses already-loaded labels and re-runs matching at each threshold.
        Does NOT re-run detection (reuses cached detections would be ideal,
        but for simplicity we re-run detection per threshold).

        For efficiency, we cache detections from the first pass and only
        re-do the matching at different IoU thresholds.
        """
        # First, collect all detections per image (run detect once per image)
        image_detections: Dict[str, list] = {}
        for stem in image_stems:
            img_path = self._find_image(image_dir, stem)
            if img_path is None:
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            image_detections[stem] = self.detector.detect(frame)

        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95
        ap_at_thresholds = []

        for iou_thresh in iou_thresholds:
            per_class_dets: Dict[int, List[tuple]] = {}
            per_class_gt_count: Dict[int, int] = {}

            for stem in image_stems:
                gt_boxes = labels[stem]
                dets = image_detections.get(stem, [])

                # Count GT per class
                for gt in gt_boxes:
                    per_class_gt_count[gt.class_id] = (
                        per_class_gt_count.get(gt.class_id, 0) + 1
                    )

                # Per-class matching at this IoU threshold
                sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
                matched_gt = set()

                for det in sorted_dets:
                    best_iou = 0.0
                    best_gt_idx = -1

                    for gt_idx, gt in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        if gt.class_id != det.class_id:
                            continue
                        iou = compute_iou(
                            det.bbox,
                            [gt.x1, gt.y1, gt.x2, gt.y2],
                        )
                        if iou >= iou_thresh and iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    is_tp = best_gt_idx >= 0
                    if is_tp:
                        matched_gt.add(best_gt_idx)

                    if det.class_id not in per_class_dets:
                        per_class_dets[det.class_id] = []
                    per_class_dets[det.class_id].append((det.confidence, is_tp))

            # Compute AP per class at this threshold
            ap_per_class: Dict[int, Optional[float]] = {}
            for class_id, gt_count in per_class_gt_count.items():
                if gt_count == 0:
                    ap_per_class[class_id] = None
                    continue

                det_records = per_class_dets.get(class_id, [])
                det_records.sort(key=lambda x: x[0], reverse=True)

                cum_tp = 0
                cum_fp = 0
                precisions = []
                recalls = []
                for conf, is_tp in det_records:
                    if is_tp:
                        cum_tp += 1
                    else:
                        cum_fp += 1
                    precisions.append(cum_tp / (cum_tp + cum_fp))
                    recalls.append(cum_tp / gt_count)

                ap_per_class[class_id] = compute_ap(precisions, recalls)

            ap_at_thresholds.append(compute_map(ap_per_class))

        if not ap_at_thresholds:
            return 0.0
        return sum(ap_at_thresholds) / len(ap_at_thresholds)
