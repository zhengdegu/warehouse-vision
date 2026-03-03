"""Video rule engine accuracy evaluator.

Evaluates intrusion, tripwire, counting, presence, and anomaly detection
rules against ground-truth video annotations.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2

from src.evaluation.annotation import AnnotationLoader, ValidationError
from src.evaluation.metrics import compute_counting_error, compute_prf
from src.evaluation.models import (
    CountingDetail,
    MatchResult,
    RuleMetrics,
    RuleResult,
    VideoAnnotation,
    VideoEvalResult,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv"}


class RuleEvaluator:
    """Video rule engine evaluator.

    Processes videos frame-by-frame using detection + tracking + rule engines,
    then matches predicted events against ground-truth annotations.
    """

    def __init__(
        self,
        model_path: str,
        pose_model_path: Optional[str] = None,
        confidence: float = 0.3,
        time_tolerance: float = 2.0,
    ):
        """Initialize detectors.

        Args:
            model_path: Path to YOLO detection model.
            pose_model_path: Path to YOLO pose model (optional).
            confidence: Detection confidence threshold.
            time_tolerance: Time tolerance in seconds for event matching.
        """
        self.model_path = model_path
        self.pose_model_path = pose_model_path
        self.confidence = confidence
        self.time_tolerance = time_tolerance

        # Lazy-loaded detectors (avoid loading models at import time)
        self._detector = None
        self._pose_detector = None

    def _get_detector(self):
        """Lazy-load YOLO detector."""
        if self._detector is None:
            from src.vision.detector import YOLODetector
            self._detector = YOLODetector(
                model_path=self.model_path,
                confidence=self.confidence,
                allowed_classes=[0, 1, 2, 3, 5, 7],
            )
        return self._detector

    def _get_pose_detector(self):
        """Lazy-load pose detector."""
        if self._pose_detector is None and self.pose_model_path:
            from src.vision.detector import PoseDetector
            self._pose_detector = PoseDetector(
                model_path=self.pose_model_path,
                confidence=self.confidence,
            )
        return self._pose_detector

    def run(
        self,
        videos_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> RuleResult:
        """Run evaluation on all videos in directory.

        Scans for video files, matches with same-name JSON annotations,
        evaluates each video, and aggregates metrics.

        Args:
            videos_dir: Directory containing video files and JSON annotations.
            progress_callback: Optional callback(current, total, video_name).

        Returns:
            RuleResult with per-video and aggregated metrics.
        """
        videos_dir = Path(videos_dir)

        # Find video files
        video_files = sorted(
            p for p in videos_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

        per_video: List[VideoEvalResult] = []

        for idx, video_path in enumerate(video_files):
            if progress_callback:
                progress_callback(idx + 1, len(video_files), video_path.name)

            # Match JSON annotation
            json_path = video_path.with_suffix(".json")
            if not json_path.is_file():
                logger.warning("No annotation for %s, skipping", video_path.name)
                continue

            # Load annotation
            try:
                annotation = AnnotationLoader.load_video_annotation(json_path)
            except (ValidationError, Exception) as e:
                logger.error("Failed to load annotation %s: %s", json_path, e)
                per_video.append(VideoEvalResult(
                    video_name=video_path.name,
                    rule_metrics=[],
                    errors=[f"Annotation error: {e}"],
                ))
                continue

            # Evaluate video
            result = self._evaluate_video(video_path, annotation)
            per_video.append(result)

        # Aggregate metrics across videos
        aggregated = self._aggregate_metrics(per_video)

        return RuleResult(per_video=per_video, aggregated=aggregated)

    def _evaluate_video(
        self, video_path: Path, annotation: VideoAnnotation
    ) -> VideoEvalResult:
        """Evaluate a single video against its annotation.

        Opens video, processes frames at fps_sample rate, runs detectors
        and rule engines, collects events, then matches against ground truth.

        Args:
            video_path: Path to video file.
            annotation: Parsed video annotation.

        Returns:
            VideoEvalResult with metrics for each rule type.
        """
        errors: List[str] = []
        rule_metrics: List[RuleMetrics] = []

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return VideoEvalResult(
                video_name=video_path.name,
                rule_metrics=[],
                errors=[f"Cannot open video: {video_path}"],
            )

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fps_sample = annotation.fps_sample or 5
            # Frame interval: process every N-th frame
            frame_interval = max(1, int(round(video_fps / fps_sample)))

            cam_cfg = annotation.camera_config
            camera_id = "eval"

            # Initialize rule engines from camera_config
            from src.rules.intrusion import IntrusionDetector
            from src.rules.tripwire import TripwireDetector
            from src.rules.counting import FlowCounter
            from src.rules.presence import PresenceDetector
            from src.rules.anomaly import AnomalyEngine

            roi = cam_cfg.get("roi", [])
            intrusion_cfg = cam_cfg.get("intrusion", {})
            intrusion_det = IntrusionDetector(
                roi=roi,
                confirm_frames=intrusion_cfg.get("confirm_frames", 5),
                cooldown=intrusion_cfg.get("cooldown", 30.0),
            ) if roi else None

            # Create tripwire detectors
            tripwire_cfgs = cam_cfg.get("tripwires", [])
            tripwire_dets: Dict[str, TripwireDetector] = {}
            for tw_cfg in tripwire_cfgs:
                tw_id = tw_cfg["id"]
                tripwire_dets[tw_id] = TripwireDetector(
                    tripwire_id=tw_id,
                    name=tw_cfg.get("name", tw_id),
                    p1=tuple(tw_cfg["p1"]),
                    p2=tuple(tw_cfg["p2"]),
                    direction=tw_cfg.get("direction", "left_to_right"),
                    cooldown=tw_cfg.get("cooldown", 2.0),
                )

            flow_counter = FlowCounter(camera_id=camera_id) if tripwire_dets else None

            # Presence detector
            presence_det = PresenceDetector(
                roi=roi if roi else None,
                cooldown=1.0,  # Low cooldown for evaluation
            )

            # Anomaly engine
            anomaly_cfg = cam_cfg.get("anomaly", {})
            anomaly_engine = AnomalyEngine(config=anomaly_cfg, roi=roi if roi else None)

            # Check if pose is needed
            needs_pose = anomaly_cfg.get("fight", {}).get("enabled", False) or \
                         anomaly_cfg.get("fall", {}).get("enabled", False)

            detector = self._get_detector()
            pose_detector = self._get_pose_detector() if needs_pose else None

            # Collect predicted events
            all_events: List[dict] = []

            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_interval != 0:
                    frame_num += 1
                    continue

                timestamp = frame_num / video_fps

                # Run detection + tracking
                detections = detector.track(frame)

                # Run pose detection if needed
                if pose_detector:
                    pose_dets = pose_detector.track(frame)
                    # Merge pose keypoints into detections by track_id
                    pose_by_track = {d.track_id: d for d in pose_dets if d.track_id >= 0}
                    for det in detections:
                        if det.track_id in pose_by_track and det.class_name == "person":
                            det.keypoints = pose_by_track[det.track_id].keypoints

                # Override timestamps to use video timeline
                import time as _time
                _original_time = _time.time
                _time.time = lambda: timestamp

                try:
                    # Run rule engines
                    if intrusion_det:
                        events = intrusion_det.update(detections, camera_id)
                        for e in events:
                            e["timestamp"] = timestamp
                        all_events.extend(events)

                    for tw_id, tw_det in tripwire_dets.items():
                        events = tw_det.update(detections, camera_id)
                        for e in events:
                            e["timestamp"] = timestamp
                        if flow_counter:
                            flow_counter.update(events)
                        all_events.extend(events)

                    presence_events = presence_det.update(detections, camera_id)
                    for e in presence_events:
                        e["timestamp"] = timestamp
                    all_events.extend(presence_events)

                    anomaly_events = anomaly_engine.update(detections, camera_id)
                    for e in anomaly_events:
                        e["timestamp"] = timestamp
                    all_events.extend(anomaly_events)
                finally:
                    _time.time = _original_time

                frame_num += 1

            # Evaluate each rule type
            gt_events = annotation.events

            # Intrusion evaluation
            rule_metrics.extend(
                self._evaluate_intrusion(all_events, gt_events)
            )

            # Tripwire evaluation
            rule_metrics.extend(
                self._evaluate_tripwire(all_events, gt_events, tripwire_dets)
            )

            # Counting evaluation
            rule_metrics.extend(
                self._evaluate_counting(flow_counter, gt_events)
            )

            # Presence evaluation
            rule_metrics.extend(
                self._evaluate_presence(all_events, gt_events)
            )

            # Anomaly evaluation
            rule_metrics.extend(
                self._evaluate_anomaly(all_events, gt_events)
            )

        except Exception as e:
            logger.error("Error processing video %s: %s", video_path.name, e)
            errors.append(f"Processing error: {e}")
        finally:
            cap.release()

        return VideoEvalResult(
            video_name=video_path.name,
            rule_metrics=rule_metrics,
            errors=errors,
        )

    def _match_events(
        self,
        predicted: List[dict],
        ground_truth: List[dict],
        tolerance: float,
    ) -> MatchResult:
        """Greedy event matching by time proximity, one-to-one constraint.

        For instant events: |pred_time - gt_time| <= tolerance
        For duration events: pred_time falls within [gt_start - tolerance, gt_end + tolerance]

        Args:
            predicted: List of predicted event dicts (must have 'timestamp').
            ground_truth: List of ground truth event dicts.
            tolerance: Time tolerance in seconds.

        Returns:
            MatchResult with tp, fp, fn counts and matched/unmatched lists.
        """
        if not predicted and not ground_truth:
            return MatchResult(
                tp=0, fp=0, fn=0,
                matched_pairs=[], false_positives=[], false_negatives=[],
            )

        if not predicted:
            return MatchResult(
                tp=0, fp=0, fn=len(ground_truth),
                matched_pairs=[],
                false_positives=[],
                false_negatives=list(ground_truth),
            )

        if not ground_truth:
            return MatchResult(
                tp=0, fp=len(predicted), fn=0,
                matched_pairs=[],
                false_positives=list(predicted),
                false_negatives=[],
            )

        # Build candidate pairs with time distance
        candidates: List[Tuple[float, int, int]] = []  # (distance, pred_idx, gt_idx)

        for pi, pred in enumerate(predicted):
            pred_time = pred.get("timestamp", 0.0)
            for gi, gt in enumerate(ground_truth):
                time_range = gt.get("time_range")
                gt_time = gt.get("time_sec")

                if time_range is not None:
                    # Duration event: check if pred falls within expanded range
                    start, end = float(time_range[0]), float(time_range[1])
                    if start - tolerance <= pred_time <= end + tolerance:
                        # Distance = how far from the range center
                        center = (start + end) / 2.0
                        dist = abs(pred_time - center)
                        candidates.append((dist, pi, gi))
                elif gt_time is not None:
                    # Instant event: check time difference
                    dist = abs(pred_time - float(gt_time))
                    if dist <= tolerance:
                        candidates.append((dist, pi, gi))

        # Sort by distance (greedy: closest first)
        candidates.sort(key=lambda x: x[0])

        matched_pred = set()
        matched_gt = set()
        matched_pairs: List[Tuple[dict, dict]] = []

        for dist, pi, gi in candidates:
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
            matched_pairs.append((predicted[pi], ground_truth[gi]))

        tp = len(matched_pairs)
        false_positives = [predicted[i] for i in range(len(predicted)) if i not in matched_pred]
        false_negatives = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt]

        return MatchResult(
            tp=tp,
            fp=len(false_positives),
            fn=len(false_negatives),
            matched_pairs=matched_pairs,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

    # ── Rule-type specific evaluations ──

    def _evaluate_intrusion(
        self, all_events: List[dict], gt_events: List[dict]
    ) -> List[RuleMetrics]:
        """Evaluate intrusion detection: P/R/F1 + false positive/negative lists."""
        gt_intrusion = [e for e in gt_events if e.get("type") == "intrusion"]
        if not gt_intrusion:
            return [RuleMetrics(
                rule_type="intrusion", sub_type="", precision=0.0, recall=0.0,
                f1=0.0, tp=0, fp=0, fn=0, has_samples=False, details={},
            )]

        pred_intrusion = [e for e in all_events if e.get("type") == "intrusion"]
        match = self._match_events(pred_intrusion, gt_intrusion, self.time_tolerance)
        p, r, f1 = compute_prf(match.tp, match.fp, match.fn)

        return [RuleMetrics(
            rule_type="intrusion", sub_type="", precision=p, recall=r, f1=f1,
            tp=match.tp, fp=match.fp, fn=match.fn, has_samples=True,
            details={
                "false_positives": [
                    {"time": e.get("timestamp"), "class_name": e.get("class_name")}
                    for e in match.false_positives
                ],
                "false_negatives": [
                    {"time_range": e.get("time_range"), "class_name": e.get("class_name")}
                    for e in match.false_negatives
                ],
            },
        )]

    def _evaluate_tripwire(
        self,
        all_events: List[dict],
        gt_events: List[dict],
        tripwire_dets: Dict[str, Any],
    ) -> List[RuleMetrics]:
        """Evaluate tripwire detection: P/R/F1 + direction accuracy, grouped by tripwire."""
        gt_tripwire = [e for e in gt_events if e.get("type") == "tripwire"]
        if not gt_tripwire:
            return [RuleMetrics(
                rule_type="tripwire", sub_type="", precision=0.0, recall=0.0,
                f1=0.0, tp=0, fp=0, fn=0, has_samples=False, details={},
            )]

        pred_tripwire = [e for e in all_events if e.get("type") == "tripwire"]

        # Group by tripwire_id
        tw_ids = set()
        for e in gt_tripwire:
            tw_ids.add(e.get("tripwire_id", "unknown"))
        for e in pred_tripwire:
            tw_ids.add(e.get("tripwire_id", "unknown"))

        metrics_list: List[RuleMetrics] = []
        total_tp, total_fp, total_fn = 0, 0, 0
        direction_correct = 0
        direction_total = 0

        for tw_id in sorted(tw_ids):
            gt_tw = [e for e in gt_tripwire if e.get("tripwire_id") == tw_id]
            pred_tw = [e for e in pred_tripwire if e.get("tripwire_id") == tw_id]

            match = self._match_events(pred_tw, gt_tw, self.time_tolerance)
            p, r, f1 = compute_prf(match.tp, match.fp, match.fn)

            # Check direction accuracy for matched pairs
            tw_dir_correct = 0
            for pred_evt, gt_evt in match.matched_pairs:
                pred_dir = pred_evt.get("crossing_direction") or pred_evt.get("sub_type")
                gt_dir = gt_evt.get("direction")
                if pred_dir and gt_dir and pred_dir == gt_dir:
                    tw_dir_correct += 1
                direction_total += 1

            direction_correct += tw_dir_correct
            total_tp += match.tp
            total_fp += match.fp
            total_fn += match.fn

            metrics_list.append(RuleMetrics(
                rule_type="tripwire", sub_type=tw_id, precision=p, recall=r,
                f1=f1, tp=match.tp, fp=match.fp, fn=match.fn, has_samples=True,
                details={
                    "direction_correct": tw_dir_correct,
                    "direction_total": match.tp,
                },
            ))

        # Overall tripwire metrics
        p, r, f1 = compute_prf(total_tp, total_fp, total_fn)
        dir_accuracy = direction_correct / direction_total if direction_total > 0 else 0.0

        metrics_list.insert(0, RuleMetrics(
            rule_type="tripwire", sub_type="overall", precision=p, recall=r,
            f1=f1, tp=total_tp, fp=total_fp, fn=total_fn, has_samples=True,
            details={
                "direction_accuracy": dir_accuracy,
                "direction_correct": direction_correct,
                "direction_total": direction_total,
            },
        ))

        return metrics_list

    def _evaluate_counting(
        self,
        flow_counter: Optional[Any],
        gt_events: List[dict],
    ) -> List[RuleMetrics]:
        """Evaluate counting: absolute/relative error by class."""
        gt_counting = [e for e in gt_events if e.get("type") == "counting"]
        if not gt_counting:
            return [RuleMetrics(
                rule_type="counting", sub_type="", precision=0.0, recall=0.0,
                f1=0.0, tp=0, fp=0, fn=0, has_samples=False, details={},
            )]

        # Use the first counting annotation (there should be one per video)
        gt = gt_counting[0]
        expected_in = gt.get("expected_in", 0)
        expected_out = gt.get("expected_out", 0)
        by_class = gt.get("by_class", {})

        actual_in = 0
        actual_out = 0
        if flow_counter:
            counts = flow_counter.get_current_counts()
            actual_in = counts.get("total_in", 0)
            actual_out = counts.get("total_out", 0)

        abs_err_in, rel_err_in = compute_counting_error(expected_in, actual_in)
        abs_err_out, rel_err_out = compute_counting_error(expected_out, actual_out)

        # Per-class counting details
        counting_details: List[dict] = []
        for cls_name, cls_expected in by_class.items():
            exp_in = cls_expected.get("in", 0)
            exp_out = cls_expected.get("out", 0)
            # We don't have per-class actual counts from FlowCounter easily,
            # so we report expected vs 0 (actual per-class would need FlowCounter extension)
            act_in = 0
            act_out = 0
            ae_in, re_in = compute_counting_error(exp_in, act_in)
            ae_out, re_out = compute_counting_error(exp_out, act_out)
            counting_details.append(CountingDetail(
                class_name=cls_name,
                expected_in=exp_in, expected_out=exp_out,
                actual_in=act_in, actual_out=act_out,
                abs_error_in=ae_in, abs_error_out=ae_out,
                rel_error_in=re_in, rel_error_out=re_out,
            ).__dict__)

        return [RuleMetrics(
            rule_type="counting", sub_type="", precision=0.0, recall=0.0,
            f1=0.0, tp=0, fp=0, fn=0, has_samples=True,
            details={
                "expected_in": expected_in,
                "expected_out": expected_out,
                "actual_in": actual_in,
                "actual_out": actual_out,
                "abs_error_in": abs_err_in,
                "abs_error_out": abs_err_out,
                "rel_error_in": rel_err_in,
                "rel_error_out": rel_err_out,
                "by_class": counting_details,
            },
        )]

    def _evaluate_presence(
        self, all_events: List[dict], gt_events: List[dict]
    ) -> List[RuleMetrics]:
        """Evaluate presence detection: trigger accuracy."""
        gt_presence = [e for e in gt_events if e.get("type") == "presence"]
        if not gt_presence:
            return [RuleMetrics(
                rule_type="presence", sub_type="", precision=0.0, recall=0.0,
                f1=0.0, tp=0, fp=0, fn=0, has_samples=False, details={},
            )]

        pred_presence = [e for e in all_events if e.get("type") == "presence"]
        match = self._match_events(pred_presence, gt_presence, self.time_tolerance)
        p, r, f1 = compute_prf(match.tp, match.fp, match.fn)

        return [RuleMetrics(
            rule_type="presence", sub_type="", precision=p, recall=r, f1=f1,
            tp=match.tp, fp=match.fp, fn=match.fn, has_samples=True,
            details={
                "trigger_accuracy": r,  # recall = correctly triggered / expected
            },
        )]

    def _evaluate_anomaly(
        self, all_events: List[dict], gt_events: List[dict]
    ) -> List[RuleMetrics]:
        """Evaluate anomaly detection: P/R/F1 per sub_type."""
        gt_anomaly = [e for e in gt_events if e.get("type") == "anomaly"]
        if not gt_anomaly:
            return [RuleMetrics(
                rule_type="anomaly", sub_type="", precision=0.0, recall=0.0,
                f1=0.0, tp=0, fp=0, fn=0, has_samples=False, details={},
            )]

        pred_anomaly = [e for e in all_events if e.get("type") == "anomaly"]

        # Group by sub_type
        sub_types = set()
        for e in gt_anomaly:
            sub_types.add(e.get("sub_type", "unknown"))

        metrics_list: List[RuleMetrics] = []

        for sub_type in sorted(sub_types):
            gt_sub = [e for e in gt_anomaly if e.get("sub_type") == sub_type]
            pred_sub = [e for e in pred_anomaly if e.get("sub_type") == sub_type]

            match = self._match_events(pred_sub, gt_sub, self.time_tolerance)
            p, r, f1 = compute_prf(match.tp, match.fp, match.fn)

            metrics_list.append(RuleMetrics(
                rule_type="anomaly", sub_type=sub_type, precision=p, recall=r,
                f1=f1, tp=match.tp, fp=match.fp, fn=match.fn, has_samples=True,
                details={
                    "false_positives": [
                        {"time": e.get("timestamp"), "sub_type": e.get("sub_type")}
                        for e in match.false_positives
                    ],
                    "false_negatives": [
                        {"time_range": e.get("time_range"),
                         "time_sec": e.get("time_sec"),
                         "sub_type": e.get("sub_type")}
                        for e in match.false_negatives
                    ],
                },
            ))

        return metrics_list

    def _aggregate_metrics(
        self, per_video: List[VideoEvalResult]
    ) -> List[RuleMetrics]:
        """Aggregate metrics across all videos.

        Groups by (rule_type, sub_type) and sums tp/fp/fn, then recomputes P/R/F1.
        """
        agg: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for video_result in per_video:
            for m in video_result.rule_metrics:
                key = (m.rule_type, m.sub_type)
                if key not in agg:
                    agg[key] = {
                        "tp": 0, "fp": 0, "fn": 0,
                        "has_samples": False, "details": {},
                    }
                agg[key]["tp"] += m.tp
                agg[key]["fp"] += m.fp
                agg[key]["fn"] += m.fn
                if m.has_samples:
                    agg[key]["has_samples"] = True

        result: List[RuleMetrics] = []
        for (rule_type, sub_type), data in sorted(agg.items()):
            if data["has_samples"]:
                p, r, f1 = compute_prf(data["tp"], data["fp"], data["fn"])
            else:
                p, r, f1 = 0.0, 0.0, 0.0

            result.append(RuleMetrics(
                rule_type=rule_type,
                sub_type=sub_type,
                precision=p,
                recall=r,
                f1=f1,
                tp=data["tp"],
                fp=data["fp"],
                fn=data["fn"],
                has_samples=data["has_samples"],
                details={},
            ))

        return result
