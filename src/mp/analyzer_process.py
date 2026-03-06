"""
分析器进程 — 参考 Frigate CameraTracker + process_frames

独立进程: 从共享内存读帧 → 运动检测 → 送检测器 → 规则引擎 → 事件

架构对标 Frigate:
- CameraTracker (FrigateProcess) → AnalyzerProcess (mp.Process)
- process_frames() → _process_loop()
- RemoteObjectDetector → RemoteDetector (通过 Queue 通信)
- ImprovedMotionDetector → MotionDetector (已有)

进程间通信:
- frame_queue (Queue): capture → analyzer，帧就绪通知
- detect_queue (Queue): analyzer → detector，检测请求
- result_queue (Queue): detector → analyzer，检测结果
- event_queue (Queue): analyzer → 主进程，事件通知
- bgr_shm: analyzer 写入 BGR 帧 → Web 层读取（MJPEG）
"""

import logging
import multiprocessing as mp
import os
import time
import numpy as np
from multiprocessing import shared_memory
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)



class AnalyzerProcess(mp.Process):
    """
    单路摄像头分析进程 — 对标 Frigate CameraTracker。

    Pipeline: 共享内存读帧 → 运动检测(Y通道) → YUV→BGR → 送检测器 → 规则引擎 → 事件

    Args:
        cam_config: 摄像头配置 dict
        model_config: 模型配置 dict
        events_config: 事件配置 dict
        frame_queue: capture → analyzer 帧通知
        detect_queue: analyzer → detector 检测请求
        result_queue: detector → analyzer 检测结果
        event_queue: analyzer → 主进程 事件通知
        perf_dict: 共享性能指标 dict (Manager().dict())
        frame_shm_info: BGR 帧共享内存信息 (name, shape)
        stop_event: 停止信号
        db_path: 数据库路径
    """

    def __init__(self, cam_config: dict, model_config: dict,
                 events_config: dict,
                 frame_queue: mp.Queue,
                 detect_queue: mp.Queue,
                 result_queue: mp.Queue,
                 event_queue: mp.Queue,
                 perf_dict: dict,
                 bgr_shm_name: str,
                 bgr_shm_shape: tuple,
                 stop_event: mp.Event,
                 db_path: str = "data/warehouse_vision.db"):
        super().__init__(
            daemon=True,
            name=f"analyzer:{cam_config['id']}")
        self.cam_config = cam_config
        self.model_config = model_config
        self.events_config = events_config
        self.frame_queue = frame_queue
        self.detect_queue = detect_queue
        self.result_queue = result_queue
        self.event_queue = event_queue
        self.perf_dict = perf_dict
        self.bgr_shm_name = bgr_shm_name
        self.bgr_shm_shape = bgr_shm_shape
        self.stop_event = stop_event
        self.db_path = db_path

    def run(self):
        """进程入口 — 对标 Frigate CameraTracker.run()"""
        # Windows spawn 模式: 确保项目根目录在 sys.path 中
        import sys, os
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import cv2
        cam_id = self.cam_config["id"]

        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s [{cam_id}:analyzer] %(levelname)s %(message)s",
        )
        log = logging.getLogger(f"analyzer.{cam_id}")

        # 在子进程中初始化所有组件（避免 pickle 问题）
        from src.vision.motion import MotionDetector
        from src.config.schema import MotionConfig
        from src.vision.stationary import StationaryTracker
        from src.rules.intrusion import IntrusionDetector
        from src.rules.tripwire import TripwireDetector
        from src.rules.counting import FlowCounter
        from src.rules.anomaly import AnomalyEngine
        from src.rules.presence import PresenceDetector
        from src.rules.area_counter import AreaCounter
        from src.events.evidence import EvidenceSaver, draw_overlay
        from src.events.logger import JSONLLogger
        from src.events.database import EventDatabase
        from src.mp.detector_process import RemoteDetector

        # 运动检测器
        motion_cfg = self.cam_config.get("motion", {})
        motion_detector = MotionDetector(MotionConfig(**motion_cfg))

        # 远程检测器代理（通过 Queue 与 DetectorProcess 通信）
        remote_detector = RemoteDetector(
            camera_id=cam_id,
            detect_queue=self.detect_queue,
            result_queue=self.result_queue,
            timeout=5.0,
        )

        # 静止目标跟踪器
        stationary_cfg = self.cam_config.get("stationary", {})
        stationary_tracker = StationaryTracker(
            move_threshold=stationary_cfg.get("move_threshold", 20.0),
            stationary_frames=stationary_cfg.get("stationary_frames", 15),
            recheck_interval=stationary_cfg.get("recheck_interval", 30),
        )

        # 规则引擎
        rules_cfg = self.cam_config.get("rules", {})
        roi = self.cam_config.get("roi")

        intrusion_detector = None
        intrusion_cfg = rules_cfg.get("intrusion", {})
        if roi and intrusion_cfg.get("enabled", False):
            intrusion_detector = IntrusionDetector(
                roi=roi,
                confirm_frames=intrusion_cfg.get("confirm_frames", 5),
                cooldown=intrusion_cfg.get("cooldown", 30),
            )

        tripwire_detectors = []
        tripwires = self.cam_config.get("tripwires", [])
        tw_cfg = rules_cfg.get("tripwire", {})
        if tw_cfg.get("enabled", False):
            for tw in tripwires:
                tripwire_detectors.append(TripwireDetector(
                    tripwire_id=tw["id"], name=tw["name"],
                    p1=tw["p1"], p2=tw["p2"],
                    direction=tw.get("direction", "left_to_right"),
                    cooldown=tw.get("cooldown", 2.0),
                ))

        counter = None
        count_cfg = rules_cfg.get("counting", {})
        if count_cfg.get("enabled", False):
            counter = FlowCounter(
                camera_id=cam_id,
                window_seconds=count_cfg.get("window_seconds", 60),
                db_path=self.db_path,
            )

        anomaly_cfg = rules_cfg.get("anomaly", {})
        anomaly_engine = AnomalyEngine(anomaly_cfg, roi=roi)

        presence_cfg = rules_cfg.get("presence", {})
        presence_detector = PresenceDetector(
            watch_classes=presence_cfg.get("watch_classes", [0, 2, 3, 5, 7]),
            cooldown=presence_cfg.get("cooldown", 10),
            min_confidence=presence_cfg.get("min_confidence", 0.5),
            roi=roi,
        )

        area_counter = AreaCounter(roi=roi)

        evidence_saver = EvidenceSaver(
            output_dir=self.events_config.get("output_dir", "events"),
            draw_bbox=self.events_config.get("draw_bbox", True),
            draw_roi=self.events_config.get("draw_roi", True),
            draw_tripwire=self.events_config.get("draw_tripwire", True),
        )

        jsonl_logger = JSONLLogger(
            self.events_config.get("output_dir", "events"))
        event_db = EventDatabase(db_path=self.db_path)

        # 告警类型过滤
        alert_types = rules_cfg.get("alert_types", [])

        # 帧率节流
        target_fps = self.cam_config.get(
            "analyze_fps", self.model_config.get("analyze_fps", 5))
        min_interval = 1.0 / max(target_fps, 1)
        last_analyze_time = 0.0

        # 帧格式参数
        cam_h = self.cam_config.get("height", 480)
        cam_w = self.cam_config.get("width", 640)
        crop_margin = self.cam_config.get("crop_margin", 50)

        # BGR 共享内存（写入给 Web 层）
        bgr_shm = None
        try:
            bgr_shm = shared_memory.SharedMemory(
                name=self.bgr_shm_name, create=False)
        except Exception:
            try:
                bgr_size = int(np.prod(self.bgr_shm_shape))
                bgr_shm = shared_memory.SharedMemory(
                    name=self.bgr_shm_name, create=True, size=bgr_size)
            except Exception as e:
                log.warning(f"BGR 共享内存创建失败: {e}")

        # 统计
        frame_count = 0
        detect_count = 0
        skip_count = 0
        last_stats_time = time.time()

        log.info(f"分析管线启动 (目标帧率: {target_fps} fps)")

        # 主循环 — 对标 Frigate process_frames()
        while not self.stop_event.is_set():
            try:
                # 从 frame_queue 获取帧通知
                try:
                    shm_name, frame_time = self.frame_queue.get(timeout=1)
                except Exception:
                    continue

                # 从共享内存读取 YUV 帧 — 对标 Frigate: frame_manager.get()
                try:
                    frame_shm = shared_memory.SharedMemory(
                        name=shm_name, create=False)
                    yuv_shape = (cam_h * 3 // 2, cam_w)
                    frame = np.ndarray(
                        yuv_shape, dtype=np.uint8,
                        buffer=frame_shm.buf).copy()
                    frame_shm.close()
                except Exception as e:
                    log.warning(f"读取帧失败 {shm_name}: {e}")
                    continue

                # 帧率节流
                now = time.time()
                if now - last_analyze_time < min_interval:
                    # 节流帧也更新 BGR 展示
                    self._write_bgr_frame(
                        cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420),
                        bgr_shm, self.bgr_shm_shape)
                    continue
                last_analyze_time = now
                frame_count += 1

                # ① 运动检测（YUV Y 通道，低开销）
                motion_boxes, has_motion = motion_detector.detect(
                    frame, frame_height=cam_h, frame_width=cam_w)

                if not has_motion:
                    skip_count += 1
                    stationary_tracker.update([])
                    # 无运动时清空区域计数，避免冻结在旧值
                    area_counter.update([])
                    area_counts = area_counter.get_counts()
                    try:
                        self.event_queue.put_nowait({
                            "_type": "_area_count_update",
                            "camera_id": cam_id,
                            "counts": area_counts,
                        })
                    except Exception:
                        pass
                    self._write_bgr_frame(
                        cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420),
                        bgr_shm, self.bgr_shm_shape)
                    self._update_perf(cam_id, frame_count, detect_count,
                                      skip_count, last_stats_time,
                                      stationary_tracker)
                    continue

                # 有运动 → YUV→BGR
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

                # ② 静止目标过滤
                stationary_dets, need_detect = \
                    stationary_tracker.get_stationary_detections(motion_boxes)

                if not need_detect and stationary_dets:
                    skip_count += 1
                    detections = stationary_dets
                else:
                    # ③ 目标检测 — 通过 RemoteDetector 发送到 DetectorProcess
                    detect_count += 1

                    crop_frame, crop_offset = self._crop_motion_region(
                        bgr_frame, motion_boxes, crop_margin)

                    try:
                        raw_dets = remote_detector.track(crop_frame)
                        detections = self._remap_detections(
                            raw_dets, crop_offset)
                    except Exception as e:
                        log.error(f"检测异常: {e}", exc_info=True)
                        detections = []

                    if stationary_dets:
                        detected_ids = {d.track_id for d in detections
                                        if d.track_id >= 0}
                        for sd in stationary_dets:
                            if sd.track_id not in detected_ids:
                                detections.append(sd)

                stationary_tracker.update(detections)

                # ④ 规则引擎
                all_events = []

                if intrusion_detector:
                    evts = intrusion_detector.update(
                        detections, cam_id, frame_ts=frame_time)
                    all_events.extend(evts)

                tw_events = []
                for tw_det in tripwire_detectors:
                    evts = tw_det.update(
                        detections, cam_id, frame_ts=frame_time)
                    tw_events.extend(evts)
                all_events.extend(tw_events)

                if counter:
                    count_results = counter.update(tw_events)
                    for cr in count_results:
                        jsonl_logger.log_count(cr)
                        event_db.insert_count_window(cr)
                    counts = counter.get_current_counts()
                    # 通过事件队列传递计数更新
                    self.event_queue.put_nowait({
                        "_type": "_count_update",
                        "camera_id": cam_id,
                        "counts": counts,
                    })

                anomaly_events = anomaly_engine.update(
                    detections, cam_id, frame_ts=frame_time)
                all_events.extend(anomaly_events)

                presence_events = presence_detector.update(
                    detections, cam_id, frame_ts=frame_time)
                all_events.extend(presence_events)

                area_counter.update(detections)
                area_counts = area_counter.get_counts()
                area_record = area_counter.maybe_record(cam_id)
                if area_record:
                    event_db.insert_area_count(area_record)

                # 区域计数更新
                try:
                    self.event_queue.put_nowait({
                        "_type": "_area_count_update",
                        "camera_id": cam_id,
                        "counts": area_counts,
                    })
                except Exception:
                    pass

                # ⑤ 事件处理
                for evt in all_events:
                    is_alert = self._should_alert(evt, alert_types)

                    if is_alert:
                        # 高频低价值事件不保存截图，减少磁盘 IO
                        evt_type = evt.get("type", "")
                        if evt_type not in self._NO_SCREENSHOT_TYPES:
                            try:
                                filepath = evidence_saver.save_screenshot(
                                    bgr_frame, evt,
                                    roi=roi, tripwires=tripwires,
                                    detections=detections,
                                )
                                evt["screenshot"] = os.path.basename(filepath)
                            except Exception as e:
                                log.error(f"截图保存失败: {e}")

                        jsonl_logger.log_event(evt)
                        try:
                            self.event_queue.put_nowait(evt)
                        except Exception:
                            pass

                    event_db.insert_event(evt)

                # ⑥ 更新 BGR 展示帧
                try:
                    overlay = draw_overlay(
                        bgr_frame, detections,
                        roi=roi, tripwires=tripwires)
                    self._write_bgr_frame(overlay, bgr_shm, self.bgr_shm_shape)
                except Exception:
                    self._write_bgr_frame(bgr_frame, bgr_shm, self.bgr_shm_shape)

                self._update_perf(cam_id, frame_count, detect_count,
                                  skip_count, last_stats_time,
                                  stationary_tracker)

            except Exception as e:
                log.error(f"分析异常: {e}", exc_info=True)
                time.sleep(0.1)

        # 清理
        if bgr_shm:
            try:
                bgr_shm.close()
            except Exception:
                pass
        log.info("分析进程退出")

    @staticmethod
    def _write_bgr_frame(frame: np.ndarray, bgr_shm, bgr_shape: tuple):
        """将 BGR 帧写入共享内存（供 Web 层读取）"""
        if bgr_shm is None:
            return
        try:
            target = np.ndarray(bgr_shape, dtype=np.uint8, buffer=bgr_shm.buf)
            # 确保帧尺寸匹配
            if frame.shape == bgr_shape:
                np.copyto(target, frame)
            else:
                import cv2
                resized = cv2.resize(frame, (bgr_shape[1], bgr_shape[0]))
                np.copyto(target, resized)
        except Exception:
            pass

    def _update_perf(self, cam_id, frame_count, detect_count,
                     skip_count, last_stats_time, stationary_tracker):
        """更新性能统计到共享 dict"""
        now = time.time()
        elapsed = now - last_stats_time
        if elapsed < 5:
            return
        try:
            self.perf_dict[cam_id] = {
                "fps": round(frame_count / elapsed, 1),
                "detection_fps": round(detect_count / elapsed, 1),
                "skip_rate": round(
                    (skip_count / max(frame_count, 1)) * 100, 1),
                "stationary_objects": stationary_tracker.stationary_count,
                "tracked_objects": stationary_tracker.tracked_count,
            }
        except Exception:
            pass

    @staticmethod
    def _should_alert(evt: dict, alert_types: list) -> bool:
        if not alert_types:
            return True
        evt_type = evt.get("type", "")
        sub_type = evt.get("sub_type", "")
        for at in alert_types:
            if at == evt_type:
                return True
            if "/" in at:
                t, s = at.split("/", 1)
                if t == evt_type and s == sub_type:
                    return True
        return False

    # 不保存截图的事件类型（高频低价值）
    _NO_SCREENSHOT_TYPES = {"presence"}

    @staticmethod
    def _crop_motion_region(frame, motion_boxes, margin=50):
        """
        根据运动区域裁剪帧，减少推理面积。
        margin 足够大以避免截断目标。如果运动区域覆盖大部分画面则返回全帧。
        """
        if not motion_boxes:
            return frame, (0, 0)

        h, w = frame.shape[:2]
        # 合并所有运动区域（MotionBox 有 .x1 .y1 .x2 .y2 属性）
        x_min = min(b.x1 for b in motion_boxes)
        y_min = min(b.y1 for b in motion_boxes)
        x_max = max(b.x2 for b in motion_boxes)
        y_max = max(b.y2 for b in motion_boxes)

        # 加大 margin（至少 bbox 尺寸的 50%，最小 100px）
        box_w = x_max - x_min
        box_h = y_max - y_min
        m = max(margin, int(max(box_w, box_h) * 0.5), 100)

        cx1 = max(0, x_min - m)
        cy1 = max(0, y_min - m)
        cx2 = min(w, x_max + m)
        cy2 = min(h, y_max + m)

        # 如果裁剪区域超过原图 70%，直接全帧
        crop_area = (cx2 - cx1) * (cy2 - cy1)
        if crop_area > 0.7 * w * h:
            return frame, (0, 0)

        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    @staticmethod
    def _remap_detections(detections, offset):
        ox, oy = offset
        if ox == 0 and oy == 0:
            return detections
        for det in detections:
            det.bbox[0] += ox
            det.bbox[1] += oy
            det.bbox[2] += ox
            det.bbox[3] += oy
            cx, cy = det.center
            det.center = (cx + ox, cy + oy)
            fx, fy = det.foot
            det.foot = (fx + ox, fy + oy)
        return detections
