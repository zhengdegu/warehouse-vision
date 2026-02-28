"""
主程序调度模块 — 参考 Frigate 多进程架构
视频管线: 拉流 → 运动检测 → 目标检测 → 规则引擎 → 事件处理

核心优化:
- 运动检测作为预过滤，只在有运动时运行 AI（大幅降低负载）
- 每路摄像头独立线程，检测器独立实例
- 共享状态通过线程安全缓存传递给 Web 层
- SQLite 替代 JSONL，支持结构化查询
"""

import os
import time
import threading
import logging
from typing import Dict, Any, List, Optional
from queue import Queue

import yaml
import numpy as np

from .config.schema import AppConfig, CameraConfig
from .ingest.rtsp_reader import RTSPReader
from .vision.detector import YOLODetector, Detection, PoseDetector
from .vision.motion import MotionDetector
from .rules.intrusion import IntrusionDetector
from .rules.tripwire import TripwireDetector
from .rules.counting import FlowCounter
from .rules.anomaly import AnomalyEngine
from .rules.presence import PresenceDetector
from .events.evidence import EvidenceSaver, draw_overlay
from .events.logger import JSONLLogger
from .events.database import EventDatabase
from .events.es_store import ESEventStore

logger = logging.getLogger(__name__)


class SharedState:
    """分析线程与 Web 层之间的共享状态"""

    def __init__(self):
        self.latest_frame_cache: Dict[str, np.ndarray] = {}
        self.count_cache: Dict[str, Dict[str, Any]] = {}
        self.event_queue: Queue = Queue(maxsize=1000)
        self.cameras: List[Dict[str, Any]] = []
        # 系统性能指标
        self.perf_stats: Dict[str, Dict[str, Any]] = {}
        self._frame_lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._perf_lock = threading.Lock()

    def update_frame(self, camera_id: str, frame: np.ndarray):
        with self._frame_lock:
            self.latest_frame_cache[camera_id] = frame

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self.latest_frame_cache.get(camera_id)

    def update_counts(self, camera_id: str, counts: Dict[str, Any]):
        with self._count_lock:
            self.count_cache[camera_id] = counts

    def get_counts(self) -> Dict[str, Dict[str, Any]]:
        with self._count_lock:
            return dict(self.count_cache)

    def update_perf(self, camera_id: str, stats: Dict[str, Any]):
        with self._perf_lock:
            self.perf_stats[camera_id] = stats

    def get_perf(self) -> Dict[str, Dict[str, Any]]:
        with self._perf_lock:
            return dict(self.perf_stats)

    def push_event(self, event: Dict[str, Any]):
        try:
            self.event_queue.put_nowait(event)
        except Exception:
            try:
                self.event_queue.get_nowait()
                self.event_queue.put_nowait(event)
            except Exception:
                pass


class CameraAnalyzer:
    """
    单路摄像头分析管线
    Pipeline: 拉流 → 运动检测 → 目标检测(仅运动区域) → 规则引擎 → 事件
    """

    def __init__(self, cam_config: dict, model_config: dict,
                 events_config: dict, shared: SharedState,
                 jsonl_logger: JSONLLogger, event_db: EventDatabase,
                 es_store: ESEventStore = None,
                 detector: YOLODetector = None,
                 pose_detector: PoseDetector = None):
        self.cam_id = cam_config["id"]
        self.cam_config = cam_config
        self.shared = shared
        self.jsonl_logger = jsonl_logger
        self.event_db = event_db
        self.es_store = es_store
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 崩溃恢复状态
        self._crash_count = 0
        self._last_crash_time = 0.0
        self._consecutive_errors = 0
        self._max_consecutive_errors = 50  # 连续异常超过此数触发自动重启拉流

        # 性能统计
        self._frame_count = 0
        self._detect_count = 0
        self._skip_count = 0
        self._last_stats_time = time.time()

        # 分析帧率节流（降低 CPU 占用）
        self._target_analyze_fps = cam_config.get("analyze_fps",
                                                   model_config.get("analyze_fps", 5))
        self._min_frame_interval = 1.0 / max(self._target_analyze_fps, 1)
        self._last_analyze_time = 0.0

        # RTSP 拉流（width/height 传 0 则自动探测）
        self.reader = RTSPReader(
            url=cam_config["url"],
            width=cam_config.get("width", 0),
            height=cam_config.get("height", 0),
            fps=cam_config.get("fps", 15),
        )

        # 运动检测器 — Frigate 核心优化
        motion_cfg = cam_config.get("motion", {})
        from .config.schema import MotionConfig
        self.motion_detector = MotionDetector(MotionConfig(**motion_cfg))

        # YOLO 检测器 — 多摄像头共享同一实例，避免重复加载
        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLODetector(
                model_path=model_config.get("path", "yolo26m.pt"),
                confidence=model_config.get("confidence", 0.5),
                allowed_classes=model_config.get("classes"),
            )

        # Pose 检测器（打架/跌倒姿态增强，可选）— 同样共享
        pose_config = model_config.get("pose", {})
        self.pose_detector = None
        if pose_config.get("enabled", False):
            if pose_detector is not None:
                self.pose_detector = pose_detector
            else:
                self.pose_detector = PoseDetector(
                    model_path=pose_config.get("path", "yolo26m-pose.pt"),
                    confidence=pose_config.get("confidence", 0.3),
                )

        # 规则引擎
        rules_cfg = cam_config.get("rules", {})
        roi = cam_config.get("roi")

        # 入侵检测
        self.intrusion_detector = None
        intrusion_cfg = rules_cfg.get("intrusion", {})
        if roi and intrusion_cfg.get("enabled", False):
            self.intrusion_detector = IntrusionDetector(
                roi=roi,
                confirm_frames=intrusion_cfg.get("confirm_frames", 5),
                cooldown=intrusion_cfg.get("cooldown", 30),
            )

        # 越线检测
        self.tripwire_detectors: List[TripwireDetector] = []
        tripwires = cam_config.get("tripwires", [])
        tw_cfg = rules_cfg.get("tripwire", {})
        if tw_cfg.get("enabled", False):
            for tw in tripwires:
                self.tripwire_detectors.append(TripwireDetector(
                    tripwire_id=tw["id"],
                    name=tw["name"],
                    p1=tw["p1"],
                    p2=tw["p2"],
                    direction=tw.get("direction", "left_to_right"),
                    cooldown=tw.get("cooldown", 2.0),
                ))

        # 计数器
        self.counter = None
        count_cfg = rules_cfg.get("counting", {})
        if count_cfg.get("enabled", False):
            self.counter = FlowCounter(
                camera_id=self.cam_id,
                window_seconds=count_cfg.get("window_seconds", 60),
            )
            saved = event_db.get_camera_stats(self.cam_id)
            if saved:
                self.counter.total_in = saved.get("total_in", 0)
                self.counter.total_out = saved.get("total_out", 0)

        # 异常检测
        anomaly_cfg = rules_cfg.get("anomaly", {})
        self.anomaly_engine = AnomalyEngine(anomaly_cfg, roi=roi)

        # 存在检测
        presence_cfg = rules_cfg.get("presence", {})
        self.presence_detector = PresenceDetector(
            watch_classes=presence_cfg.get("watch_classes", [0, 2, 3, 5, 7]),
            cooldown=presence_cfg.get("cooldown", 10),
            min_confidence=presence_cfg.get("min_confidence", 0.5),
            roi=roi,
        )

        # 截图保存
        self.evidence_saver = EvidenceSaver(
            output_dir=events_config.get("output_dir", "events"),
            draw_bbox=events_config.get("draw_bbox", True),
            draw_roi=events_config.get("draw_roi", True),
            draw_tripwire=events_config.get("draw_tripwire", True),
        )

        self._roi = roi
        self._tripwires = tripwires

        # 告警类型过滤：空列表=全部告警，否则只有匹配的类型才触发告警
        # 支持格式: "tripwire", "intrusion", "presence", "anomaly/dwell" 等
        self._alert_types: list = rules_cfg.get("alert_types", [])



    def _should_alert(self, evt: dict) -> bool:
        """
        判断事件是否应触发告警。
        alert_types 为空 → 全部告警
        否则匹配 "tripwire", "intrusion", "presence", "anomaly/dwell" 等
        """
        if not self._alert_types:
            return True  # 未配置 = 全部告警

        evt_type = evt.get("type", "")
        sub_type = evt.get("sub_type", "")

        for at in self._alert_types:
            # 精确匹配: "tripwire", "intrusion"
            if at == evt_type:
                return True
            # 子类型匹配: "anomaly/dwell"
            if "/" in at:
                t, s = at.split("/", 1)
                if t == evt_type and s == sub_type:
                    return True

        return False

    def _update_perf_stats(self):
        """更新性能统计"""
        now = time.time()
        elapsed = now - self._last_stats_time
        if elapsed < 5:
            return
        fps = self._frame_count / elapsed
        det_fps = self._detect_count / elapsed
        skip_rate = (self._skip_count / max(self._frame_count, 1)) * 100

        self.shared.update_perf(self.cam_id, {
            "fps": round(fps, 1),
            "detection_fps": round(det_fps, 1),
            "skip_rate": round(skip_rate, 1),
            "frames_processed": self._frame_count,
            "detections_run": self._detect_count,
        })

        self._frame_count = 0
        self._detect_count = 0
        self._skip_count = 0
        self._last_stats_time = now

    def _analyze_loop(self):
        """分析主循环 — 带运动预过滤 + 帧率节流 + 崩溃自恢复"""
        logger.info(f"[{self.cam_id}] 分析管线启动 (目标分析帧率: {self._target_analyze_fps} fps)")
        no_frame_count = 0

        while self._running:
            try:
                frame = self.reader.read_latest()
                if frame is None:
                    no_frame_count += 1
                    # 长时间无帧 → 拉流可能挂了，触发 reader 重启
                    if no_frame_count > 300:  # ~3s at 0.01s sleep
                        logger.warning(f"[{self.cam_id}] 长时间无帧，重启拉流")
                        self._restart_reader()
                        no_frame_count = 0
                    time.sleep(0.01)
                    continue

                no_frame_count = 0
                self._consecutive_errors = 0

                # 帧率节流：控制分析频率，避免 CPU 空转
                now = time.time()
                elapsed_since_last = now - self._last_analyze_time
                if elapsed_since_last < self._min_frame_interval:
                    self.shared.update_frame(self.cam_id, frame)
                    time.sleep(max(0, self._min_frame_interval - elapsed_since_last))
                    continue
                self._last_analyze_time = now

                self._frame_count += 1

                # ① 运动检测（低开销预过滤）
                motion_boxes, has_motion = self.motion_detector.detect(frame)

                if not has_motion:
                    self._skip_count += 1
                    self.shared.update_frame(self.cam_id, frame)
                    self._update_perf_stats()
                    continue

                # ② 目标检测 + 跟踪（仅在有运动时执行）
                self._detect_count += 1
                try:
                    detections = self.detector.track(frame)
                except Exception as det_err:
                    logger.error(f"[{self.cam_id}] 检测异常: {det_err}",
                                 exc_info=True)
                    detections = []

                # ②-b Pose 检测（可选，为打架/跌倒提供关键点）
                if self.pose_detector and detections:
                    try:
                        pose_dets = self.pose_detector.track(frame)
                        pose_map = {d.track_id: d.keypoints
                                    for d in pose_dets
                                    if d.track_id >= 0 and d.keypoints is not None}
                        if pose_map:
                            for det in detections:
                                if det.class_name == "person" and det.track_id in pose_map:
                                    det.keypoints = pose_map[det.track_id]
                    except Exception as pose_err:
                        logger.error(f"[{self.cam_id}] Pose 检测异常: {pose_err}",
                                     exc_info=True)

                all_events = []

                # ③ 规则引擎
                if self.intrusion_detector:
                    events = self.intrusion_detector.update(
                        detections, self.cam_id)
                    all_events.extend(events)

                tw_events = []
                for tw_det in self.tripwire_detectors:
                    events = tw_det.update(detections, self.cam_id)
                    tw_events.extend(events)
                all_events.extend(tw_events)

                if self.counter:
                    count_results = self.counter.update(tw_events)
                    for cr in count_results:
                        self.jsonl_logger.log_count(cr)
                        self.event_db.insert_count_window(cr)
                    counts = self.counter.get_current_counts()
                    self.shared.update_counts(self.cam_id, counts)
                    if tw_events:
                        self.jsonl_logger.update_counts(
                            self.cam_id, counts["total_in"], counts["total_out"])
                        self.event_db.update_camera_stats(
                            self.cam_id, counts["total_in"], counts["total_out"])

                anomaly_events = self.anomaly_engine.update(
                    detections, self.cam_id)
                all_events.extend(anomaly_events)

                presence_events = self.presence_detector.update(
                    detections, self.cam_id)
                all_events.extend(presence_events)

                # ④ 事件处理（区分记录 vs 告警）
                for evt in all_events:
                    is_alert = self._should_alert(evt)

                    if is_alert:
                        try:
                            filepath = self.evidence_saver.save_screenshot(
                                frame, evt,
                                roi=self._roi,
                                tripwires=self._tripwires,
                                detections=detections,
                            )
                            evt["screenshot"] = os.path.basename(filepath)
                        except Exception as e:
                            logger.error(f"截图保存失败: {e}")

                        self.jsonl_logger.log_event(evt)
                        self.event_db.update_camera_stats(
                            self.cam_id, increment_alert=True)
                        self.shared.push_event(evt)

                        if self.es_store:
                            self.es_store.store_event(evt)

                    self.event_db.insert_event(evt)

                # ⑤ 更新 Web 展示帧
                try:
                    overlay_frame = draw_overlay(
                        frame, detections,
                        roi=self._roi,
                        tripwires=self._tripwires,
                    )
                    self.shared.update_frame(self.cam_id, overlay_frame)
                except Exception:
                    self.shared.update_frame(self.cam_id, frame)

                self._update_perf_stats()

            except Exception as e:
                self._consecutive_errors += 1
                logger.error(f"[{self.cam_id}] 分析异常 (连续第{self._consecutive_errors}次): {e}",
                             exc_info=True)

                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error(f"[{self.cam_id}] 连续异常过多，重启拉流")
                    self._restart_reader()
                    self._consecutive_errors = 0

                time.sleep(0.5)

        logger.info(f"[{self.cam_id}] 分析管线退出")

    def _restart_reader(self):
        """安全重启拉流进程"""
        try:
            self.reader.stop()
        except Exception:
            pass
        time.sleep(1)
        try:
            self.reader.start()
            logger.info(f"[{self.cam_id}] 拉流已重启")
        except Exception as e:
            logger.error(f"[{self.cam_id}] 拉流重启失败: {e}")

    def start(self):
        self._running = True
        self._crash_count = 0
        self.reader.start()
        self._thread = threading.Thread(
            target=self._analyze_loop, daemon=True,
            name=f"analyzer-{self.cam_id}")
        self._thread.start()

    def stop(self):
        self._running = False
        self.reader.stop()
        if self._thread:
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        """检查分析线程是否存活"""
        return (self._running
                and self._thread is not None
                and self._thread.is_alive())


def load_config(config_path: str = "configs/cameras.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str = "configs/cameras.yaml"):
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class Application:
    """主应用 — 管理所有摄像头管线和共享资源"""

    def __init__(self, config_path: str = "configs/cameras.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.shared = SharedState()
        self.jsonl_logger = JSONLLogger(
            self.config.get("events", {}).get("output_dir", "events"))

        # SQLite 事件数据库
        db_cfg = self.config.get("database", {})
        self.event_db = EventDatabase(
            db_path=db_cfg.get("path", "data/warehouse_vision.db"))

        # Elasticsearch 告警存储
        es_cfg = self.config.get("elasticsearch", {})
        self.es_store = ESEventStore(
            host=es_cfg.get("host", "http://localhost:9222"),
            index_prefix=es_cfg.get("index_prefix", "warehouse-alerts"),
            enabled=es_cfg.get("enabled", False),
        )

        self.analyzers: Dict[str, CameraAnalyzer] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_running = False

        cameras = self.config.get("cameras", [])
        self.shared.cameras = cameras

        model_config = self.config.get("model", {})
        events_config = self.config.get("events", {})

        # 共享模型实例 — 所有摄像头复用同一个检测器，避免重复加载
        self._shared_detector = YOLODetector(
            model_path=model_config.get("path", "yolo26m.pt"),
            confidence=model_config.get("confidence", 0.5),
            allowed_classes=model_config.get("classes"),
        )
        self._shared_pose_detector = None
        pose_config = model_config.get("pose", {})
        if pose_config.get("enabled", False):
            self._shared_pose_detector = PoseDetector(
                model_path=pose_config.get("path", "yolo26m-pose.pt"),
                confidence=pose_config.get("confidence", 0.3),
            )

        for cam_cfg in cameras:
            analyzer = CameraAnalyzer(
                cam_config=cam_cfg,
                model_config=model_config,
                events_config=events_config,
                shared=self.shared,
                jsonl_logger=self.jsonl_logger,
                event_db=self.event_db,
                es_store=self.es_store,
                detector=self._shared_detector,
                pose_detector=self._shared_pose_detector,
            )
            self.analyzers[cam_cfg["id"]] = analyzer

    def _persist_config(self):
        self.config["cameras"] = list(self.shared.cameras)
        save_config(self.config, self.config_path)

    def add_camera(self, cam_cfg: dict):
        cam_id = cam_cfg["id"]
        with self._lock:
            if cam_id in self.analyzers:
                self.analyzers[cam_id].stop()

            model_config = self.config.get("model", {})
            events_config = self.config.get("events", {})

            analyzer = CameraAnalyzer(
                cam_config=cam_cfg,
                model_config=model_config,
                events_config=events_config,
                shared=self.shared,
                jsonl_logger=self.jsonl_logger,
                event_db=self.event_db,
                es_store=self.es_store,
                detector=self._shared_detector,
                pose_detector=self._shared_pose_detector,
            )
            self.analyzers[cam_id] = analyzer
            analyzer.start()

            self.shared.cameras = [
                c for c in self.shared.cameras if c["id"] != cam_id
            ]
            self.shared.cameras.append(cam_cfg)
            self._persist_config()

        self.es_store.store_camera_config(cam_cfg)
        logger.info(f"摄像头 {cam_id} 已添加并启动")

    def update_camera(self, cam_id: str, cam_cfg: dict):
        cam_cfg["id"] = cam_id
        self.add_camera(cam_cfg)
        logger.info(f"摄像头 {cam_id} 已更新")

    def remove_camera(self, cam_id: str):
        with self._lock:
            if cam_id in self.analyzers:
                self.analyzers[cam_id].stop()
                del self.analyzers[cam_id]

            self.shared.cameras = [
                c for c in self.shared.cameras if c["id"] != cam_id
            ]
            self.shared.latest_frame_cache.pop(cam_id, None)
            self.shared.count_cache.pop(cam_id, None)
            self._persist_config()

        self.es_store.delete_camera_config(cam_id)
        logger.info(f"摄像头 {cam_id} 已移除")

    def get_camera_config(self, cam_id: str) -> Optional[dict]:
        for cam in self.shared.cameras:
            if cam["id"] == cam_id:
                return cam
        return None

    def get_all_camera_configs(self) -> List[dict]:
        return list(self.shared.cameras)

    def start(self):
        logger.info(f"启动 {len(self.analyzers)} 路摄像头分析管线")
        for analyzer in self.analyzers.values():
            analyzer.start()
        # 同步摄像头配置到 ES
        for cam_cfg in self.shared.cameras:
            self.es_store.store_camera_config(cam_cfg)
        # 启动看门狗 + 定期清理
        self._start_watchdog()
        self._start_cleanup_thread()

    def _start_watchdog(self):
        """
        看门狗线程：每 10 秒检查所有摄像头分析线程，
        崩溃的自动重建并重启，带指数退避防止频繁重启。
        """
        self._watchdog_running = True

        def _watchdog_loop():
            logger.info("看门狗已启动，监控间隔 10s")
            while self._watchdog_running:
                time.sleep(10)
                with self._lock:
                    for cam_id, analyzer in list(self.analyzers.items()):
                        if not analyzer._running:
                            # 被主动 stop 的，不恢复
                            continue
                        if analyzer.is_alive():
                            continue

                        # 线程已死但 _running 仍为 True → 崩溃了
                        analyzer._crash_count += 1
                        analyzer._last_crash_time = time.time()

                        # 指数退避：2^n 秒，最大 120 秒
                        backoff = min(2 ** analyzer._crash_count, 120)
                        logger.warning(
                            f"[看门狗] {cam_id} 分析线程已崩溃 "
                            f"(第{analyzer._crash_count}次)，"
                            f"{backoff}s 后重启"
                        )
                        time.sleep(backoff)

                        # 重建分析器
                        try:
                            self._recover_analyzer(cam_id, analyzer)
                        except Exception as e:
                            logger.error(
                                f"[看门狗] {cam_id} 恢复失败: {e}",
                                exc_info=True
                            )

        self._watchdog_thread = threading.Thread(
            target=_watchdog_loop, daemon=True, name="watchdog")
        self._watchdog_thread.start()

    def _recover_analyzer(self, cam_id: str, old_analyzer: 'CameraAnalyzer'):
        """重建并重启一个崩溃的摄像头分析器"""
        crash_count = old_analyzer._crash_count

        # 先清理旧的
        try:
            old_analyzer.stop()
        except Exception:
            pass

        # 用原始配置重建
        model_config = self.config.get("model", {})
        events_config = self.config.get("events", {})

        new_analyzer = CameraAnalyzer(
            cam_config=old_analyzer.cam_config,
            model_config=model_config,
            events_config=events_config,
            shared=self.shared,
            jsonl_logger=self.jsonl_logger,
            event_db=self.event_db,
            es_store=self.es_store,
            detector=self._shared_detector,
            pose_detector=self._shared_pose_detector,
        )
        # 继承崩溃计数
        new_analyzer._crash_count = crash_count

        self.analyzers[cam_id] = new_analyzer
        new_analyzer.start()
        logger.info(f"[看门狗] {cam_id} 已恢复启动 (累计崩溃{crash_count}次)")

    def _start_cleanup_thread(self):
        """定期清理过期数据"""
        def _cleanup_loop():
            while True:
                time.sleep(3600)  # 每小时清理一次
                try:
                    self.event_db.cleanup_old_events(retain_days=30)
                except Exception as e:
                    logger.error(f"清理过期数据失败: {e}")
        self._cleanup_thread = threading.Thread(
            target=_cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def reload_model(self, new_model_path: str) -> None:
        model_config = self.config.get("model", {})
        confidence = model_config.get("confidence", 0.5)
        allowed_classes = model_config.get("classes")

        try:
            new_detector = YOLODetector(
                model_path=new_model_path,
                confidence=confidence,
                allowed_classes=allowed_classes,
            )
        except Exception as e:
            logger.error(f"新模型加载失败，保持旧模型运行: {new_model_path}, 错误: {e}")
            return

        with self._lock:
            self._shared_detector = new_detector
            for cam_id, analyzer in self.analyzers.items():
                analyzer.detector = new_detector
                logger.info(f"[{cam_id}] 检测器已替换为新模型: {new_model_path}")

            if "model" not in self.config:
                self.config["model"] = {}
            self.config["model"]["path"] = new_model_path
            self._persist_config()

        logger.info(f"模型热重载完成: {new_model_path}")

    def stop(self):
        logger.info("停止所有分析管线...")
        self._watchdog_running = False
        for analyzer in self.analyzers.values():
            analyzer.stop()
        logger.info("所有分析管线已停止")
