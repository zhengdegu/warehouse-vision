"""
主程序调度模块 — 参考 Frigate 多进程架构

视频管线: 拉流(进程) → 运动检测(进程) → 目标检测(进程) → 规则引擎 → 事件处理

核心架构（对标 Frigate）:
- 每路摄像头 2 个独立进程: CaptureProcess + AnalyzerProcess
- 检测器 1 个独立进程: DetectorProcess（所有摄像头共享）
- 进程间通信: 共享内存(帧) + Queue(通知/事件)
- Web 层通过共享内存零拷贝读取 BGR 帧

参考:
- frigate/frigate/camera/maintainer.py: CameraMaintainer
- frigate/frigate/video.py: CameraCapture, CameraTracker
- frigate/frigate/object_detection/base.py: ObjectDetectProcess, DetectorRunner
"""

import os
import time
import threading
import logging
import multiprocessing as mp
from typing import Dict, Any, List, Optional
from queue import Empty

import yaml
import numpy as np
import cv2

from .config.schema import AppConfig, CameraConfig
from .ingest.go2rtc import Go2RTCManager
from .events.logger import JSONLLogger
from .events.database import EventDatabase
from .events.es_store import ESEventStore
from .mp.capture_process import CaptureProcess
from .mp.analyzer_process import AnalyzerProcess
from .mp.detector_process import DetectorProcess

logger = logging.getLogger(__name__)

# 共享内存帧缓冲区大小（对标 Frigate shm_frame_count）
SHM_FRAME_COUNT = 10



class SharedState:
    """
    Web 层共享状态 — 从共享内存读取帧，从 Queue 接收事件。

    与旧版的区别:
    - 旧版: 分析线程直接写入 dict（线程锁保护）
    - 新版: 帧通过共享内存读取，事件通过 Queue 接收
    """

    def __init__(self):
        self.cameras: List[Dict[str, Any]] = []
        self.count_cache: Dict[str, Dict[str, Any]] = {}
        self.area_count_cache: Dict[str, Dict[str, Any]] = {}
        self.event_queue: mp.Queue = mp.Queue(maxsize=1000)
        self.perf_stats: Dict[str, Dict[str, Any]] = {}

        # BGR 帧共享内存映射: {camera_id: (shm_name, shape)}
        self._bgr_shm_info: Dict[str, tuple] = {}
        self._bgr_shm_cache: Dict[str, Any] = {}

        self._count_lock = threading.Lock()
        self._area_count_lock = threading.Lock()
        self._perf_lock = threading.Lock()

    def register_bgr_shm(self, camera_id: str, shm_name: str, shape: tuple):
        """注册 BGR 帧共享内存信息"""
        self._bgr_shm_info[camera_id] = (shm_name, shape)

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """从共享内存读取 BGR 帧（零拷贝 → copy 返回）"""
        info = self._bgr_shm_info.get(camera_id)
        if info is None:
            return None

        shm_name, shape = info
        try:
            if camera_id not in self._bgr_shm_cache:
                from multiprocessing import shared_memory
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
                self._bgr_shm_cache[camera_id] = shm

            shm = self._bgr_shm_cache[camera_id]
            frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
            return frame.copy()
        except Exception:
            # 共享内存可能还没创建
            return None

    def update_frame(self, camera_id: str, frame: np.ndarray):
        """兼容旧接口 — 直接写入共享内存"""
        info = self._bgr_shm_info.get(camera_id)
        if info is None:
            return
        shm_name, shape = info
        try:
            if camera_id not in self._bgr_shm_cache:
                from multiprocessing import shared_memory
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
                self._bgr_shm_cache[camera_id] = shm
            shm = self._bgr_shm_cache[camera_id]
            target = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
            if frame.shape == shape:
                np.copyto(target, frame)
        except Exception:
            pass

    def update_counts(self, camera_id: str, counts: Dict[str, Any]):
        with self._count_lock:
            self.count_cache[camera_id] = counts

    def get_counts(self) -> Dict[str, Dict[str, Any]]:
        with self._count_lock:
            return dict(self.count_cache)

    def update_area_counts(self, camera_id: str, counts: Dict[str, Any]):
        with self._area_count_lock:
            self.area_count_cache[camera_id] = counts

    def get_area_counts(self) -> Dict[str, Dict[str, Any]]:
        with self._area_count_lock:
            return dict(self.area_count_cache)

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

    def cleanup_shm(self):
        """清理共享内存缓存"""
        for shm in self._bgr_shm_cache.values():
            try:
                shm.close()
            except Exception:
                pass
        self._bgr_shm_cache.clear()


def load_config(config_path: str = "configs/cameras.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str = "configs/cameras.yaml"):
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)



class CameraProcessGroup:
    """
    单路摄像头的进程组 — 对标 Frigate CameraMaintainer 中的 per-camera 管理。

    包含:
    - CaptureProcess: ffmpeg 拉流 + 写共享内存
    - AnalyzerProcess: 运动检测 + 规则引擎
    - 相关的 Queue 和共享内存
    """

    def __init__(self, cam_config: dict, model_config: dict,
                 events_config: dict,
                 detect_queue: mp.Queue,
                 result_queue: mp.Queue,
                 event_queue: mp.Queue,
                 perf_dict: dict,
                 stop_event: mp.Event,
                 db_path: str):
        self.cam_id = cam_config["id"]
        self.cam_config = cam_config
        self.stop_event = stop_event

        cam_w = cam_config.get("width", 0)
        cam_h = cam_config.get("height", 0)
        fps = cam_config.get("fps", 15)

        # 帧通知队列: capture → analyzer
        self.frame_queue = mp.Queue(maxsize=SHM_FRAME_COUNT)

        # BGR 共享内存（analyzer 写入，Web 读取）
        # 如果分辨率未知，先用默认值，capture 探测后会更新
        bgr_h = cam_h if cam_h > 0 else 720
        bgr_w = cam_w if cam_w > 0 else 1280
        self.bgr_shm_name = f"wv_bgr_{self.cam_id}"
        self.bgr_shm_shape = (bgr_h, bgr_w, 3)
        bgr_size = bgr_h * bgr_w * 3

        # 预创建 BGR 共享内存
        from multiprocessing import shared_memory
        try:
            self._bgr_shm = shared_memory.SharedMemory(
                name=self.bgr_shm_name, create=True, size=bgr_size)
        except FileExistsError:
            self._bgr_shm = shared_memory.SharedMemory(
                name=self.bgr_shm_name, create=False)

        # CaptureProcess
        self.capture = CaptureProcess(
            camera_id=self.cam_id,
            url=cam_config["url"],
            width=cam_w,
            height=cam_h,
            fps=fps,
            shm_frame_count=SHM_FRAME_COUNT,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            use_restream=True,
            time_offset=cam_config.get("time_offset", 0.0) or 0.0,
        )

        # AnalyzerProcess
        self.analyzer = AnalyzerProcess(
            cam_config=cam_config,
            model_config=model_config,
            events_config=events_config,
            frame_queue=self.frame_queue,
            detect_queue=detect_queue,
            result_queue=result_queue,
            event_queue=event_queue,
            perf_dict=perf_dict,
            bgr_shm_name=self.bgr_shm_name,
            bgr_shm_shape=self.bgr_shm_shape,
            stop_event=self.stop_event,
            db_path=db_path,
        )

    def start(self):
        """启动 capture + analyzer 进程"""
        self.capture.start()
        self.analyzer.start()
        logger.info(
            f"[{self.cam_id}] 进程组已启动 "
            f"(capture pid={self.capture.pid}, "
            f"analyzer pid={self.analyzer.pid})")

    def stop(self):
        """停止进程组"""
        self.stop_event.set()
        if self.capture.is_alive():
            self.capture.terminate()
            self.capture.join(timeout=10)
        if self.analyzer.is_alive():
            self.analyzer.terminate()
            self.analyzer.join(timeout=10)
        # 清理 BGR 共享内存
        try:
            self._bgr_shm.close()
            self._bgr_shm.unlink()
        except Exception:
            pass
        logger.info(f"[{self.cam_id}] 进程组已停止")

    def is_alive(self) -> bool:
        return self.capture.is_alive() and self.analyzer.is_alive()



class Application:
    """
    主应用 — 对标 Frigate CameraMaintainer + AppRunner。

    管理所有摄像头进程组和检测器进程。

    架构:
    ┌─────────────────────────────────────────────────────────┐
    │ 主进程 (Application)                                     │
    │  ├─ Web Server (FastAPI)                                 │
    │  ├─ Event Dispatcher (从 event_queue 读取事件)           │
    │  └─ Watchdog (监控子进程健康)                             │
    ├─────────────────────────────────────────────────────────┤
    │ DetectorProcess (独立进程)                                │
    │  └─ 加载模型 → 从 detect_queue 取请求 → 推理 → 返回结果  │
    ├─────────────────────────────────────────────────────────┤
    │ Camera 1:                                                │
    │  ├─ CaptureProcess: ffmpeg → 共享内存                    │
    │  └─ AnalyzerProcess: 运动检测 → 检测 → 规则 → 事件      │
    ├─────────────────────────────────────────────────────────┤
    │ Camera 2:                                                │
    │  ├─ CaptureProcess                                       │
    │  └─ AnalyzerProcess                                      │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, config_path: str = "configs/cameras.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.shared = SharedState()
        self.jsonl_logger = JSONLLogger(
            self.config.get("events", {}).get("output_dir", "events"))

        # SQLite 事件数据库
        db_cfg = self.config.get("database", {})
        self.db_path = db_cfg.get("path", "data/warehouse_vision.db")
        self.event_db = EventDatabase(db_path=self.db_path)

        # Elasticsearch
        es_cfg = self.config.get("elasticsearch", {})
        self.es_store = ESEventStore(
            host=es_cfg.get("host", "http://localhost:9222"),
            index_prefix=es_cfg.get("index_prefix", "warehouse-alerts"),
            enabled=es_cfg.get("enabled", False),
        )

        # go2rtc 流管理器
        go2rtc_cfg = self.config.get("go2rtc", {})
        self.go2rtc_manager = Go2RTCManager(
            api_url=go2rtc_cfg.get("api_url", "http://127.0.0.1:1984"),
            rtsp_port=go2rtc_cfg.get("rtsp_port", 8555),
            config_path=go2rtc_cfg.get("config_path", "configs/go2rtc.yaml"),
        )

        # 多进程共享资源
        self._manager = mp.Manager()
        self._perf_dict = self._manager.dict()  # 性能指标共享

        # 检测请求队列（所有摄像头 → 检测器）
        self._detect_queue = mp.Queue(maxsize=32)

        # 检测结果队列 — per-camera 隔离，避免 stash 竞争
        self._result_queues: Dict[str, mp.Queue] = {}

        # 事件队列（所有 analyzer → 主进程）
        self._event_queue = mp.Queue(maxsize=1000)

        # 全局停止信号
        self._stop_event = mp.Event()

        # 摄像头进程组
        self._camera_groups: Dict[str, CameraProcessGroup] = {}
        self._lock = threading.Lock()

        # 看门狗
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_running = False
        self._event_dispatcher_thread: Optional[threading.Thread] = None

        cameras = self.config.get("cameras", [])
        self.shared.cameras = cameras

        # 同步 go2rtc
        for cam_cfg in cameras:
            rtsp_url = cam_cfg.get("rtsp_url", "")
            if rtsp_url:
                self.go2rtc_manager.add_stream(cam_cfg["id"], rtsp_url)

        model_config = self.config.get("model", {})
        events_config = self.config.get("events", {})

        # 创建摄像头进程组
        for cam_cfg in cameras:
            self._create_camera_group(cam_cfg, model_config, events_config)

        # 检测器进程
        self._detector_process = DetectorProcess(
            detector_config=model_config,
            detect_queue=self._detect_queue,
            result_queues=self._result_queues,
            stop_event=self._stop_event,
        )

    def _create_camera_group(self, cam_cfg: dict, model_config: dict,
                              events_config: dict):
        """创建单路摄像头的进程组"""
        cam_id = cam_cfg["id"]

        # 每路摄像头独立的停止信号
        cam_stop = mp.Event()

        # per-camera result_queue
        if cam_id not in self._result_queues:
            self._result_queues[cam_id] = mp.Queue(maxsize=32)

        group = CameraProcessGroup(
            cam_config=cam_cfg,
            model_config=model_config,
            events_config=events_config,
            detect_queue=self._detect_queue,
            result_queue=self._result_queues[cam_id],
            event_queue=self._event_queue,
            perf_dict=self._perf_dict,
            stop_event=cam_stop,
            db_path=self.db_path,
        )

        self._camera_groups[cam_id] = group

        # 注册 BGR 共享内存到 SharedState
        self.shared.register_bgr_shm(
            cam_id, group.bgr_shm_name, group.bgr_shm_shape)

    def _persist_config(self):
        self.config["cameras"] = list(self.shared.cameras)
        save_config(self.config, self.config_path)

    def add_camera(self, cam_cfg: dict):
        cam_id = cam_cfg["id"]

        # go2rtc 管理
        rtsp_url = cam_cfg.get("rtsp_url", "")
        if rtsp_url:
            self.go2rtc_manager.add_stream(cam_id, rtsp_url)
            cam_cfg["url"] = self.go2rtc_manager.get_restream_url(cam_id)

        with self._lock:
            # 停止旧进程组
            if cam_id in self._camera_groups:
                self._camera_groups[cam_id].stop()
                del self._camera_groups[cam_id]

            model_config = self.config.get("model", {})
            events_config = self.config.get("events", {})

            self._create_camera_group(cam_cfg, model_config, events_config)

            # 启动新进程组
            self._camera_groups[cam_id].start()

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
            if cam_id in self._camera_groups:
                self._camera_groups[cam_id].stop()
                del self._camera_groups[cam_id]

            # 清理 per-camera result_queue
            self._result_queues.pop(cam_id, None)

            self.go2rtc_manager.remove_stream(cam_id)

            self.shared.cameras = [
                c for c in self.shared.cameras if c["id"] != cam_id
            ]
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
        logger.info(f"启动多进程架构: {len(self._camera_groups)} 路摄像头")

        # 启动检测器进程
        self._detector_process.start()
        logger.info(f"检测器进程已启动 (pid={self._detector_process.pid})")

        # 启动所有摄像头进程组
        for group in self._camera_groups.values():
            group.start()

        # 同步 ES
        for cam_cfg in self.shared.cameras:
            self.es_store.store_camera_config(cam_cfg)

        # 启动事件分发线程
        self._start_event_dispatcher()

        # 启动看门狗
        self._start_watchdog()

        # 启动定期清理
        self._start_cleanup_thread()

        logger.info("所有进程已启动")

    def _start_event_dispatcher(self):
        """
        事件分发线程 — 从 event_queue 读取事件，分发到 SharedState。

        analyzer 进程通过 event_queue 发送:
        - 普通事件 → push_event (WebSocket 推送)
        - _count_update → 更新 count_cache
        - _area_count_update → 更新 area_count_cache
        """
        def _dispatch_loop():
            while not self._stop_event.is_set():
                try:
                    evt = self._event_queue.get(timeout=1)
                except Exception:
                    continue

                if evt is None:
                    continue

                evt_type = evt.get("_type", "")

                if evt_type == "_count_update":
                    self.shared.update_counts(
                        evt["camera_id"], evt["counts"])
                elif evt_type == "_area_count_update":
                    self.shared.update_area_counts(
                        evt["camera_id"], evt["counts"])
                elif evt_type == "_perf_update":
                    self.shared.update_perf(
                        evt["camera_id"], evt["stats"])
                else:
                    # 普通告警事件
                    self.shared.push_event(evt)
                    if self.es_store:
                        self.es_store.store_event(evt)

                # 同步性能指标
                try:
                    for cam_id, stats in dict(self._perf_dict).items():
                        self.shared.update_perf(cam_id, stats)
                except Exception:
                    pass

        self._event_dispatcher_thread = threading.Thread(
            target=_dispatch_loop, daemon=True, name="event-dispatcher")
        self._event_dispatcher_thread.start()

    def _start_watchdog(self):
        """
        看门狗 — 对标 Frigate CameraMaintainer.run() 中的进程监控。

        每 10 秒检查所有子进程，崩溃的自动重启。
        """
        self._watchdog_running = True

        def _watchdog_loop():
            logger.info("看门狗已启动，监控间隔 10s")
            crash_counts: Dict[str, int] = {}

            while self._watchdog_running and not self._stop_event.is_set():
                time.sleep(10)

                # 检查检测器进程
                if not self._detector_process.is_alive():
                    logger.error("检测器进程已崩溃，重启...")
                    try:
                        self._detector_process = DetectorProcess(
                            detector_config=self.config.get("model", {}),
                            detect_queue=self._detect_queue,
                            result_queues=self._result_queues,
                            stop_event=self._stop_event,
                        )
                        self._detector_process.start()
                        logger.info(f"检测器进程已重启 (pid={self._detector_process.pid})")
                    except Exception as e:
                        logger.error(f"检测器重启失败: {e}")

                # 检查摄像头进程组
                with self._lock:
                    for cam_id, group in list(self._camera_groups.items()):
                        if group.stop_event.is_set():
                            continue  # 被主动停止的

                        if group.is_alive():
                            crash_counts[cam_id] = 0
                            continue

                        # 进程崩溃
                        crash_counts[cam_id] = crash_counts.get(cam_id, 0) + 1
                        count = crash_counts[cam_id]
                        backoff = min(2 ** count, 120)

                        logger.warning(
                            f"[看门狗] {cam_id} 进程崩溃 "
                            f"(第{count}次)，{backoff}s 后重启")
                        time.sleep(backoff)

                        try:
                            self._recover_camera(cam_id)
                        except Exception as e:
                            logger.error(f"[看门狗] {cam_id} 恢复失败: {e}")

        self._watchdog_thread = threading.Thread(
            target=_watchdog_loop, daemon=True, name="watchdog")
        self._watchdog_thread.start()

    def _recover_camera(self, cam_id: str):
        """重建并重启崩溃的摄像头进程组"""
        old_group = self._camera_groups.get(cam_id)
        if old_group:
            try:
                old_group.stop()
            except Exception:
                pass

        cam_cfg = self.get_camera_config(cam_id)
        if cam_cfg is None:
            return

        model_config = self.config.get("model", {})
        events_config = self.config.get("events", {})

        self._create_camera_group(cam_cfg, model_config, events_config)
        self._camera_groups[cam_id].start()
        logger.info(f"[看门狗] {cam_id} 已恢复启动")

    def _start_cleanup_thread(self):
        def _cleanup_loop():
            while not self._stop_event.is_set():
                time.sleep(3600)
                try:
                    self.event_db.cleanup_old_events(retain_days=30)
                except Exception as e:
                    logger.error(f"清理过期数据失败: {e}")
        threading.Thread(target=_cleanup_loop, daemon=True).start()

    def reload_model(self, new_model_path: str) -> None:
        """
        模型热重载 — 重启检测器进程。

        与旧版的区别: 旧版替换对象引用，新版重启整个检测器进程。
        """
        model_config = self.config.get("model", {})

        # 更新配置
        if model_config.get("detector_type", "yolo") == "roboflow":
            model_config.setdefault("roboflow", {})["model_id"] = new_model_path
        else:
            model_config["path"] = new_model_path

        # 停止旧检测器进程
        if self._detector_process.is_alive():
            self._detector_process.terminate()
            self._detector_process.join(timeout=10)

        # 启动新检测器进程
        self._detector_process = DetectorProcess(
            detector_config=model_config,
            detect_queue=self._detect_queue,
            result_queues=self._result_queues,
            stop_event=self._stop_event,
        )
        self._detector_process.start()

        self.config["model"] = model_config
        self._persist_config()
        logger.info(f"模型热重载完成: {new_model_path}")

    def stop(self):
        logger.info("停止所有进程...")
        self._watchdog_running = False
        self._stop_event.set()

        # 停止所有摄像头进程组
        for group in self._camera_groups.values():
            group.stop()

        # 停止检测器进程
        if self._detector_process.is_alive():
            self._detector_process.terminate()
            self._detector_process.join(timeout=10)

        # 清理共享内存
        self.shared.cleanup_shm()

        # 清理 Manager
        try:
            self._manager.shutdown()
        except Exception:
            pass

        logger.info("所有进程已停止")
