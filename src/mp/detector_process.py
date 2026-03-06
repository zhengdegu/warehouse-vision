"""
检测器进程 — 参考 Frigate DetectorRunner / ObjectDetectProcess

独立进程: 从 detect_queue 接收检测请求 → 推理 → 通过 per-camera result_queue 返回结果

通信协议:
- detect_queue (Queue): analyzer → detector
  格式: (req_id, camera_id, shm_name, frame_shape, request_type)
- result_queues (dict): detector → per-camera analyzer
  格式: {camera_id: Queue}，每个 Queue 中: (req_id, camera_id, detections_list)
- stop_event (Event): 主进程 → detector

每个 camera_id 拥有独立的 YOLO 模型实例，隔离 ByteTrack tracker 状态，
避免多摄像头交替推理导致 track_id 混淆。
"""

import logging
import multiprocessing as mp
import time
import numpy as np
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)


class DetectorProcess(mp.Process):
    """
    独立推理进程 — 对标 Frigate DetectorRunner。

    所有摄像头共享一个检测器进程，串行推理避免 GPU 争抢。
    每个 camera_id 拥有独立的 YOLO 模型实例（隔离 ByteTrack 状态）。
    使用 per-camera result_queue 返回结果，消除 stash 竞争。

    Args:
        detector_config: 检测器配置 dict
        detect_queue: 检测请求队列
        result_queues: per-camera 检测结果队列 {camera_id: Queue}
        stop_event: 停止信号
    """

    def __init__(self, detector_config: dict,
                 detect_queue: mp.Queue,
                 result_queues: dict,
                 stop_event: mp.Event):
        super().__init__(daemon=True, name="detector")
        self.detector_config = detector_config
        self.detect_queue = detect_queue
        self.result_queues = result_queues  # {camera_id: mp.Queue}
        self.stop_event = stop_event

    def run(self):
        import sys, os
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [detector] %(levelname)s %(message)s",
        )
        log = logging.getLogger("detector")

        # 每个 camera_id 独立的检测器实例，避免 ByteTrack 状态跨摄像头干扰
        # persist=True 时 YOLO 内部维护 tracker 状态，多摄像头共享会导致 track_id 混淆
        self._detectors: dict = {}   # camera_id -> YOLODetector
        self._pose_detectors: dict = {}  # camera_id -> PoseDetector
        self._detector_log = log
        log.info("检测器进程已启动（per-camera 隔离模式），等待请求...")

        total_inferences = 0
        total_time = 0.0

        while not self.stop_event.is_set():
            try:
                try:
                    request = self.detect_queue.get(timeout=1)
                except Exception:
                    continue
                if request is None:
                    continue

                req_id, camera_id, shm_name, frame_shape, request_type = request

                # 从共享内存读取帧
                try:
                    shm = shared_memory.SharedMemory(name=shm_name, create=False)
                    frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf).copy()
                    shm.close()
                except Exception as e:
                    log.warning(f"读取共享内存失败 {shm_name}: {e}")
                    try:
                        rq = self.result_queues.get(camera_id)
                        if rq:
                            rq.put_nowait((req_id, camera_id, []))
                    except Exception:
                        pass
                    continue

                # 推理 — 按 camera_id 获取/创建独立检测器
                t0 = time.monotonic()
                try:
                    if request_type in ("track", "detect"):
                        det = self._get_detector(camera_id)
                        if det is None:
                            results = []
                        else:
                            dets = det.track(frame) if request_type == "track" else det.detect(frame)
                            results = self._serialize(dets)
                    elif request_type in ("pose_track", "pose_detect"):
                        pdet = self._get_pose_detector(camera_id)
                        if pdet is None:
                            results = []
                        else:
                            dets = pdet.track(frame) if request_type == "pose_track" else pdet.detect(frame)
                            results = self._serialize(dets)
                    else:
                        results = []
                except Exception as e:
                    log.error(f"推理异常: {e}", exc_info=True)
                    results = []

                duration = time.monotonic() - t0
                total_inferences += 1
                total_time += duration

                try:
                    rq = self.result_queues.get(camera_id)
                    if rq:
                        rq.put_nowait((req_id, camera_id, results))
                except Exception:
                    pass

                if total_inferences % 100 == 0:
                    avg = total_time / total_inferences
                    log.info(f"推理统计: {total_inferences} 次, 平均 {avg*1000:.1f}ms")

            except Exception as e:
                log.error(f"检测器循环异常: {e}", exc_info=True)
                time.sleep(0.1)

        log.info(f"检测器进程退出 (共推理 {total_inferences} 次)")

    @staticmethod
    def _serialize(dets) -> list:
        results = []
        for d in dets:
            results.append({
                "track_id": d.track_id, "class_id": d.class_id,
                "class_name": d.class_name, "confidence": d.confidence,
                "bbox": list(d.bbox), "center": d.center, "foot": d.foot,
                "keypoints": d.keypoints.tolist() if d.keypoints is not None else None,
            })
        return results

    def _get_detector(self, camera_id: str):
        """获取 camera_id 对应的独立检测器实例（懒创建）"""
        if camera_id not in self._detectors:
            det = self._create_detector(self._detector_log)
            if det is None:
                return None
            self._detectors[camera_id] = det
            self._detector_log.info(f"为 {camera_id} 创建独立检测器实例")
        return self._detectors[camera_id]

    def _get_pose_detector(self, camera_id: str):
        """获取 camera_id 对应的独立姿态检测器实例（懒创建）"""
        if camera_id not in self._pose_detectors:
            pdet = self._create_pose_detector(self._detector_log)
            self._pose_detectors[camera_id] = pdet
            if pdet:
                self._detector_log.info(f"为 {camera_id} 创建独立姿态检测器实例")
        return self._pose_detectors[camera_id]

    def _create_detector(self, log):
        cfg = self.detector_config
        try:
            if cfg.get("detector_type", "yolo") == "roboflow":
                from src.vision.detector import RoboflowDetector
                rf_cfg = cfg.get("roboflow", {})
                return RoboflowDetector(
                    model_id=rf_cfg.get("model_id", "rfdetr-base"),
                    confidence=cfg.get("confidence", 0.5),
                    allowed_classes=rf_cfg.get("classes") or None,
                )
            else:
                from src.vision.detector import YOLODetector
                return YOLODetector(
                    model_path=cfg.get("path", "yolo26m.pt"),
                    confidence=cfg.get("confidence", 0.5),
                    allowed_classes=cfg.get("classes"),
                    tracker_config=cfg.get("tracker", "configs/bytetrack_sensitive.yaml"),
                )
        except Exception as e:
            log.error(f"创建检测器失败: {e}", exc_info=True)
            return None

    def _create_pose_detector(self, log):
        cfg = self.detector_config
        pose_cfg = cfg.get("pose", {})
        if not pose_cfg.get("enabled", False):
            return None
        try:
            from src.vision.detector import PoseDetector
            return PoseDetector(
                model_path=pose_cfg.get("path", "yolo26m-pose.pt"),
                confidence=pose_cfg.get("confidence", 0.3),
                tracker_config=cfg.get("tracker", "configs/bytetrack_sensitive.yaml"),
            )
        except Exception as e:
            log.error(f"创建姿态检测器失败: {e}", exc_info=True)
            return None


class RemoteDetector:
    """
    远程检测器代理 — 对标 Frigate RemoteObjectDetector。

    在 analyzer 进程中使用，通过 Queue 与 DetectorProcess 通信。
    接口与 YOLODetector/RoboflowDetector 一致（detect/track）。

    使用 per-camera result_queue，无需 stash 机制。
    """

    def __init__(self, camera_id: str,
                 detect_queue: mp.Queue,
                 result_queue: mp.Queue,
                 timeout: float = 5.0):
        self.camera_id = camera_id
        self.detect_queue = detect_queue
        self.result_queue = result_queue
        self.timeout = timeout
        self._req_counter = 0
        self._det_shm_pool = {}

    def _submit(self, frame: np.ndarray, request_type: str) -> list:
        self._req_counter += 1
        req_id = self._req_counter

        # 写入临时共享内存
        shm_name = f"det_{self.camera_id}_{req_id % 4}"
        frame_bytes = frame.tobytes()
        try:
            if shm_name in self._det_shm_pool:
                shm = self._det_shm_pool[shm_name]
                if shm.size >= len(frame_bytes):
                    shm.buf[:len(frame_bytes)] = frame_bytes
                else:
                    shm.close()
                    shm.unlink()
                    raise KeyError
            else:
                raise KeyError
        except KeyError:
            try:
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=len(frame_bytes))
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
            shm.buf[:len(frame_bytes)] = frame_bytes
            self._det_shm_pool[shm_name] = shm

        # 发送请求
        try:
            self.detect_queue.put_nowait(
                (req_id, self.camera_id, shm_name, frame.shape, request_type))
        except Exception:
            return []

        # 等待结果 — per-camera queue，无需 stash
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rid, cam_id, results = self.result_queue.get(
                    timeout=min(remaining, 0.5))
                if rid == req_id and cam_id == self.camera_id:
                    return self._deserialize(results)
                # 极少情况：旧请求的结果，丢弃
            except Exception:
                continue

        return []

    @staticmethod
    def _deserialize(results: list) -> list:
        from src.vision.detector import Detection
        dets = []
        for r in results:
            kp = r.get("keypoints")
            if kp is not None:
                kp = np.array(kp)
            dets.append(Detection(
                track_id=r["track_id"], class_id=r["class_id"],
                class_name=r["class_name"], confidence=r["confidence"],
                bbox=r["bbox"], center=tuple(r["center"]),
                foot=tuple(r["foot"]), keypoints=kp,
            ))
        return dets

    def detect(self, frame: np.ndarray) -> list:
        return self._submit(frame, "detect")

    def track(self, frame: np.ndarray) -> list:
        return self._submit(frame, "track")

    def pose_track(self, frame: np.ndarray) -> list:
        return self._submit(frame, "pose_track")

    def cleanup(self):
        for shm in self._det_shm_pool.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._det_shm_pool.clear()
