"""
推理队列 — 参考 Frigate ObjectDetectProcess 架构改造

核心设计（对比 Frigate）：
┌─────────────────────────────────────────────────────────────────┐
│ Frigate:                                                        │
│   Camera Process ──SharedMemory──▶ DetectorRunner (独立进程)    │
│                  ◀──ZMQ pub/sub──                               │
│                                                                 │
│ 本项目（线程版等价实现）：                                       │
│   Camera Thread ──slot写入──▶ Worker Thread (串行推理)          │
│                 ◀──Event通知──                                  │
└─────────────────────────────────────────────────────────────────┘

优化策略：
1. 每摄像头独立 slot（类似 Frigate 的 per-camera SharedMemory）
   - 新帧直接覆盖旧帧，天然去重
   - worker 轮询 slot 而非从队列取，避免队列堆积
2. 过期帧跳过（类似 Frigate 的 detection_start 超时检测）
3. 姿态降频（pose 每 N 帧推理一次）
4. 公平调度：Round-Robin 轮询各摄像头，避免单路饥饿
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class CameraSlot:
    """
    每摄像头的推理槽位 — 类似 Frigate 的 per-camera SharedMemory。

    摄像头线程写入 frame，worker 线程读取并推理。
    新帧覆盖旧帧，天然实现"只推理最新帧"。
    """
    camera_id: str

    # 输入：摄像头线程写入
    frame: Optional[np.ndarray] = None
    request_type: str = "track"
    submit_time: float = 0.0
    pending: bool = False  # 是否有待处理的请求

    # 输出：worker 线程写入
    result: List[Detection] = field(default_factory=list)
    error: Optional[Exception] = None
    done: threading.Event = field(default_factory=threading.Event)

    # 锁：保护 frame/pending 的读写
    lock: threading.Lock = field(default_factory=threading.Lock)

    # 姿态降频计数
    pose_count: int = 0


class InferenceQueue:
    """
    推理队列 — Frigate 风格的 slot-based 架构。

    与旧版 Queue-based 的区别：
    - 旧版：camera → Queue.put(request) → worker Queue.get() → Event.wait()
      问题：多路请求堆积，FIFO 导致旧帧也被推理，超时频繁
    - 新版：camera → slot.frame = frame → worker 轮询 slot → slot.done.set()
      优势：每路只保留最新帧，worker 永远推理最新数据

    参数：
        detector: YOLO 检测器实例
        pose_detector: 姿态检测器实例（可选）
        stale_threshold: 过期阈值（秒），超过此时间的请求跳过
        pose_every: 姿态检测降频，每 N 帧推理一次
    """

    def __init__(self, detector, pose_detector=None,
                 max_queue_size: int = 16,  # 保留参数兼容，不再使用
                 stale_threshold: float = 2.0,
                 pose_every: int = 3):
        self.detector = detector
        self.pose_detector = pose_detector
        self._stale_threshold = stale_threshold
        self._pose_every = pose_every

        self._slots: Dict[str, CameraSlot] = {}
        self._slot_order: List[str] = []  # Round-Robin 顺序
        self._slots_lock = threading.Lock()

        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._has_work = threading.Event()  # 通知 worker 有新请求

        # 统计
        self._total_inferences = 0
        self._total_skipped = 0
        self._total_stale = 0
        self._total_wait_time = 0.0

    def _ensure_slot(self, camera_id: str, request_type: str = "track") -> CameraSlot:
        """确保 (camera_id, request_type) 有对应的 slot（懒初始化）"""
        key = f"{camera_id}:{request_type}"
        with self._slots_lock:
            if key not in self._slots:
                slot = CameraSlot(camera_id=camera_id)
                self._slots[key] = slot
                self._slot_order.append(key)
                logger.info(f"[{camera_id}] 推理 slot 已创建 ({request_type})")
            return self._slots[key]

    def start(self):
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="inference-worker")
        self._worker.start()
        logger.info(
            f"推理队列已启动 (slot-based, "
            f"stale={self._stale_threshold}s, pose_every={self._pose_every})")

    def stop(self):
        self._running = False
        self._has_work.set()  # 唤醒 worker 以退出
        if self._worker:
            self._worker.join(timeout=5)
        logger.info(
            f"推理队列已停止 "
            f"(推理={self._total_inferences}, "
            f"跳过={self._total_skipped}, "
            f"过期={self._total_stale})")

    def _worker_loop(self):
        """
        推理工作线程 — Round-Robin 轮询各摄像头 slot。

        类似 Frigate 的 DetectorRunner.run()：
        - Frigate: detection_queue.get(connection_id) → shm 读帧 → 推理 → shm 写结果
        - 本实现: 轮询 slot.pending → 读 slot.frame → 推理 → 写 slot.result
        """
        rr_index = 0  # Round-Robin 索引

        while self._running:
            # 等待有工作可做
            self._has_work.wait(timeout=0.1)
            self._has_work.clear()

            if not self._running:
                break

            # Round-Robin 遍历所有 slot
            with self._slots_lock:
                slot_ids = list(self._slot_order)

            if not slot_ids:
                continue

            processed_any = False
            for _ in range(len(slot_ids)):
                rr_index = rr_index % len(slot_ids)
                slot_key = slot_ids[rr_index]
                rr_index += 1

                slot = self._slots.get(slot_key)
                if slot is None:
                    continue
                cam_id = slot.camera_id

                # 检查是否有待处理请求
                with slot.lock:
                    if not slot.pending:
                        continue
                    # 取出请求数据（类似 Frigate 从 SharedMemory 读帧）
                    frame = slot.frame
                    request_type = slot.request_type
                    submit_time = slot.submit_time
                    slot.pending = False
                    slot.frame = None  # 释放引用，允许 GC

                # 检查是否过期
                age = time.monotonic() - submit_time
                if age > self._stale_threshold:
                    logger.debug(
                        f"[{cam_id}] 丢弃过期请求 {request_type} "
                        f"(等待 {age:.1f}s > {self._stale_threshold}s)")
                    self._total_stale += 1
                    # 通知等待方：结果为空
                    slot.result = []
                    slot.error = None
                    slot.done.set()
                    processed_any = True
                    continue

                if frame is None:
                    slot.result = []
                    slot.error = None
                    slot.done.set()
                    continue

                # 执行推理（类似 Frigate 的 object_detector.detect_raw）
                t0 = time.monotonic()
                try:
                    if request_type == "track":
                        slot.result = self.detector.track(frame)
                    elif request_type == "detect":
                        slot.result = self.detector.detect(frame)
                    elif request_type == "pose_track":
                        if self.pose_detector:
                            slot.result = self.pose_detector.track(frame)
                        else:
                            slot.result = []
                    elif request_type == "pose_detect":
                        if self.pose_detector:
                            slot.result = self.pose_detector.detect(frame)
                        else:
                            slot.result = []
                    slot.error = None
                except Exception as e:
                    slot.result = []
                    slot.error = e
                    logger.error(f"[{cam_id}] 推理异常: {e}")
                finally:
                    duration = time.monotonic() - t0
                    self._total_inferences += 1
                    self._total_wait_time += duration
                    if duration > 1.0:
                        logger.info(
                            f"[{cam_id}] {request_type} 推理耗时 {duration:.1f}s")
                    slot.done.set()
                    processed_any = True

                # 每处理一个就检查其他 slot 是否有更紧急的请求
                # （Round-Robin 公平性）
                break

            # 如果本轮处理了请求，立即检查是否还有更多
            if processed_any:
                self._has_work.set()

    def submit(self, frame: np.ndarray, camera_id: str,
               request_type: str = "track",
               timeout: float = 5.0) -> List[Detection]:
        """
        提交推理请求并等待结果。

        与旧版的区别：
        - 旧版：创建 InferenceRequest → Queue.put → Event.wait
        - 新版：写入 slot.frame → 通知 worker → slot.done.wait

        新帧会覆盖旧帧（如果旧帧还没被处理），天然去重。

        姿态降频：pose 请求每 pose_every 帧才真正推理一次。
        """
        # 姿态降频
        if request_type in ("pose_track", "pose_detect"):
            slot = self._ensure_slot(camera_id, request_type)
            slot.pose_count += 1
            if slot.pose_count % self._pose_every != 0:
                return []

        slot = self._ensure_slot(camera_id, request_type)

        # 重置 done event，准备新一轮等待
        slot.done.clear()

        # 写入 slot（覆盖旧帧）
        with slot.lock:
            slot.frame = frame
            slot.request_type = request_type
            slot.submit_time = time.monotonic()
            slot.pending = True
            slot.result = []
            slot.error = None

        # 通知 worker 有新请求
        self._has_work.set()

        # 等待结果
        if slot.done.wait(timeout=timeout):
            if slot.error:
                raise slot.error
            return slot.result
        else:
            logger.warning(f"[{camera_id}] 推理超时 ({timeout}s)")
            return []
