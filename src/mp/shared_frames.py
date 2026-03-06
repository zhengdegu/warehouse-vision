"""
共享内存帧管理器 — 参考 Frigate SharedMemoryFrameManager

核心设计:
- 每路摄像头预分配一块共享内存（YUV420P 帧大小）
- capture 进程写入，analyzer 进程零拷贝读取
- 通过 numpy ndarray(buffer=shm.buf) 实现零拷贝视图

参考: frigate/frigate/util/image.py SharedMemoryFrameManager
"""

import logging
import numpy as np
from multiprocessing import shared_memory
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def frame_shm_name(camera_id: str) -> str:
    """共享内存命名: wv_frame_{camera_id}"""
    return f"wv_frame_{camera_id}"


def bgr_shm_name(camera_id: str) -> str:
    """BGR 帧共享内存命名（用于 Web 展示）"""
    return f"wv_bgr_{camera_id}"


class SharedFrameManager:
    """
    共享内存帧管理器 — 参考 Frigate SharedMemoryFrameManager。

    每路摄像头两块共享内存:
    - YUV 帧: capture 进程写入，analyzer 进程读取（运动检测用 Y 通道）
    - BGR 帧: analyzer 进程写入，Web 进程读取（MJPEG 编码用）
    """

    def __init__(self):
        self._shm_store: dict[str, shared_memory.SharedMemory] = {}

    def create(self, name: str, size: int) -> memoryview:
        """创建共享内存块"""
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            # 已存在则直接打开
            shm = shared_memory.SharedMemory(name=name, create=False)
        self._shm_store[name] = shm
        return shm.buf

    def get(self, name: str, shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """获取共享内存的 numpy 视图（零拷贝）"""
        try:
            if name not in self._shm_store:
                shm = shared_memory.SharedMemory(name=name, create=False)
                self._shm_store[name] = shm
            return np.ndarray(shape, dtype=np.uint8, buffer=self._shm_store[name].buf)
        except FileNotFoundError:
            return None

    def write(self, name: str) -> Optional[memoryview]:
        """获取共享内存的可写 buffer"""
        try:
            if name not in self._shm_store:
                shm = shared_memory.SharedMemory(name=name, create=False)
                self._shm_store[name] = shm
            return self._shm_store[name].buf
        except FileNotFoundError:
            return None

    def close(self, name: str):
        """关闭共享内存（不删除）"""
        if name in self._shm_store:
            self._shm_store[name].close()
            del self._shm_store[name]

    def delete(self, name: str):
        """关闭并删除共享内存"""
        if name in self._shm_store:
            self._shm_store[name].close()
            try:
                self._shm_store[name].unlink()
            except Exception:
                pass
            del self._shm_store[name]
        else:
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def cleanup(self):
        """清理所有共享内存"""
        for name in list(self._shm_store.keys()):
            self.delete(name)
