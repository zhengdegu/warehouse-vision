"""
多进程架构模块 — 参考 Frigate 多进程设计

架构映射:
- Frigate CameraCapture (进程) → CaptureProcess — ffmpeg + 帧写入共享内存
- Frigate CameraTracker (进程) → AnalyzerProcess — 运动检测 + 规则引擎
- Frigate ObjectDetectProcess (进程) → DetectorProcess — 独立推理进程
- Frigate SharedMemoryFrameManager → SharedFrameManager (multiprocessing.shared_memory)
- Frigate InterProcessCommunicator → multiprocessing.Queue / Value

进程间通信:
- 共享内存: 帧数据零拷贝传递（capture → analyzer, analyzer → web）
- Queue: 帧就绪通知、检测请求/结果、事件传递
- Value: 进程状态、时间戳等标量
"""

from .shared_frames import SharedFrameManager, frame_shm_name, bgr_shm_name
from .capture_process import CaptureProcess
from .analyzer_process import AnalyzerProcess
from .detector_process import DetectorProcess
