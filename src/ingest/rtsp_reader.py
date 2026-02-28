"""
RTSP 拉流模块
使用 ffmpeg subprocess 以 TCP 方式拉取 RTSP 流，
输出 rawvideo bgr24，线程 + Queue(maxsize=1) 仅保留最新帧。
支持自动重连。
"""

import subprocess
import threading
import time
import logging
import json
import numpy as np
from queue import Queue, Full

logger = logging.getLogger(__name__)


class RTSPReader:
    """RTSP 拉流器，基于 ffmpeg subprocess"""

    def __init__(self, url: str, width: int = 0, height: int = 0,
                 fps: int = 15, reconnect_delay: float = 3.0):
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_delay = reconnect_delay

        self._frame_size = 0  # 启动时根据实际分辨率计算
        self._queue: Queue = Queue(maxsize=1)
        self._running = False
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def _build_ffmpeg_cmd(self) -> list:
        """构建 ffmpeg 命令行"""
        return [
            "ffmpeg",
            "-fflags", "+genpts+discardcorrupt+nobuffer",
            "-rtsp_transport", "tcp",
            "-timeout", "5000000",         # RTSP 超时 5s（微秒）
            "-analyzeduration", "2000000", # 缩短探测时间 2s
            "-probesize", "1000000",       # 缩小探测大小 1MB
            "-flags", "+low_delay",
            "-err_detect", "ignore_err",
            "-i", self.url,
            "-vf", f"scale={self.width}:{self.height},fps={self.fps}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",              # 不要音频
            "-sn",              # 不要字幕
            "-loglevel", "warning",
            "-"
        ]

    def _drain_stderr(self, proc: subprocess.Popen):
        """读取并丢弃 stderr，防止 pipe buffer 满导致 ffmpeg 阻塞"""
        try:
            for line in proc.stderr:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if line:
                    logger.debug(f"ffmpeg stderr: {line}")
        except Exception:
            pass

    def _read_loop(self):
        """拉流主循环，自动重连"""
        while self._running:
            try:
                cmd = self._build_ffmpeg_cmd()
                logger.info(f"启动 ffmpeg: {' '.join(cmd)}")
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=self._frame_size * 2
                )

                # 启动 stderr 消费线程，防止 buffer 满阻塞
                stderr_thread = threading.Thread(
                    target=self._drain_stderr,
                    args=(self._process,),
                    daemon=True,
                )
                stderr_thread.start()

                while self._running:
                    raw = self._process.stdout.read(self._frame_size)
                    if len(raw) != self._frame_size:
                        logger.warning("ffmpeg 输出不完整，准备重连")
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    )

                    # Queue(maxsize=1)：丢弃旧帧，只保留最新帧
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except Full:
                        pass

            except Exception as e:
                logger.error(f"拉流异常: {e}")
            finally:
                self._kill_process()

            if self._running:
                logger.info(f"{self.reconnect_delay}s 后重连...")
                time.sleep(self.reconnect_delay)

    def _kill_process(self):
        """安全终止 ffmpeg 进程"""
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass
            self._process = None

    def probe_resolution(self, timeout: float = 10.0) -> tuple:
        """
        用 ffprobe 探测 RTSP 流的原始分辨率。
        返回 (width, height)，失败返回 (0, 0)。
        """
        cmd = [
            "ffprobe",
            "-rtsp_transport", "tcp",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            self.url,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            info = json.loads(result.stdout)
            stream = info.get("streams", [{}])[0]
            w = int(stream.get("width", 0))
            h = int(stream.get("height", 0))
            if w > 0 and h > 0:
                logger.info(f"ffprobe 探测到分辨率: {w}x{h}")
                return w, h
        except Exception as e:
            logger.warning(f"ffprobe 探测失败: {e}")
        return 0, 0

    def start(self):
        """启动拉流线程"""
        if self._running:
            return

        # 如果未指定宽高，自动探测
        if self.width <= 0 or self.height <= 0:
            pw, ph = self.probe_resolution()
            if pw > 0 and ph > 0:
                self.width = pw
                self.height = ph
                logger.info(f"自动使用探测分辨率: {self.width}x{self.height}")
            else:
                self.width = self.width or 1280
                self.height = self.height or 720
                logger.warning(f"探测失败，使用默认分辨率: {self.width}x{self.height}")

        self._frame_size = self.width * self.height * 3
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"RTSPReader 已启动: {self.url} ({self.width}x{self.height})")

    def read_latest(self) -> np.ndarray | None:
        """获取最新帧，非阻塞"""
        try:
            return self._queue.get_nowait()
        except Exception:
            return None

    def stop(self):
        """停止拉流"""
        self._running = False
        self._kill_process()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"RTSPReader 已停止: {self.url}")
