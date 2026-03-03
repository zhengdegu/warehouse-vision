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
                 fps: int = 15, reconnect_delay: float = 3.0,
                 time_offset: float = None):
        """
        Args:
            time_offset: 摄像头时间与本地时间的偏移量（秒），
                         camera_time = local_time + time_offset。
                         None 表示启动时自动探测，0 表示不修正。
        """
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_delay = reconnect_delay
        self._time_offset: float = time_offset if time_offset is not None else 0.0
        self._auto_detect_offset = time_offset is None

        self._frame_size = 0  # 启动时根据实际分辨率计算
        self._queue: Queue = Queue(maxsize=1)
        self._running = False
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def _build_ffmpeg_cmd(self) -> list:
        """构建 ffmpeg 命令行"""
        cmd = [
            "ffmpeg",
            "-fflags", "+genpts+discardcorrupt+nobuffer",
            "-rtsp_transport", "tcp",
            "-timeout", "10000000",        # RTSP 超时 10s（微秒）
            "-analyzeduration", "10000000", # 探测时间 10s（解决 MPEG4 解析问题）
            "-probesize", "5000000",        # 探测大小 5MB
            "-flags", "+low_delay",
            "-err_detect", "ignore_err",
            "-i", self.url,
        ]
        # 只有指定了分辨率才做缩放
        if self.width > 0 and self.height > 0:
            cmd.extend(["-vf", f"scale={self.width}:{self.height},fps={self.fps}"])
        else:
            cmd.extend(["-vf", f"fps={self.fps}"])
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",              # 不要音频
            "-sn",              # 不要字幕
            "-loglevel", "warning",
            "-"
        ])
        return cmd

    def _drain_stderr(self, proc: subprocess.Popen):
        """读取 stderr，只打印关键错误信息"""
        try:
            for line in proc.stderr:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if not line:
                    continue
                # 过滤掉重复的解码警告，只保留关键错误
                if any(skip in line for skip in [
                    "Last message repeated",
                    "VOL Header truncated", 
                    "PPS id out of range",
                    "Skipping invalid undecodable NALU",
                    "Could not find ref with POC",
                    "deprecated pixel format",
                ]):
                    continue
                logger.warning(f"ffmpeg: {line}")
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
                    # 使用摄像头时间：本地时间 + 偏移量
                    capture_ts = time.time() + self._time_offset

                    # Queue(maxsize=1)：丢弃旧帧，只保留最新帧
                    try:
                        self._queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._queue.put_nowait((frame, capture_ts))
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

    def probe_resolution(self, timeout: float = 15.0) -> tuple:
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

    def probe_time_offset(self, timeout: float = 10.0) -> float:
        """
        探测摄像头时间与本地时间的偏移量。
        通过 ffprobe 获取 RTSP 流的 start_time（基于 NTP/RTCP），
        与本地时间对比得到偏移量。
        返回偏移量（秒），camera_time = local_time + offset。
        探测失败返回 0.0。
        """
        cmd = [
            "ffprobe",
            "-rtsp_transport", "tcp",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            self.url,
        ]
        try:
            before = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            after = time.time()
            probe_local_time = (before + after) / 2  # 取探测中间时刻

            info = json.loads(result.stdout)
            fmt = info.get("format", {})
            # start_time 是流的起始 PTS（秒），对于 RTSP 通常基于 NTP
            start_time_str = fmt.get("start_time")
            if start_time_str:
                stream_time = float(start_time_str)
                # 对于某些摄像头，start_time 是 Unix epoch 秒
                # 如果值很大（>1e9），说明是绝对时间戳
                if stream_time > 1e9:
                    offset = stream_time - probe_local_time
                    logger.info(f"探测到摄像头时间偏移: {offset:+.2f}s "
                                f"(摄像头={stream_time:.0f}, 本地={probe_local_time:.0f})")
                    return offset
            logger.info("未能从流中获取绝对时间戳，使用本地时间（偏移=0）")
        except Exception as e:
            logger.warning(f"时间偏移探测失败: {e}")
        return 0.0

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

        # 跳过时间偏移探测（太慢且经常超时），直接使用配置值或默认 0
        if self._auto_detect_offset:
            # 不再自动探测，直接用 0
            self._time_offset = 0.0
            logger.info(f"使用本地时间（偏移=0）")

        self._frame_size = self.width * self.height * 3
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"RTSPReader 已启动: {self.url} ({self.width}x{self.height})")

    def read_latest(self) -> tuple:
        """获取最新帧，非阻塞。返回 (frame, capture_ts) 或 (None, 0.0)"""
        try:
            return self._queue.get_nowait()
        except Exception:
            return None, 0.0

    def stop(self):
        """停止拉流"""
        self._running = False
        self._kill_process()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"RTSPReader 已停止: {self.url}")
