"""
RTSP 拉流模块 — 严格参考 Frigate 架构

核心设计（对标 Frigate）:
- ffmpeg subprocess 输出 rawvideo yuv420p 到 stdout pipe
- 输入预设: preset-rtsp-restream（从 go2rtc 拉流，参数精简）
- 缩放预设: -r {fps} -vf fps={fps},scale={w}:{h}
- 输出参数: -threads 2 -f rawvideo -pix_fmt yuv420p pipe:
- bufsize = frame_size * 10（Frigate start_or_restart_ffmpeg）
- stderr 通过 LogPipe 线程消费，保留最近 100 行用于调试
- 看门狗: 1秒轮询，20秒无帧重启，进程崩溃重启
- 帧队列 maxsize=1，只保留最新帧

参考文件:
- frigate/frigate/video.py: start_or_restart_ffmpeg, capture_frames, CameraWatchdog
- frigate/frigate/ffmpeg_presets.py: PRESETS_INPUT, PRESETS_HW_ACCEL_SCALE
- frigate/frigate/config/camera/ffmpeg.py: DETECT_FFMPEG_OUTPUT_ARGS_DEFAULT
- frigate/frigate/log.py: LogPipe
"""

import os
import subprocess
import threading
import time
import logging
import json
from collections import deque

import numpy as np
import cv2
from queue import Queue, Full

logger = logging.getLogger(__name__)


class LogPipe(threading.Thread):
    """
    stderr 日志管道 — 参考 Frigate LogPipe。
    将 ffmpeg stderr 通过 pipe 读取，存入 deque(maxlen=100)，
    需要时 dump() 输出最近日志用于调试。
    """

    def __init__(self, name: str, level: int = logging.WARNING):
        super().__init__(daemon=True)
        self._logger = logging.getLogger(f"ffmpeg.{name}")
        self._level = level
        self.deque: deque = deque(maxlen=100)
        self._fd_read, self._fd_write = os.pipe()
        self._reader = os.fdopen(self._fd_read)
        self.start()

    def fileno(self) -> int:
        """返回写端 fd，供 subprocess stderr 使用"""
        return self._fd_write

    def run(self):
        """读取 stderr 所有行，存入 deque"""
        try:
            for line in iter(self._reader.readline, ""):
                cleaned = line.strip()
                if cleaned:
                    self.deque.append(cleaned)
        except Exception:
            pass
        finally:
            try:
                self._reader.close()
            except Exception:
                pass

    def dump(self):
        """输出最近的 stderr 日志（重启前调用）"""
        if self.deque:
            self._logger.log(
                self._level,
                f"最近 {len(self.deque)} 行 ffmpeg 日志:"
            )
            while self.deque:
                self._logger.log(self._level, self.deque.popleft())

    def close(self):
        """关闭写端 fd"""
        try:
            os.close(self._fd_write)
        except Exception:
            pass


def _stop_ffmpeg(process: subprocess.Popen, log: logging.Logger):
    """
    安全终止 ffmpeg 进程 — 参考 Frigate stop_ffmpeg。
    先 terminate，等 30s，超时则 kill。
    """
    if process is None:
        return
    log.info("终止 ffmpeg 进程...")
    try:
        process.terminate()
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg 未退出，强制 kill")
        process.kill()
        process.wait()
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _start_ffmpeg(cmd: list, log: logging.Logger,
                  logpipe: LogPipe, frame_size: int) -> subprocess.Popen:
    """
    启动 ffmpeg 进程 — 参考 Frigate start_or_restart_ffmpeg。
    stdout=PIPE, stderr=logpipe, bufsize=frame_size*10
    """
    log.info(f"启动 ffmpeg: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=logpipe,
        stdin=subprocess.DEVNULL,
        bufsize=frame_size * 10,
    )
    return process


class RTSPReader:
    """
    RTSP 拉流器 — 严格参考 Frigate 架构。

    架构对标:
    - CameraWatchdog.start_ffmpeg_detect() → start()
    - capture_frames() → _capture_loop()
    - CameraWatchdog.run() → _watchdog_loop()
    - CameraWatchdog.reset_capture_thread() → _restart_ffmpeg()

    ffmpeg 命令结构（对标 Frigate）:
    ffmpeg [global_args] [input_args] -i [url] [scale_args] [output_args] pipe:

    从 go2rtc restream 拉流时使用 preset-rtsp-restream:
    -rtsp_transport tcp -timeout 10000000
    """

    # Frigate preset-rtsp-restream 输入参数
    PRESET_RTSP_RESTREAM = [
        "-rtsp_transport", "tcp",
        "-timeout", "10000000",       # 10s RTSP 超时（微秒）
    ]

    # Frigate preset-rtsp-generic 输入参数（直连摄像头时使用）
    PRESET_RTSP_GENERIC = [
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts+discardcorrupt",
        "-rtsp_transport", "tcp",
        "-timeout", "10000000",
        "-use_wallclock_as_timestamps", "1",
    ]

    # Frigate DETECT_FFMPEG_OUTPUT_ARGS_DEFAULT
    DETECT_OUTPUT_ARGS = [
        "-threads", "2",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
    ]

    # 看门狗参数（对标 Frigate CameraWatchdog）
    WATCHDOG_INTERVAL = 1.0       # 1秒轮询
    NO_FRAME_TIMEOUT = 20.0       # 20秒无帧 → 重启
    FPS_OVERFLOW_THRESHOLD = 3    # FPS 溢出3次 → 重启

    def __init__(self, url: str, width: int = 0, height: int = 0,
                 fps: int = 15, reconnect_delay: float = 3.0,
                 time_offset: float = None, use_restream: bool = True):
        """
        Args:
            url: RTSP 流地址（通常是 go2rtc restream 地址）
            width/height: 目标分辨率，0 表示自动探测
            fps: 目标帧率
            reconnect_delay: 重连等待时间（Frigate 的 sleeptime）
            time_offset: 摄像头时间偏移（秒），None=不修正
            use_restream: True=从 go2rtc 拉流（preset-rtsp-restream），
                         False=直连摄像头（preset-rtsp-generic）
        """
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_delay = reconnect_delay
        self._time_offset: float = time_offset if time_offset is not None else 0.0
        self._use_restream = use_restream

        self._frame_size = 0
        self._queue: Queue = Queue(maxsize=1)
        self._running = False

        # Frigate 架构: 独立的 capture 线程 + watchdog 线程
        self._capture_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._logpipe: LogPipe | None = None

        # 看门狗状态（对标 Frigate CameraWatchdog）
        self._last_frame_time: float = 0.0  # 最后收帧时间
        self._camera_fps: float = 0.0       # 当前实际 FPS
        self._fps_overflow_count: int = 0

    def _build_ffmpeg_cmd(self) -> list:
        """
        构建 ffmpeg 命令 — 严格对标 Frigate _get_ffmpeg_cmd。

        结构: ffmpeg [global] [input_args] -i [url] [scale] [output] pipe:

        scale 预设（default）: -r {fps} -vf fps={fps},scale={w}:{h}
        output 预设: -threads 2 -f rawvideo -pix_fmt yuv420p
        """
        # global args
        cmd = ["ffmpeg", "-hide_banner"]

        # input args — 根据是否从 go2rtc restream 拉流选择预设
        if self._use_restream:
            cmd.extend(self.PRESET_RTSP_RESTREAM)
        else:
            cmd.extend(self.PRESET_RTSP_GENERIC)

        # -i url
        cmd.extend(["-i", self.url])

        # scale args — 对标 Frigate PRESETS_HW_ACCEL_SCALE["default"]
        # default: "-r {fps} -vf fps={fps},scale={w}:{h}"
        if self.width > 0 and self.height > 0:
            cmd.extend([
                "-r", str(self.fps),
                "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            ])
        else:
            cmd.extend(["-r", str(self.fps)])

        # output args — 对标 Frigate DETECT_FFMPEG_OUTPUT_ARGS_DEFAULT
        cmd.extend(self.DETECT_OUTPUT_ARGS)

        # pipe: — 对标 Frigate ffmpeg_output_args + ["pipe:"]
        cmd.append("pipe:")

        return cmd

    def _capture_loop(self):
        """
        帧捕获循环 — 严格对标 Frigate capture_frames()。

        从 ffmpeg stdout 读取固定大小的 YUV420P 帧，
        直接传递 YUV 帧（不做颜色转换，参考 Frigate 零转换设计）。
        """
        frame_size = self._frame_size

        try:
            while self._running and self._process:
                # 对标 Frigate: frame_buffer[:] = ffmpeg_process.stdout.read(frame_size)
                raw = self._process.stdout.read(frame_size)

                if len(raw) != frame_size:
                    # 对标 Frigate: "Unable to read frames from ffmpeg process."
                    if not self._running:
                        break

                    if self._process.poll() is not None:
                        logger.error(
                            f"ffmpeg 进程已退出 (pid={self._process.pid})，"
                            f"capture 线程结束"
                        )
                        break

                    logger.warning("ffmpeg 输出不完整，跳过此帧")
                    continue

                # 更新最后收帧时间（看门狗用）
                self._last_frame_time = time.time()

                # 保持 YUV420P 格式 — 对标 Frigate（全程 YUV，不做转换）
                # YUV420P 布局: Y(h*w) + U(h/2*w/2) + V(h/2*w/2) = h*w*3/2
                # reshape 为 (h*3/2, w) 的 2D 数组
                yuv_frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.height * 3 // 2, self.width)
                )

                # 推入队列（丢弃旧帧，只保留最新帧）
                capture_ts = time.time() + self._time_offset
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._queue.put_nowait((yuv_frame, capture_ts))
                except Full:
                    pass

        except Exception as e:
            if self._running:
                logger.error(f"capture 线程异常: {e}")

    def _watchdog_loop(self):
        """
        看门狗循环 — 严格对标 Frigate CameraWatchdog.run()。

        1秒轮询:
        - capture 线程崩溃 → 重启 ffmpeg
        - 20秒无帧 → 重启 ffmpeg
        - 进程退出 → 重启 ffmpeg
        """
        logger.info(f"看门狗已启动: {self.url}")

        # 启动 ffmpeg + capture 线程
        self._start_ffmpeg()
        time.sleep(self.reconnect_delay)

        while self._running:
            time.sleep(self.WATCHDOG_INTERVAL)

            if not self._running:
                break

            # 检查 capture 线程是否存活
            # 对标 Frigate: if not self.capture_thread.is_alive()
            if self._capture_thread is None or not self._capture_thread.is_alive():
                logger.error("capture 线程已退出，重启 ffmpeg")
                self._restart_ffmpeg()
                continue

            # 检查是否超过 20 秒无帧
            # 对标 Frigate: now - self.capture_thread.current_frame.value > 20
            now = time.time()
            if self._last_frame_time > 0 and (now - self._last_frame_time > self.NO_FRAME_TIMEOUT):
                logger.warning(
                    f"{self.NO_FRAME_TIMEOUT}秒无帧，重启 ffmpeg"
                )
                self._restart_ffmpeg()
                continue

            # 检查 ffmpeg 进程是否还在运行
            if self._process is not None and self._process.poll() is not None:
                logger.error(
                    f"ffmpeg 进程已退出 (returncode={self._process.returncode})，"
                    f"重启"
                )
                self._restart_ffmpeg()
                continue

    def _start_ffmpeg(self):
        """
        启动 ffmpeg + capture 线程 — 对标 Frigate start_ffmpeg_detect()。
        """
        cmd = self._build_ffmpeg_cmd()

        # 创建 LogPipe（对标 Frigate LogPipe）
        self._logpipe = LogPipe(name=self.url.split("/")[-1])

        # 启动 ffmpeg（对标 Frigate start_or_restart_ffmpeg）
        self._process = _start_ffmpeg(
            cmd, logger, self._logpipe, self._frame_size
        )

        # 重置看门狗状态
        self._last_frame_time = time.time()  # 给初始宽限
        self._fps_overflow_count = 0

        # 启动 capture 线程（对标 Frigate CameraCaptureRunner）
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True,
            name=f"capture-{self.url.split('/')[-1]}"
        )
        self._capture_thread.start()

    def _restart_ffmpeg(self):
        """
        重启 ffmpeg — 对标 Frigate CameraWatchdog.reset_capture_thread()。

        1. 终止旧 ffmpeg 进程
        2. 等待 capture 线程退出
        3. dump stderr 日志
        4. 重新启动
        """
        # 终止 ffmpeg
        _stop_ffmpeg(self._process, logger)
        self._process = None

        # 等待 capture 线程退出
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)

        # dump 最近的 stderr 日志（对标 Frigate logpipe.dump()）
        if self._logpipe:
            logger.error("重启前 ffmpeg 日志:")
            self._logpipe.dump()
            self._logpipe.close()
            self._logpipe = None

        # 等待后重启
        if self._running:
            logger.info(f"{self.reconnect_delay}s 后重启 ffmpeg...")
            time.sleep(self.reconnect_delay)
            self._start_ffmpeg()

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

    def start(self):
        """启动拉流 — 启动看门狗线程（看门狗内部启动 ffmpeg + capture）"""
        if self._running:
            return

        # 自动探测分辨率
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

        # YUV420P: 1.5 bytes/pixel
        self._frame_size = self.width * self.height * 3 // 2
        self._running = True

        # 启动看门狗线程（对标 Frigate CameraWatchdog 独立线程）
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True,
            name=f"watchdog-{self.url.split('/')[-1]}"
        )
        self._watchdog_thread.start()
        logger.info(f"RTSPReader 已启动: {self.url} ({self.width}x{self.height})")

    def read_latest(self) -> tuple:
        """获取最新帧，非阻塞。返回 (frame, capture_ts) 或 (None, 0.0)"""
        try:
            return self._queue.get_nowait()
        except Exception:
            return None, 0.0

    def stop(self):
        """停止拉流 — 对标 Frigate stop_all_ffmpeg"""
        self._running = False

        # 终止 ffmpeg
        _stop_ffmpeg(self._process, logger)
        self._process = None

        # 等待 capture 线程
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)

        # 等待看门狗线程
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5)

        # 清理 logpipe
        if self._logpipe:
            self._logpipe.close()
            self._logpipe = None

        logger.info(f"RTSPReader 已停止: {self.url}")
