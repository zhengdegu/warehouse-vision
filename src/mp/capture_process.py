"""
帧捕获进程 — 参考 Frigate CameraCapture + CameraWatchdog

独立进程: 运行 ffmpeg → 读取 rawvideo YUV420P → 写入共享内存 → 通知 analyzer

架构对标 Frigate:
- CameraCapture (FrigateProcess) → CaptureProcess (mp.Process)
- CameraWatchdog → 内嵌看门狗逻辑
- CameraCaptureRunner → _capture_loop (线程)
- capture_frames() → _read_frames()
- SharedMemoryFrameManager → 直接操作 shared_memory

帧环形缓冲区（对标 Frigate shm_frame_count）:
- 预分配 N 块共享内存: {camera_id}_frame{0..N-1}
- capture 写入 frame_index % N，通过 Queue 通知 analyzer
- analyzer 处理完后释放（close）

进程间通信:
- frame_queue (Queue): capture → analyzer，传递 (frame_name, frame_time)
- stop_event (Event): 主进程 → capture，停止信号
"""

import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
import json
import numpy as np
from collections import deque
from multiprocessing import shared_memory
from typing import Optional

logger = logging.getLogger(__name__)


class LogPipe(threading.Thread):
    """ffmpeg stderr 日志管道 — 参考 Frigate LogPipe"""

    def __init__(self, name: str):
        super().__init__(daemon=True)
        self.deque: deque = deque(maxlen=100)
        self._fd_read, self._fd_write = os.pipe()
        self._reader = os.fdopen(self._fd_read)
        self._name = name
        self.start()

    def fileno(self) -> int:
        return self._fd_write

    def run(self):
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

    def dump(self) -> list:
        lines = list(self.deque)
        self.deque.clear()
        return lines

    def close(self):
        try:
            os.close(self._fd_write)
        except Exception:
            pass


def _probe_resolution(url: str, timeout: float = 15.0) -> tuple:
    """用 ffprobe 探测分辨率"""
    cmd = [
        "ffprobe", "-rtsp_transport", "tcp", "-v", "quiet",
        "-print_format", "json", "-show_streams",
        "-select_streams", "v:0", url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 0, 0



class CaptureProcess(mp.Process):
    """
    帧捕获进程 — 对标 Frigate CameraCapture。

    每路摄像头一个独立进程:
    1. 启动 ffmpeg 拉流（rawvideo yuv420p）
    2. 读取帧 → 写入共享内存环形缓冲区
    3. 通过 frame_queue 通知 analyzer 进程
    4. 内嵌看门狗: 20秒无帧/进程崩溃 → 自动重启 ffmpeg

    Args:
        camera_id: 摄像头 ID
        url: RTSP 流地址（通常是 go2rtc restream）
        width, height: 目标分辨率（0=自动探测）
        fps: 目标帧率
        shm_frame_count: 共享内存帧缓冲区大小
        frame_queue: 帧就绪通知队列 → analyzer
        stop_event: 停止信号
        use_restream: 是否从 go2rtc 拉流
        time_offset: 时间偏移（秒）
    """

    # ffmpeg 预设参数
    PRESET_RTSP_RESTREAM = ["-rtsp_transport", "tcp", "-timeout", "10000000"]
    PRESET_RTSP_GENERIC = [
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts+discardcorrupt",
        "-rtsp_transport", "tcp", "-timeout", "10000000",
        "-use_wallclock_as_timestamps", "1",
    ]
    DETECT_OUTPUT_ARGS = ["-threads", "2", "-f", "rawvideo", "-pix_fmt", "yuv420p"]

    WATCHDOG_INTERVAL = 1.0
    NO_FRAME_TIMEOUT = 20.0

    def __init__(self, camera_id: str, url: str,
                 width: int, height: int, fps: int,
                 shm_frame_count: int,
                 frame_queue: mp.Queue,
                 stop_event: mp.Event,
                 use_restream: bool = True,
                 time_offset: float = 0.0):
        super().__init__(daemon=True, name=f"capture:{camera_id}")
        self.camera_id = camera_id
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.shm_frame_count = shm_frame_count
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.use_restream = use_restream
        self.time_offset = time_offset

    def run(self):
        """进程入口 — 对标 Frigate CameraCapture.run()"""
        # Windows spawn 模式: 确保项目根目录在 sys.path 中
        import sys, os as _os
        project_root = _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s [{self.camera_id}:capture] %(levelname)s %(message)s",
        )
        log = logging.getLogger(f"capture.{self.camera_id}")

        # 自动探测分辨率
        if self.width <= 0 or self.height <= 0:
            pw, ph = _probe_resolution(self.url)
            if pw > 0 and ph > 0:
                self.width, self.height = pw, ph
                log.info(f"探测到分辨率: {self.width}x{self.height}")
            else:
                self.width = self.width or 1280
                self.height = self.height or 720
                log.warning(f"探测失败，使用默认: {self.width}x{self.height}")

        frame_size = self.width * self.height * 3 // 2  # YUV420P
        frame_shape = (self.height * 3 // 2, self.width)

        # 预创建共享内存环形缓冲区 — 对标 Frigate __start_camera_capture
        shm_blocks = {}
        for i in range(self.shm_frame_count):
            name = f"{self.camera_id}_frame{i}"
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=frame_size)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False)
            shm_blocks[name] = shm

        log.info(
            f"共享内存已创建: {self.shm_frame_count} 块, "
            f"帧大小={frame_size}, 分辨率={self.width}x{self.height}"
        )

        # 看门狗 + ffmpeg 循环
        frame_index = 0
        process: Optional[subprocess.Popen] = None
        logpipe: Optional[LogPipe] = None
        capture_thread: Optional[threading.Thread] = None
        last_frame_time = mp.Value("d", 0.0)

        def _build_cmd():
            cmd = ["ffmpeg", "-hide_banner"]
            cmd.extend(self.PRESET_RTSP_RESTREAM if self.use_restream
                       else self.PRESET_RTSP_GENERIC)
            cmd.extend(["-i", self.url])
            if self.width > 0 and self.height > 0:
                cmd.extend(["-r", str(self.fps),
                            "-vf", f"fps={self.fps},scale={self.width}:{self.height}"])
            else:
                cmd.extend(["-r", str(self.fps)])
            cmd.extend(self.DETECT_OUTPUT_ARGS)
            cmd.append("pipe:")
            return cmd

        def _capture_loop(proc, lft):
            """帧读取线程 — 对标 Frigate capture_frames()"""
            nonlocal frame_index
            try:
                while not self.stop_event.is_set() and proc.poll() is None:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) != frame_size:
                        if not self.stop_event.is_set():
                            if proc.poll() is not None:
                                break
                        continue

                    lft.value = time.time()

                    # 写入共享内存 — 对标 Frigate frame_buffer[:] = ...
                    idx = frame_index % self.shm_frame_count
                    shm_name = f"{self.camera_id}_frame{idx}"
                    shm = shm_blocks[shm_name]
                    np_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
                    np.copyto(np_frame, np.frombuffer(raw, dtype=np.uint8).reshape(frame_shape))

                    # 通知 analyzer — 对标 Frigate frame_queue.put()
                    frame_time = time.time() + self.time_offset
                    try:
                        self.frame_queue.put_nowait((shm_name, frame_time))
                    except Exception:
                        pass  # 队列满则丢弃（analyzer 来不及处理）

                    frame_index += 1
            except Exception as e:
                if not self.stop_event.is_set():
                    log.error(f"capture 线程异常: {e}")

        def _start_ffmpeg():
            nonlocal process, logpipe, capture_thread
            cmd = _build_cmd()
            log.info(f"启动 ffmpeg: {' '.join(cmd)}")
            logpipe = LogPipe(self.camera_id)
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=logpipe,
                stdin=subprocess.DEVNULL, bufsize=frame_size * 10,
            )
            last_frame_time.value = time.time()
            capture_thread = threading.Thread(
                target=_capture_loop, args=(process, last_frame_time),
                daemon=True, name=f"cap-read-{self.camera_id}")
            capture_thread.start()

        def _stop_ffmpeg():
            nonlocal process, logpipe, capture_thread
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except Exception:
                    try:
                        process.kill()
                        process.wait()
                    except Exception:
                        pass
                process = None
            if capture_thread and capture_thread.is_alive():
                capture_thread.join(timeout=5)
            if logpipe:
                lines = logpipe.dump()
                if lines:
                    log.warning(f"ffmpeg 最近日志: {lines[-5:]}")
                logpipe.close()
                logpipe = None

        def _restart_ffmpeg():
            _stop_ffmpeg()
            if not self.stop_event.is_set():
                time.sleep(3)
                _start_ffmpeg()

        # 启动 ffmpeg
        _start_ffmpeg()
        time.sleep(3)  # 初始宽限

        # 看门狗循环 — 对标 Frigate CameraWatchdog.run()
        while not self.stop_event.is_set():
            time.sleep(self.WATCHDOG_INTERVAL)

            if capture_thread is None or not capture_thread.is_alive():
                log.error("capture 线程已退出，重启 ffmpeg")
                _restart_ffmpeg()
                continue

            now = time.time()
            if last_frame_time.value > 0 and (now - last_frame_time.value > self.NO_FRAME_TIMEOUT):
                log.warning(f"{self.NO_FRAME_TIMEOUT}s 无帧，重启 ffmpeg")
                _restart_ffmpeg()
                continue

            if process is not None and process.poll() is not None:
                log.error(f"ffmpeg 已退出 (rc={process.returncode})，重启")
                _restart_ffmpeg()
                continue

        # 清理
        _stop_ffmpeg()
        for shm in shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        log.info("capture 进程退出")
