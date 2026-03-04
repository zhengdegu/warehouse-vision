"""
主启动脚本
启动 go2rtc 流中转 + 所有摄像头分析线程 + Web 服务。
"""

import sys
import os
import signal
import logging
import threading
import subprocess
import time

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 限制 PyTorch CPU 线程数，避免多摄像头线程争抢导致 CPU 过载
import torch
_cpu_count = os.cpu_count() or 4
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, _cpu_count // 2))
    torch.set_num_interop_threads(max(1, _cpu_count // 4))

from src.app import Application
from src.web.server import run_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def start_go2rtc(config_path: str = "configs/go2rtc.yaml") -> subprocess.Popen | None:
    """
    启动 go2rtc 流中转服务。
    go2rtc 负责将所有 RTSP 源规范化（特别是 mpeg4 转码为 h264），
    warehouse-vision 从 go2rtc 的 restream 拉流。
    """
    # 查找 go2rtc 可执行文件
    if sys.platform == "win32":
        candidates = ["go2rtc/go2rtc.exe", "go2rtc.exe"]
    else:
        candidates = ["/usr/local/bin/go2rtc", "go2rtc/go2rtc", "go2rtc"]

    go2rtc_bin = None
    for c in candidates:
        if os.path.isfile(c):
            go2rtc_bin = c
            break

    if not go2rtc_bin:
        logger.warning("未找到 go2rtc，跳过流中转服务（mpeg4 等问题流可能无法工作）")
        return None

    if not os.path.isfile(config_path):
        logger.warning(f"go2rtc 配置文件不存在: {config_path}，跳过")
        return None

    logger.info(f"启动 go2rtc: {go2rtc_bin} -config {config_path}")
    proc = subprocess.Popen(
        [go2rtc_bin, "-config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # 启动日志消费线程
    def _drain_output():
        try:
            for line in proc.stdout:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.rstrip()
                if line:
                    logger.info(f"[go2rtc] {line}")
        except Exception:
            pass

    threading.Thread(target=_drain_output, daemon=True).start()

    # 等待 go2rtc 启动
    time.sleep(2)
    if proc.poll() is not None:
        logger.error(f"go2rtc 启动失败，退出码: {proc.returncode}")
        return None

    logger.info("go2rtc 已启动")
    return proc


def stop_go2rtc(proc: subprocess.Popen | None):
    """停止 go2rtc 进程"""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    logger.info("go2rtc 已停止")


def main():
    config_path = os.environ.get("CONFIG", "configs/cameras.yaml")
    logger.info(f"配置文件: {config_path}")

    # 启动 go2rtc 流中转（mpeg4 转码、减少摄像头连接数）
    go2rtc_config = os.environ.get("GO2RTC_CONFIG", "configs/go2rtc.yaml")
    go2rtc_proc = start_go2rtc(go2rtc_config)

    app = Application(config_path)

    # 优雅退出
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("收到退出信号，正在停止...")
        stop_event.set()
        app.stop()
        stop_go2rtc(go2rtc_proc)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动分析线程
    app.start()

    # 启动 Web 服务（阻塞主线程）
    web_config = app.config.get("web", {})
    host = web_config.get("host", "0.0.0.0")
    port = web_config.get("port", 8000)

    logger.info(f"Web 服务启动: http://{host}:{port}")
    try:
        run_server(app.shared, app.jsonl_logger, host=host, port=port,
                   application=app)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        stop_go2rtc(go2rtc_proc)
        logger.info("系统已完全停止")


if __name__ == "__main__":
    main()
