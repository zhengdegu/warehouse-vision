"""
主启动脚本
启动所有摄像头分析线程 + Web 服务。
"""

import sys
import os
import signal
import logging
import threading

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


def main():
    config_path = os.environ.get("CONFIG", "configs/cameras.yaml")
    logger.info(f"配置文件: {config_path}")

    app = Application(config_path)

    # 优雅退出
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("收到退出信号，正在停止...")
        stop_event.set()
        app.stop()

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
        logger.info("系统已完全停止")


if __name__ == "__main__":
    main()
