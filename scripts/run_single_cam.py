"""
单摄像头测试脚本
用于测试 RTSP 拉流是否正常。
"""

import sys
import os
import time
import logging

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import yaml
from src.ingest.rtsp_reader import RTSPReader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    config_path = os.environ.get("CONFIG", "configs/cameras.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam = config["cameras"][0]
    print(f"测试摄像头: {cam['id']} - {cam.get('name', '')}")
    print(f"URL: {cam['url']}")

    reader = RTSPReader(
        url=cam["url"],
        width=cam.get("width", 1280),
        height=cam.get("height", 720),
        fps=cam.get("fps", 15),
    )
    reader.start()

    print("按 'q' 退出...")
    try:
        while True:
            frame, _ = reader.read_latest()
            if frame is not None:
                cv2.imshow(f"Camera: {cam['id']}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
