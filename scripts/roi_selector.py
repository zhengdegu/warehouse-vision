"""
ROI 点选工具
打开 RTSP 流，鼠标点击画多边形，按 's' 保存为 YAML。
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import yaml
import numpy as np
from src.ingest.rtsp_reader import RTSPReader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

points = []
current_frame = None


def mouse_callback(event, x, y, flags, param):
    """鼠标回调：左键添加点，右键删除最后一个点"""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"添加点: ({x}, {y})  共 {len(points)} 个点")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f"删除点: ({removed[0]}, {removed[1]})  剩余 {len(points)} 个点")


def draw_polygon(frame):
    """在帧上绘制当前多边形"""
    img = frame.copy()
    if len(points) > 0:
        for i, pt in enumerate(points):
            cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(img, str(i), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
            # 半透明填充
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    # 操作提示
    cv2.putText(img, "Left click: add point | Right click: undo | 's': save | 'c': clear | 'q': quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Points: {len(points)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def save_roi(output_path: str):
    """保存 ROI 为 YAML"""
    data = {"roi": points}
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"ROI 已保存到: {output_path}")
    print(f"ROI 坐标: {points}")


def main():
    config_path = os.environ.get("CONFIG", "configs/cameras.yaml")
    output_path = os.environ.get("ROI_OUTPUT", "configs/roi_output.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam = config["cameras"][0]
    print(f"打开摄像头: {cam['id']} - {cam.get('name', '')}")

    reader = RTSPReader(
        url=cam["url"],
        width=cam.get("width", 1280),
        height=cam.get("height", 720),
        fps=cam.get("fps", 15),
    )
    reader.start()

    window_name = f"ROI Selector - {cam['id']}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("操作说明:")
    print("  左键点击 - 添加多边形顶点")
    print("  右键点击 - 删除最后一个顶点")
    print("  's' - 保存 ROI")
    print("  'c' - 清除所有点")
    print("  'q' - 退出")

    global points
    try:
        while True:
            frame, _ = reader.read_latest()
            if frame is not None:
                display = draw_polygon(frame)
                cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if len(points) >= 3:
                    save_roi(output_path)
                else:
                    print("至少需要 3 个点才能保存")
            elif key == ord("c"):
                points = []
                print("已清除所有点")

    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
