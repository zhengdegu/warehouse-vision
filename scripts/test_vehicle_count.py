# -*- coding: utf-8 -*-
"""统计视频中出现的车辆数量 (YOLO + ByteTrack + 多尺度推理)"""
import sys, argparse, time
import cv2, torch
import numpy as np
from ultralytics import YOLO

# COCO 类别: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def nms_boxes(boxes, scores, iou_thresh=0.5):
    """简单 NMS，boxes: Nx4 (x1,y1,x2,y2), scores: N"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def multiscale_detect(model, frame, conf, device, imgsizes):
    """多尺度推理 + NMS 合并，提升小目标召回"""
    all_boxes, all_scores, all_cls = [], [], []
    for sz in imgsizes:
        results = model.predict(frame, conf=conf, device=device,
                                verbose=False, imgsz=sz,
                                classes=list(TARGET_CLASSES.keys()))
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            all_boxes.append(r.boxes.xyxy.cpu().numpy())
            all_scores.append(r.boxes.conf.cpu().numpy())
            all_cls.append(r.boxes.cls.cpu().numpy())

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls = np.concatenate(all_cls)

    # 按类别做 NMS
    keep_all = []
    for c in np.unique(cls):
        mask = cls == c
        idx = np.where(mask)[0]
        kept = nms_boxes(boxes[idx], scores[idx], iou_thresh=0.5)
        keep_all.extend(idx[kept].tolist())

    return boxes[keep_all], scores[keep_all], cls[keep_all]


def main():
    ap = argparse.ArgumentParser(description="统计视频中车辆数量")
    ap.add_argument("source", help="视频文件路径")
    ap.add_argument("--model", default="yolo26m.pt", help="YOLO检测模型")
    ap.add_argument("--conf", type=float, default=0.10, help="置信度阈值")
    ap.add_argument("--skip", type=int, default=1, help="每N帧处理一帧")
    ap.add_argument("--resize", type=int, default=1920, help="主推理尺寸")
    ap.add_argument("--tracker", default="configs/bytetrack_sensitive.yaml",
                    help="Tracker 配置文件")
    ap.add_argument("--multi-scale", action="store_true", default=False,
                    help="启用多尺度推理（慢但召回高）")
    ap.add_argument("--save-frames", type=int, default=30,
                    help="每N帧保存一张标注图片，0=不保存")
    ap.add_argument("--out-dir", default="output_frames", help="标注图片输出目录")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model} | Device: {device} | Conf: {args.conf}")
    print(f"Multi-scale: {args.multi_scale} | Resize: {args.resize}")
    model = YOLO(args.model)
    model.to(device)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"无法打开: {args.source}"); sys.exit(1)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h} @ {fps}fps, {total} frames, {total/fps:.1f}s")
    print("=" * 60)

    # 类别颜色 (BGR)
    CLASS_COLORS = {
        "person": (0, 165, 255),    # 橙色
        "car": (0, 255, 0),         # 绿色
        "motorcycle": (255, 255, 0), # 青色
        "bus": (0, 0, 255),          # 红色
        "truck": (255, 0, 255),      # 紫色
    }

    import os
    if args.save_frames > 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"标注图片输出: {args.out_dir}/ (每{args.save_frames}帧)")

    seen = {name: set() for name in TARGET_CLASSES.values()}
    # 多尺度额外检测计数（无 track_id 的补充）
    ms_extra_count = {name: 0 for name in TARGET_CLASSES.values()}
    fc = 0; pc = 0; t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        if fc % args.skip != 0:
            continue
        pc += 1

        # 主推理 + tracking（保持 track_id 连续性）
        results = model.track(frame, conf=args.conf, persist=True,
                              tracker=args.tracker, device=device,
                              verbose=False, imgsz=args.resize,
                              classes=list(TARGET_CLASSES.keys()))

        tracked_boxes = []
        frame_dets = []  # 保存当前帧检测结果用于画图
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                cid = int(r.boxes.cls[i].item())
                tid = int(r.boxes.id[i].item()) if r.boxes.id is not None else -1
                conf_val = float(r.boxes.conf[i].item())
                if cid in TARGET_CLASSES and tid >= 0:
                    seen[TARGET_CLASSES[cid]].add(tid)
                    bx = r.boxes.xyxy[i].cpu().numpy()
                    tracked_boxes.append(bx)
                    frame_dets.append((bx, TARGET_CLASSES[cid], tid, conf_val))

        # 多尺度补充检测：找 tracker 漏掉的目标
        if args.multi_scale and pc % 5 == 0:  # 每5帧做一次多尺度
            ms_boxes, ms_scores, ms_cls = multiscale_detect(
                model, frame, args.conf, device,
                imgsizes=[1280, 1920, 2560]
            )
            tracked_arr = np.array(tracked_boxes) if tracked_boxes else np.zeros((0, 4))
            for j in range(len(ms_boxes)):
                bx = ms_boxes[j]
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                # 检查是否已被 tracker 覆盖
                matched = False
                if len(tracked_arr) > 0:
                    tcx = (tracked_arr[:, 0] + tracked_arr[:, 2]) / 2
                    tcy = (tracked_arr[:, 1] + tracked_arr[:, 3]) / 2
                    dists = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                    if dists.min() < 60:
                        matched = True
                if not matched:
                    cid = int(ms_cls[j])
                    if cid in TARGET_CLASSES:
                        ms_extra_count[TARGET_CLASSES[cid]] += 1

        # 保存标注图片
        if args.save_frames > 0 and pc % args.save_frames == 0:
            annotated = frame.copy()
            for (bx, cls_name, tid, conf_val) in frame_dets:
                x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} #{tid} {conf_val:.2f}"
                # 背景框
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            # 左上角统计信息
            counts_now = {k: len(v) for k, v in seen.items() if v}
            info = f"Frame {fc}/{total} | {counts_now}"
            cv2.putText(annotated, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            out_path = os.path.join(args.out_dir, f"frame_{fc:06d}.jpg")
            cv2.imwrite(out_path, annotated)

        if pc % 30 == 0:
            spd = pc / (time.time() - t0)
            counts = {k: len(v) for k, v in seen.items() if v}
            print(f"  Frame {fc}/{total} | {spd:.1f}fps | tracked: {counts}")

    cap.release()
    elapsed = time.time() - t0

    print("=" * 60)
    grand = 0
    for name in ["person", "car", "motorcycle", "bus", "truck"]:
        n = len(seen[name])
        if n > 0:
            print(f"  {name:<12s}: {n} (tracked)")
            grand += n
    if args.multi_scale:
        ms_total = sum(ms_extra_count.values())
        if ms_total > 0:
            print(f"  多尺度补充检测: ~{ms_total} (无track_id，可能有重复)")
    print(f"  {'总计':<10s}: {grand}")
    print(f"耗时: {elapsed:.1f}s ({pc/(elapsed or 1):.1f}fps), 处理 {pc}/{fc} 帧")

if __name__ == "__main__":
    main()
