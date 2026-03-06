"""
Web 服务模块 — 参考 Frigate 的 FastAPI 架构
RESTful API + WebSocket + MJPEG 实时视频流 + 系统监控
"""

import asyncio
import json
import os
import time
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.training.api import configure as configure_training_api, router as training_router
from src.training.sample_manager import SampleManager
from src.training.annotation_engine import AnnotationEngine
from src.training.training_engine import TrainingEngine
from src.training.model_registry import ModelRegistry
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

_shared_state = None
_jsonl_logger = None
_application = None
_event_db = None

app = FastAPI(title="仓储安防视频分析系统", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(training_router)


@app.on_event("startup")
async def _capture_event_loop():
    global _event_loop
    _event_loop = asyncio.get_running_loop()

EVENTS_DIR = Path(__file__).parent.parent.parent / "events"
SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples"
FRONTEND_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

os.makedirs(str(EVENTS_DIR), exist_ok=True)
os.makedirs(str(SAMPLES_DIR), exist_ok=True)
app.mount("/events", StaticFiles(directory=str(EVENTS_DIR)), name="events")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

_ws_clients: list = []
_event_loop = None  # asyncio event loop reference for cross-thread WS broadcast


# ========== Pydantic 模型 ==========

class TripwireConfigBody(BaseModel):
    id: str = ""
    name: str = ""
    p1: List[float] = [0, 0]
    p2: List[float] = [0, 0]
    direction: str = "left_to_right"
    cooldown: float = 2.0

class IntrusionRuleBody(BaseModel):
    enabled: bool = False
    confirm_frames: int = 5
    cooldown: float = 30

class TripwireRuleBody(BaseModel):
    enabled: bool = False

class CountingRuleBody(BaseModel):
    enabled: bool = False
    window_seconds: int = 60

class DwellBody(BaseModel):
    enabled: bool = False
    max_seconds: float = 120
    confirm_frames: int = 5
    cooldown: float = 60

class CrowdBody(BaseModel):
    enabled: bool = False
    max_count: int = 5
    radius: float = 200
    confirm_frames: int = 5
    cooldown: float = 60

class ProximityBody(BaseModel):
    enabled: bool = False
    min_distance: float = 50
    confirm_frames: int = 3
    cooldown: float = 30

class FightBody(BaseModel):
    enabled: bool = False
    proximity_radius: float = 150
    min_speed: float = 60
    min_persons: int = 2
    confirm_frames: int = 3
    cooldown: float = 30

class FallBody(BaseModel):
    enabled: bool = False
    ratio_threshold: float = 1.0
    min_ratio_change: float = 0.5
    min_y_drop: float = 20
    confirm_frames: int = 2
    cooldown: float = 30

class TimePeriodBody(BaseModel):
    enabled: bool = False
    start: str = "00:00"
    end: str = "23:59"
    days: List[int] = [0, 1, 2, 3, 4, 5, 6]

class AnomalyRuleBody(BaseModel):
    dwell: DwellBody = DwellBody()
    crowd: CrowdBody = CrowdBody()
    proximity: ProximityBody = ProximityBody()
    fight: FightBody = FightBody()
    fall: FallBody = FallBody()
    time_period: TimePeriodBody = TimePeriodBody()

class MotionConfigBody(BaseModel):
    enabled: bool = True
    threshold: int = 40
    contour_area: int = 200
    frame_alpha: float = 0.02

class RulesBody(BaseModel):
    intrusion: IntrusionRuleBody = IntrusionRuleBody()
    tripwire: TripwireRuleBody = TripwireRuleBody()
    counting: CountingRuleBody = CountingRuleBody()
    anomaly: AnomalyRuleBody = AnomalyRuleBody()
    alert_types: List[str] = Field(default_factory=list)

class CameraConfigBody(BaseModel):
    id: str = ""
    name: str = ""
    url: str = ""
    rtsp_url: str = ""  # 原始 RTSP 地址（自动注册到 go2rtc）
    width: int = 0
    height: int = 0
    fps: int = 15
    roi: List[List[float]] = []
    tripwires: List[TripwireConfigBody] = []
    rules: RulesBody = RulesBody()
    motion: MotionConfigBody = MotionConfigBody()


# ========== 页面 ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return HTMLResponse("<h1>前端未构建，请先运行 npm run build</h1>", status_code=404)


# ========== 系统状态 API（Frigate 风格）==========

@app.get("/api/system/stats")
async def system_stats():
    """系统性能统计 — 参考 Frigate /api/stats"""
    result = {
        "cameras": {},
        "uptime": time.time(),
    }
    if _shared_state:
        result["cameras"] = _shared_state.get_perf()
    if _application:
        result["camera_count"] = len(_application._camera_groups)
    return JSONResponse(result)


@app.get("/api/system/health")
async def system_health():
    """健康检查端点"""
    healthy = True
    cameras_status = {}
    if _application:
        for cam_id, group in _application._camera_groups.items():
            cap_alive = group.capture.is_alive()
            ana_alive = group.analyzer.is_alive()
            cameras_status[cam_id] = {
                "running": not group.stop_event.is_set(),
                "alive": cap_alive and ana_alive,
                "capture_alive": cap_alive,
                "analyzer_alive": ana_alive,
                "capture_pid": group.capture.pid,
                "analyzer_pid": group.analyzer.pid,
            }
            if not (cap_alive and ana_alive):
                healthy = False
    return JSONResponse({
        "status": "healthy" if healthy else "degraded",
        "cameras": cameras_status,
    })


@app.get("/api/system/config")
async def get_system_config():
    """获取系统级配置（model, system, events 等）"""
    if _application is None:
        return JSONResponse({"error": "系统未初始化"}, status_code=500)
    cfg = _application.config
    return JSONResponse({
        "model": cfg.get("model", {}),
        "system": cfg.get("system", {}),
        "events": cfg.get("events", {}),
        "elasticsearch": cfg.get("elasticsearch", {}),
    })


class SystemConfigUpdateBody(BaseModel):
    model: Optional[Dict[str, Any]] = None
    system: Optional[Dict[str, Any]] = None
    events: Optional[Dict[str, Any]] = None
    elasticsearch: Optional[Dict[str, Any]] = None


@app.put("/api/system/config")
async def update_system_config(body: SystemConfigUpdateBody):
    """更新系统级配置（model, system, events 等），模型路径变更时自动热重载"""
    if _application is None:
        return JSONResponse({"error": "系统未初始化"}, status_code=500)

    cfg = _application.config
    old_model_path = cfg.get("model", {}).get("path")
    changed_sections = []

    data = body.model_dump(exclude_none=True)
    for section, values in data.items():
        if section not in cfg:
            cfg[section] = {}
        cfg[section].update(values)
        changed_sections.append(section)

    # 持久化
    from src.app import save_config
    save_config(cfg, _application.config_path)

    # 模型路径变更 → 热重载
    new_model_path = cfg.get("model", {}).get("path")
    model_reloaded = False
    if "model" in changed_sections and new_model_path and new_model_path != old_model_path:
        try:
            _application.reload_model(new_model_path)
            model_reloaded = True
        except Exception as e:
            logger.error(f"模型热重载失败: {e}", exc_info=True)
            return JSONResponse({"ok": False, "error": f"配置已保存，但模型重载失败: {e}"}, status_code=500)

    return JSONResponse({
        "ok": True,
        "message": "配置已更新" + ("，模型已重载" if model_reloaded else ""),
        "changed": changed_sections,
    })

# ========== 摄像头 API ==========

@app.get("/api/cameras")
async def get_cameras():
    if _application is None:
        return JSONResponse([])
    cameras = []
    for cam in _application.get_all_camera_configs():
        cameras.append({
            "id": cam["id"],
            "name": cam.get("name", cam["id"]),
            "timezone": cam.get("timezone", ""),
        })
    return JSONResponse(cameras)


@app.get("/api/cameras/{camera_id}")
async def get_camera_detail(camera_id: str):
    if _application is None:
        return JSONResponse({"error": "系统未初始化"}, status_code=500)
    cfg = _application.get_camera_config(camera_id)
    if cfg is None:
        return JSONResponse({"error": "摄像头不存在"}, status_code=404)
    return JSONResponse(cfg)


@app.post("/api/cameras")
async def add_camera(body: CameraConfigBody):
    if _application is None:
        return JSONResponse({"error": "系统未初始化"}, status_code=500)
    if not body.id:
        return JSONResponse({"error": "id 为必填项"}, status_code=400)
    if not body.rtsp_url and not body.url:
        return JSONResponse({"error": "rtsp_url 或 url 至少填一个"}, status_code=400)
    existing = _application.get_camera_config(body.id)
    if existing:
        return JSONResponse({"error": f"摄像头 {body.id} 已存在"}, status_code=409)
    cam_cfg = body.model_dump()
    cam_cfg["tripwires"] = [tw if isinstance(tw, dict) else tw
                            for tw in cam_cfg.get("tripwires", [])]
    cam_cfg["rules"] = cam_cfg.get("rules", {})
    try:
        _application.add_camera(cam_cfg)
        return JSONResponse({"ok": True, "message": f"摄像头 {body.id} 已添加"})
    except Exception as e:
        logger.error(f"添加摄像头失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.put("/api/cameras/{camera_id}")
async def update_camera(camera_id: str, body: CameraConfigBody):
    if _application is None:
        return JSONResponse({"error": "系统未初始化"}, status_code=500)
    if not body.rtsp_url and not body.url:
        return JSONResponse({"error": "rtsp_url 或 url 至少填一个"}, status_code=400)
    cam_cfg = body.model_dump()
    cam_cfg["id"] = camera_id
    try:
        _application.update_camera(camera_id, cam_cfg)
        return JSONResponse({"ok": True, "message": f"摄像头 {camera_id} 已更新"})
    except Exception as e:
        logger.error(f"更新摄像头失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    if _application is None:
        return JSONResponse({"error": "系统未初始化"}, status_code=500)
    existing = _application.get_camera_config(camera_id)
    if not existing:
        return JSONResponse({"error": "摄像头不存在"}, status_code=404)
    try:
        _application.remove_camera(camera_id)
        return JSONResponse({"ok": True, "message": f"摄像头 {camera_id} 已删除"})
    except Exception as e:
        logger.error(f"删除摄像头失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ========== 事件 API（增强版，支持分页和过滤）==========

@app.get("/api/events")
async def get_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    camera_id: str = Query(None),
    event_type: str = Query(None),
    start_time: float = Query(None),
    end_time: float = Query(None),
):
    """事件查询 — 支持分页、按摄像头/类型/时间过滤"""
    if _event_db:
        events = _event_db.query_events(
            camera_id=camera_id,
            event_type=event_type,
            limit=limit,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
        )
        total = _event_db.count_events(
            camera_id=camera_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
        )
        return JSONResponse({
            "items": events,
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    # 回退到 JSONL
    if _jsonl_logger is None:
        return JSONResponse({"items": [], "total": 0})
    events = _jsonl_logger.read_events(limit, camera_id=camera_id)
    return JSONResponse({"items": events, "total": len(events)})


@app.get("/api/events/summary")
async def event_summary(
    camera_id: str = Query(None),
    hours: int = Query(24, ge=1, le=720),
):
    """事件统计摘要"""
    if _event_db:
        raw = _event_db.get_event_summary(camera_id=camera_id, hours=hours)
        # Aggregate by base type for frontend
        by_type: Dict[str, int] = {}
        total = 0
        for key, cnt in raw.items():
            base = key.split("/")[0]
            by_type[base] = by_type.get(base, 0) + cnt
            total += cnt
        return JSONResponse({"total": total, "by_type": by_type})
    return JSONResponse({"total": 0, "by_type": {}})


@app.get("/api/counts")
async def get_counts():
    """获取计数统计：越线统计（今日进出）+ 区域内实时人车数"""
    if _shared_state is None:
        return JSONResponse({})
    
    # 越线计数（包含 today_in, today_out）
    counts = _shared_state.get_counts()
    
    # 区域内实时计数
    area_counts = _shared_state.get_area_counts()
    
    # 合并数据
    for cam_id, area in area_counts.items():
        if cam_id not in counts:
            counts[cam_id] = {}
        counts[cam_id]["area"] = area  # 区域内实时人车数
    
    # 从数据库补充告警数
    if _event_db:
        all_stats = _event_db.get_camera_stats()
        for cam_id, stats in all_stats.items():
            if cam_id not in counts:
                counts[cam_id] = {}
            counts[cam_id]["alert_count"] = stats.get("alert_count", 0)
            if "total_in" not in counts[cam_id]:
                counts[cam_id]["total_in"] = stats.get("total_in", 0)
                counts[cam_id]["total_out"] = stats.get("total_out", 0)
    elif _jsonl_logger:
        all_stats = _jsonl_logger.get_camera_stats()
        for cam_id, stats in all_stats.items():
            if cam_id not in counts:
                counts[cam_id] = {}
            counts[cam_id]["alert_count"] = stats.get("alert_count", 0)
            if "total_in" not in counts[cam_id]:
                counts[cam_id]["total_in"] = stats.get("total_in", 0)
                counts[cam_id]["total_out"] = stats.get("total_out", 0)
    return JSONResponse(counts)


# ========== 准确率统计 API ==========

@app.get("/api/accuracy/stats")
async def accuracy_stats(
    camera_id: str = Query(None),
    hours: int = Query(24, ge=1, le=720),
):
    """获取各规则类型的报警统计（按摄像头、类型聚合）"""
    if not _event_db:
        return JSONResponse({"items": [], "summary": {}})
    stats = _event_db.compute_accuracy_stats(camera_id=camera_id, hours=hours)

    # 汇总
    summary: Dict[str, Any] = {}
    for s in stats:
        key = s["rule_type"]
        if s["sub_type"]:
            key += f"/{s['sub_type']}"
        if key not in summary:
            summary[key] = {"total": 0, "avg_confidence": 0, "cameras": 0}
        summary[key]["total"] += s["total_alerts"]
        summary[key]["cameras"] += 1
        summary[key]["avg_confidence"] = round(
            (summary[key]["avg_confidence"] * (summary[key]["cameras"] - 1) + s["avg_confidence"])
            / summary[key]["cameras"], 3
        )

    return JSONResponse({"items": stats, "summary": summary})


@app.get("/api/accuracy/trend")
async def accuracy_trend(
    camera_id: str = Query(None),
    hours: int = Query(24, ge=1, le=720),
    bucket_minutes: int = Query(60, ge=5, le=1440),
):
    """获取报警趋势数据"""
    if not _event_db:
        return JSONResponse({"items": []})
    trend = _event_db.get_alert_trend(
        camera_id=camera_id, hours=hours, bucket_minutes=bucket_minutes
    )
    return JSONResponse({"items": trend})


@app.get("/api/accuracy/records")
async def accuracy_records(
    camera_id: str = Query(None),
    rule_type: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """查询已保存的准确率评估记录"""
    if not _event_db:
        return JSONResponse({"items": []})
    records = _event_db.get_accuracy_records(
        camera_id=camera_id, rule_type=rule_type, limit=limit
    )
    return JSONResponse({"items": records})


class AccuracyRecordBody(BaseModel):
    camera_id: str = ""
    rule_type: str
    sub_type: str = ""
    total_alerts: int = 0
    confirmed: int = 0
    false_positive: int = 0
    missed: int = 0
    period_start: float = 0
    period_end: float = 0


@app.post("/api/accuracy/records")
async def save_accuracy_record(body: AccuracyRecordBody):
    """保存一条准确率评估记录（人工标注结果）"""
    if not _event_db:
        return JSONResponse({"error": "数据库未初始化"}, status_code=500)
    data = body.model_dump()
    # 计算 precision / recall / f1
    tp = data["confirmed"]
    fp = data["false_positive"]
    fn = data["missed"]
    data["precision"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    data["recall"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    p, r = data["precision"], data["recall"]
    data["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0
    rid = _event_db.save_accuracy_record(data)
    return JSONResponse({"ok": True, "id": rid})


# ========== 视频流 ==========

async def _mjpeg_generator(camera_id: str):
    while True:
        if _shared_state is None:
            await asyncio.sleep(0.5)
            continue
        frame = _shared_state.get_frame(camera_id)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Waiting: {camera_id}", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, jpeg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")
        await asyncio.sleep(0.1)  # ~10fps MJPEG，降低编码开销


@app.get("/stream/{camera_id}")
async def video_stream(camera_id: str):
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/snapshot/{camera_id}")
async def snapshot(camera_id: str):
    frame = None
    if _shared_state:
        frame = _shared_state.get_frame(camera_id)
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"No signal: {camera_id}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg"
    )


# ========== WebSocket ==========

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"WebSocket 客户端连接，当前 {len(_ws_clients)} 个")
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
            if _shared_state:
                while not _shared_state.event_queue.empty():
                    try:
                        event = _shared_state.event_queue.get_nowait()
                        msg = json.dumps(event, ensure_ascii=False)
                        # 异步并发广播，避免单个慢客户端阻塞
                        await _broadcast_ws(msg)
                    except Exception:
                        break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket 异常: {e}")
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info(f"WebSocket 客户端断开，剩余 {len(_ws_clients)} 个")


async def _broadcast_ws(msg: str):
    """异步并发广播消息到所有 WebSocket 客户端"""
    if not _ws_clients:
        return
    disconnected = []
    tasks = []
    for client in list(_ws_clients):
        tasks.append(_safe_send(client, msg, disconnected))
    if tasks:
        await asyncio.gather(*tasks)
    for c in disconnected:
        if c in _ws_clients:
            _ws_clients.remove(c)


async def _safe_send(client: WebSocket, msg: str, disconnected: list):
    """带超时的安全发送，避免慢客户端阻塞"""
    try:
        await asyncio.wait_for(client.send_text(msg), timeout=2.0)
    except Exception:
        disconnected.append(client)


def run_server(shared_state, jsonl_logger, host: str = "0.0.0.0",
               port: int = 8000, application=None):
    """启动 Web 服务"""
    global _shared_state, _jsonl_logger, _application, _event_db
    _shared_state = shared_state
    _jsonl_logger = jsonl_logger
    _application = application
    _event_db = application.event_db if application else None

    # 初始化训练子系统
    base_dir = "data"
    sample_manager = SampleManager(base_dir=base_dir)
    annotation_engine = AnnotationEngine(base_dir=base_dir)

    on_published = application.reload_model if application is not None else None
    model_registry = ModelRegistry(base_dir=base_dir, on_model_published=on_published)

    def _broadcast_training_progress(job_id: str, data: dict):
        """将训练进度推送到所有 WebSocket 客户端（从工作线程调用）"""
        msg = json.dumps(data, ensure_ascii=False)
        loop = _event_loop
        if loop is None or not _ws_clients:
            return
        for client in list(_ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(client.send_text(msg), loop)
            except Exception:
                pass

    training_engine = TrainingEngine(
        base_dir=base_dir,
        model_registry=model_registry,
        on_progress=_broadcast_training_progress,
    )

    configure_training_api(
        sample_manager=sample_manager,
        annotation_engine=annotation_engine,
        training_engine=training_engine,
        model_registry=model_registry,
    )

    # 前端静态文件（放在最后，避免覆盖 API 路由）
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
