"""
配置模型 — Pydantic v2 严格校验
参考 Frigate 的分层配置设计：全局默认 → 摄像头级覆盖
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


# ── 检测区域（Zone）──────────────────────────────────────────────────────────

class ZoneConfig(BaseModel):
    """命名检测区域，支持多区域独立规则"""
    name: str = ""
    coordinates: List[List[float]] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list,
                               description="该区域关注的目标类别，空=全部")
    inactivity_threshold: float = Field(0, description="区域内无活动超时(秒)，0=禁用")


# ── 运动检测 ─────────────────────────────────────────────────────────────────

class MotionConfig(BaseModel):
    """运动检测配置 — Frigate 核心优化：只在有运动的区域跑 AI"""
    enabled: bool = True
    threshold: int = Field(40, ge=1, le=255, description="像素差阈值")
    contour_area: int = Field(200, ge=10, description="最小轮廓面积(像素)")
    frame_alpha: float = Field(0.02, ge=0.001, le=1.0, description="背景更新速率")
    mask: List[List[List[float]]] = Field(default_factory=list,
                                          description="运动遮罩多边形列表")


# ── 目标检测 ─────────────────────────────────────────────────────────────────

class DetectConfig(BaseModel):
    enabled: bool = True
    width: int = Field(0, description="检测分辨率宽，0=自动")
    height: int = Field(0, description="检测分辨率高，0=自动")
    fps: int = Field(5, ge=1, le=30, description="送检帧率")
    max_disappeared: int = Field(30, description="目标消失多少帧后移除")


# ── 规则 ─────────────────────────────────────────────────────────────────────

class IntrusionRuleConfig(BaseModel):
    enabled: bool = False
    confirm_frames: int = 5
    cooldown: float = 30.0

class TripwireConfig(BaseModel):
    id: str = ""
    name: str = ""
    p1: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    p2: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    direction: str = "left_to_right"

class TripwireRuleConfig(BaseModel):
    enabled: bool = False

class CountingRuleConfig(BaseModel):
    enabled: bool = False
    window_seconds: int = 60

class DwellConfig(BaseModel):
    enabled: bool = False
    max_seconds: float = 120
    confirm_frames: int = 5
    cooldown: float = 60

class CrowdConfig(BaseModel):
    enabled: bool = False
    max_count: int = 5
    radius: float = 200
    confirm_frames: int = 5
    cooldown: float = 60

class ProximityConfig(BaseModel):
    enabled: bool = False
    min_distance: float = 50
    confirm_frames: int = 3
    cooldown: float = 30

class FightConfig(BaseModel):
    enabled: bool = False
    proximity_radius: float = 150
    min_speed: float = 60
    min_persons: int = 2
    confirm_frames: int = 3
    cooldown: float = 30

class FallConfig(BaseModel):
    enabled: bool = False
    ratio_threshold: float = 1.0
    min_ratio_change: float = 0.5
    min_y_drop: float = 20
    confirm_frames: int = 2
    cooldown: float = 30


# ── 时间周期约束 ─────────────────────────────────────────────────────────

class TimePeriodConfig(BaseModel):
    """时间周期约束：仅在指定时段内启用规则"""
    enabled: bool = Field(False, description="是否启用时间约束，False=全天候运行")
    start: str = Field("00:00", description="开始时间 HH:MM")
    end: str = Field("23:59", description="结束时间 HH:MM")
    days: List[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6],
        description="生效的星期几，0=周一 ... 6=周日"
    )


class AnomalyRuleConfig(BaseModel):
    dwell: DwellConfig = DwellConfig()
    crowd: CrowdConfig = CrowdConfig()
    proximity: ProximityConfig = ProximityConfig()
    fight: FightConfig = FightConfig()
    fall: FallConfig = FallConfig()
    time_period: TimePeriodConfig = TimePeriodConfig()

class RulesConfig(BaseModel):
    intrusion: IntrusionRuleConfig = IntrusionRuleConfig()
    tripwire: TripwireRuleConfig = TripwireRuleConfig()
    counting: CountingRuleConfig = CountingRuleConfig()
    anomaly: AnomalyRuleConfig = AnomalyRuleConfig()
    alert_types: List[str] = Field(
        default_factory=list,
        description="触发告警的事件类型列表，空=全部告警。"
                    "可选值: intrusion, tripwire, presence, anomaly/dwell, anomaly/crowd, "
                    "anomaly/proximity, anomaly/fight, anomaly/fall"
    )


# ── 录像 ─────────────────────────────────────────────────────────────────────

class RecordConfig(BaseModel):
    """录像配置 — 参考 Frigate 的事件驱动录像"""
    enabled: bool = False
    retain_days: int = Field(7, ge=1, description="录像保留天数")
    events_retain_days: int = Field(30, ge=1, description="事件录像保留天数")
    segment_seconds: int = Field(60, ge=10, description="录像分段时长(秒)")
    output_dir: str = "recordings"


# ── 快照 ─────────────────────────────────────────────────────────────────────

class SnapshotConfig(BaseModel):
    enabled: bool = True
    draw_bbox: bool = True
    draw_zones: bool = True
    draw_tripwire: bool = True
    quality: int = Field(70, ge=1, le=100)


# ── 摄像头 ───────────────────────────────────────────────────────────────────

class CameraConfig(BaseModel):
    id: str
    name: str = ""
    url: str
    width: int = Field(0, description="0=自动探测")
    height: int = Field(0, description="0=自动探测")
    fps: int = 15
    roi: List[List[float]] = Field(default_factory=list)
    zones: List[ZoneConfig] = Field(default_factory=list)
    tripwires: List[TripwireConfig] = Field(default_factory=list)
    rules: RulesConfig = RulesConfig()
    motion: MotionConfig = MotionConfig()
    detect: DetectConfig = DetectConfig()
    record: RecordConfig = RecordConfig()
    snapshot: SnapshotConfig = SnapshotConfig()

    @field_validator("name", mode="before")
    @classmethod
    def default_name(cls, v, info):
        return v or info.data.get("id", "")


# ── 模型 ─────────────────────────────────────────────────────────────────────

class RoboflowConfig(BaseModel):
    """Roboflow RF-DETR 检测器配置"""
    model_id: str = Field("rfdetr-base",
                          description="RF-DETR 模型: rfdetr-base, rfdetr-large, rfdetr-nano, "
                                      "rfdetr-small, rfdetr-medium, rfdetr-seg-preview")
    classes: List[str] = Field(
        default_factory=list,
        description="检测类别名称白名单（如 ['person', 'car']），空=全部")


class ModelConfig(BaseModel):
    detector_type: str = Field("yolo",
                               description="检测器类型: yolo 或 roboflow")
    path: str = "yolov8n.pt"
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    classes: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    roboflow: RoboflowConfig = RoboflowConfig()


# ── 事件 ─────────────────────────────────────────────────────────────────────

class EventsConfig(BaseModel):
    output_dir: str = "events"
    screenshot: bool = True
    draw_bbox: bool = True
    draw_roi: bool = True
    draw_tripwire: bool = True


# ── Web ──────────────────────────────────────────────────────────────────────

class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


# ── 数据库 ───────────────────────────────────────────────────────────────────

class DatabaseConfig(BaseModel):
    path: str = "data/warehouse_vision.db"


# ── 系统健康 ─────────────────────────────────────────────────────────────────

class SystemConfig(BaseModel):
    log_level: str = "INFO"
    stats_interval: int = Field(60, description="系统状态上报间隔(秒)")


# ── 顶层配置 ─────────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    cameras: List[CameraConfig] = Field(default_factory=list)
    model: ModelConfig = ModelConfig()
    events: EventsConfig = EventsConfig()
    web: WebConfig = WebConfig()
    database: DatabaseConfig = DatabaseConfig()
    system: SystemConfig = SystemConfig()
