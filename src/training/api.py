"""训练子系统 API 路由 — 挂载到 /api/training"""

import dataclasses
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .annotation_engine import AnnotationEngine
from .model_registry import ModelRegistry
from .models import (
    BatchLimitError,
    InsufficientSamplesError,
    InvalidAnnotationError,
    InvalidFormatError,
    InvalidStateError,
    ModelInUseError,
    ModelNotFoundError,
    TrainingError,
)
from .sample_manager import SampleManager
from .training_engine import TrainingEngine

logger = logging.getLogger(__name__)

# ── 全局组件实例（由 server.py 注入） ─────────────────────────────────────────

_sample_manager: Optional[SampleManager] = None
_annotation_engine: Optional[AnnotationEngine] = None
_training_engine: Optional[TrainingEngine] = None
_model_registry: Optional[ModelRegistry] = None


def configure(
    sample_manager: SampleManager,
    annotation_engine: AnnotationEngine,
    training_engine: TrainingEngine,
    model_registry: ModelRegistry,
) -> None:
    """由 server.py 调用，注入组件实例"""
    global _sample_manager, _annotation_engine, _training_engine, _model_registry
    _sample_manager = sample_manager
    _annotation_engine = annotation_engine
    _training_engine = training_engine
    _model_registry = model_registry


# ── 依赖注入 ──────────────────────────────────────────────────────────────────


def get_sample_manager() -> SampleManager:
    assert _sample_manager is not None, "SampleManager 未初始化"
    return _sample_manager


def get_annotation_engine() -> AnnotationEngine:
    assert _annotation_engine is not None, "AnnotationEngine 未初始化"
    return _annotation_engine


def get_training_engine() -> TrainingEngine:
    assert _training_engine is not None, "TrainingEngine 未初始化"
    return _training_engine


def get_model_registry() -> ModelRegistry:
    assert _model_registry is not None, "ModelRegistry 未初始化"
    return _model_registry


# ── Pydantic 请求/响应模型 ────────────────────────────────────────────────────


class CreateDatasetRequest(BaseModel):
    name: str
    classes: List[str]


class AnnotationBoxBody(BaseModel):
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


class SaveAnnotationRequest(BaseModel):
    annotations: List[AnnotationBoxBody]
    dataset_classes: List[str]


class AutoAnnotateRequest(BaseModel):
    model_path: str
    confidence_threshold: float = 0.5
    class_mapping: Optional[Dict[int, int]] = None


class CreateJobRequest(BaseModel):
    dataset_name: str
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640
    base_model: str = "yolov8n.pt"


class CreateIterationJobRequest(BaseModel):
    dataset_name: str
    parent_model_id: str
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

# 异常类 → HTTP 状态码映射
_ERROR_STATUS_MAP: Dict[type, int] = {
    InvalidFormatError: 400,
    BatchLimitError: 400,
    InvalidAnnotationError: 400,
    InsufficientSamplesError: 400,
    ModelNotFoundError: 404,
    InvalidStateError: 409,
    ModelInUseError: 409,
}


def _error_response(exc: TrainingError) -> JSONResponse:
    """将 TrainingError 转换为标准 JSON 错误响应"""
    status_code = _ERROR_STATUS_MAP.get(type(exc), 400)
    # 对于 "not_found" 类错误，统一返回 404
    if exc.error in ("sample_not_found", "dataset_not_found", "job_not_found"):
        status_code = 404
    return JSONResponse(
        status_code=status_code,
        content={"error": exc.error, "message": exc.message, "details": exc.details},
    )


def _dc_to_dict(obj):
    """将 dataclass 实例（或列表）转换为可序列化的 dict"""
    if isinstance(obj, list):
        return [_dc_to_dict(item) for item in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


# ── 路由 ──────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/training", tags=["training"])


# ── 样本管理 ──────────────────────────────────────────────────────────────────


@router.post("/samples/upload")
async def upload_samples(
    files: List[UploadFile] = File(...),
    dataset_name: Optional[str] = Query(None),
    sm: SampleManager = Depends(get_sample_manager),
):
    """上传样本图片（支持多文件）"""
    try:
        samples = sm.upload_samples(files=files, dataset_name=dataset_name)
        return {"items": _dc_to_dict(samples)}
    except TrainingError as exc:
        return _error_response(exc)


@router.get("/samples")
async def list_samples(
    dataset_name: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sm: SampleManager = Depends(get_sample_manager),
):
    """分页查询样本列表"""
    result = sm.list_samples(dataset_name=dataset_name, page=page, page_size=page_size)
    return _dc_to_dict(result)


@router.delete("/samples/{sample_id}")
async def delete_sample(
    sample_id: str,
    sm: SampleManager = Depends(get_sample_manager),
):
    """删除样本"""
    try:
        sm.delete_sample(sample_id)
        return {"message": "样本已删除", "sample_id": sample_id}
    except TrainingError as exc:
        return _error_response(exc)


# ── 数据集管理 ────────────────────────────────────────────────────────────────


@router.post("/datasets")
async def create_dataset(
    body: CreateDatasetRequest,
    sm: SampleManager = Depends(get_sample_manager),
):
    """创建数据集"""
    try:
        meta = sm.create_dataset(dataset_name=body.name, classes=body.classes)
        return _dc_to_dict(meta)
    except TrainingError as exc:
        return _error_response(exc)


@router.get("/datasets")
async def list_datasets(
    sm: SampleManager = Depends(get_sample_manager),
):
    """查询数据集列表"""
    datasets = sm.list_datasets()
    return {"items": _dc_to_dict(datasets)}


@router.get("/datasets/{dataset_name}/stats")
async def get_dataset_stats(
    dataset_name: str,
    sm: SampleManager = Depends(get_sample_manager),
):
    """数据集统计"""
    try:
        stats = sm.get_dataset_stats(dataset_name)
        return _dc_to_dict(stats)
    except TrainingError as exc:
        return _error_response(exc)


@router.delete("/datasets/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    sm: SampleManager = Depends(get_sample_manager),
):
    """删除数据集"""
    try:
        sm.delete_dataset(dataset_name)
        return {"message": "数据集已删除", "dataset_name": dataset_name}
    except TrainingError as exc:
        return _error_response(exc)


# ── 标注 ──────────────────────────────────────────────────────────────────────


@router.put("/samples/{sample_id}/annotations")
async def save_annotation(
    sample_id: str,
    body: SaveAnnotationRequest,
    ae: AnnotationEngine = Depends(get_annotation_engine),
):
    """提交/更新标注"""
    from .models import AnnotationBox

    try:
        boxes = [
            AnnotationBox(
                class_id=b.class_id,
                center_x=b.center_x,
                center_y=b.center_y,
                width=b.width,
                height=b.height,
            )
            for b in body.annotations
        ]
        ae.save_annotation(
            sample_id=sample_id,
            annotations=boxes,
            dataset_classes=body.dataset_classes,
        )
        return {"message": "标注已保存", "sample_id": sample_id}
    except TrainingError as exc:
        return _error_response(exc)


@router.get("/samples/{sample_id}/annotations")
async def get_annotation(
    sample_id: str,
    ae: AnnotationEngine = Depends(get_annotation_engine),
):
    """查询标注"""
    try:
        annotations = ae.get_annotation(sample_id)
        return {"annotations": _dc_to_dict(annotations)}
    except TrainingError as exc:
        return _error_response(exc)


@router.post("/datasets/{dataset_name}/auto-annotate")
async def auto_annotate(
    dataset_name: str,
    body: AutoAnnotateRequest,
    ae: AnnotationEngine = Depends(get_annotation_engine),
):
    """自动标注"""
    try:
        result = ae.auto_annotate(
            dataset_name=dataset_name,
            model_path=body.model_path,
            confidence_threshold=body.confidence_threshold,
            class_mapping=body.class_mapping,
        )
        return _dc_to_dict(result)
    except TrainingError as exc:
        return _error_response(exc)


# ── 训练 ──────────────────────────────────────────────────────────────────────


@router.post("/jobs")
async def create_job(
    body: CreateJobRequest,
    te: TrainingEngine = Depends(get_training_engine),
):
    """创建训练任务"""
    try:
        job = te.create_job(
            dataset_name=body.dataset_name,
            epochs=body.epochs,
            batch_size=body.batch_size,
            image_size=body.image_size,
            base_model=body.base_model,
        )
        return _dc_to_dict(job)
    except TrainingError as exc:
        return _error_response(exc)


@router.post("/jobs/iterate")
async def create_iteration_job(
    body: CreateIterationJobRequest,
    te: TrainingEngine = Depends(get_training_engine),
):
    """创建迭代训练任务"""
    try:
        job = te.create_iteration_job(
            dataset_name=body.dataset_name,
            parent_model_id=body.parent_model_id,
            epochs=body.epochs,
            batch_size=body.batch_size,
            image_size=body.image_size,
        )
        return _dc_to_dict(job)
    except TrainingError as exc:
        return _error_response(exc)


@router.get("/jobs")
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    te: TrainingEngine = Depends(get_training_engine),
):
    """训练任务列表"""
    result = te.list_jobs(page=page, page_size=page_size)
    return _dc_to_dict(result)


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    te: TrainingEngine = Depends(get_training_engine),
):
    """查询任务状态"""
    try:
        job = te.get_job(job_id)
        return _dc_to_dict(job)
    except TrainingError as exc:
        return _error_response(exc)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    te: TrainingEngine = Depends(get_training_engine),
):
    """取消任务"""
    try:
        te.cancel_job(job_id)
        return {"message": "任务已取消", "job_id": job_id}
    except TrainingError as exc:
        return _error_response(exc)


# ── 模型管理 ──────────────────────────────────────────────────────────────────


@router.get("/models")
async def list_models(
    mr: ModelRegistry = Depends(get_model_registry),
):
    """模型列表"""
    models = mr.list_models()
    return {"items": _dc_to_dict(models)}


@router.post("/models/{model_id}/publish")
async def publish_model(
    model_id: str,
    mr: ModelRegistry = Depends(get_model_registry),
):
    """发布模型"""
    try:
        mr.publish_model(model_id)
        return {"message": "模型已发布", "model_id": model_id}
    except TrainingError as exc:
        return _error_response(exc)


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    mr: ModelRegistry = Depends(get_model_registry),
):
    """删除模型"""
    try:
        mr.delete_model(model_id)
        return {"message": "模型已删除", "model_id": model_id}
    except TrainingError as exc:
        return _error_response(exc)
