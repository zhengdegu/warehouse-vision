"""训练引擎 — 负责训练任务的创建、执行和生命周期管理"""

import json
import logging
import math
import os
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

import torch

from .dataset_preparer import DatasetPreparer
from .model_registry import ModelRegistry
from .models import (
    InsufficientSamplesError,
    InvalidStateError,
    JobStatus,
    ModelNotFoundError,
    PaginatedResult,
    TrainingJob,
    TrainingMetrics,
)
from .sample_manager import SampleManager

logger = logging.getLogger(__name__)


class TrainingEngine:
    """训练引擎，管理训练任务队列和执行"""

    MIN_ANNOTATED_SAMPLES = 10

    def __init__(
        self,
        base_dir: str = "data",
        model_registry: Optional[ModelRegistry] = None,
        on_progress: Optional[callable] = None,
    ):
        """初始化训练引擎，包含任务队列和工作线程"""
        self.base_dir = base_dir
        self.model_registry = model_registry or ModelRegistry(base_dir=base_dir)
        self.sample_manager = SampleManager(base_dir=base_dir)
        self.dataset_preparer = DatasetPreparer(base_dir=base_dir)
        self.on_progress = on_progress  # callback(job_id, data_dict)

        self.metadata_dir = os.path.join(base_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        self.jobs_json = os.path.join(self.metadata_dir, "jobs.json")

        # 任务队列和线程安全锁
        self._lock = threading.Lock()
        self._queue: deque = deque()
        self._current_job_id: Optional[str] = None
        self._cancel_flags: dict = {}  # job_id -> threading.Event
        self._worker_thread: Optional[threading.Thread] = None

        # 从持久化文件加载已有任务
        self._jobs: dict = {}  # job_id -> TrainingJob
        self._load_jobs()

    # ── 公开方法 ──────────────────────────────────────────────────────────

    def create_job(
        self,
        dataset_name: str,
        epochs: int = 50,
        batch_size: int = 16,
        image_size: int = 640,
        base_model: str = "yolov8n.pt",
    ) -> TrainingJob:
        """
        创建训练任务。
        - 验证数据集已标注样本数 ≥ 10
        - 创建 TrainingJob 记录（状态 pending）
        - 加入任务队列
        - 如果当前无运行中任务，立即启动
        """
        # 验证已标注样本数
        stats = self.sample_manager.get_dataset_stats(dataset_name)
        if stats.annotated_count < self.MIN_ANNOTATED_SAMPLES:
            raise InsufficientSamplesError(
                error="insufficient_samples",
                message=f"已标注样本数不足，需要至少 {self.MIN_ANNOTATED_SAMPLES} 个，"
                        f"当前仅有 {stats.annotated_count} 个",
                details={
                    "minimum_required": self.MIN_ANNOTATED_SAMPLES,
                    "current_count": stats.annotated_count,
                },
            )

        job = TrainingJob(
            job_id=uuid.uuid4().hex,
            dataset_name=dataset_name,
            status=JobStatus.PENDING,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            base_model=base_model,
            created_time=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            self._save_jobs()
            self._try_start_next()

        return job

    def create_iteration_job(
        self,
        dataset_name: str,
        parent_model_id: str,
        epochs: int = 50,
        batch_size: int = 16,
        image_size: int = 640,
    ) -> TrainingJob:
        """
        创建迭代训练任务。
        - 验证 parent_model_id 在 Model Registry 中存在
        - 以父模型权重作为 base_model
        - 记录父模型版本到任务元数据
        """
        # 验证父模型存在
        parent_model = self.model_registry.get_model(parent_model_id)
        if parent_model is None:
            raise ModelNotFoundError(
                error="model_not_found",
                message=f"父模型 {parent_model_id} 不存在",
                details={
                    "model_id": parent_model_id,
                    "available_models": [
                        m.model_id for m in self.model_registry.list_models()
                    ],
                },
            )

        # 验证已标注样本数
        stats = self.sample_manager.get_dataset_stats(dataset_name)
        if stats.annotated_count < self.MIN_ANNOTATED_SAMPLES:
            raise InsufficientSamplesError(
                error="insufficient_samples",
                message=f"已标注样本数不足，需要至少 {self.MIN_ANNOTATED_SAMPLES} 个，"
                        f"当前仅有 {stats.annotated_count} 个",
                details={
                    "minimum_required": self.MIN_ANNOTATED_SAMPLES,
                    "current_count": stats.annotated_count,
                },
            )

        job = TrainingJob(
            job_id=uuid.uuid4().hex,
            dataset_name=dataset_name,
            status=JobStatus.PENDING,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            base_model=parent_model.weights_path,
            parent_model_id=parent_model_id,
            created_time=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            self._save_jobs()
            self._try_start_next()

        return job

    def get_job(self, job_id: str) -> TrainingJob:
        """查询任务状态、进度和指标"""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise InvalidStateError(
                error="job_not_found",
                message=f"训练任务不存在: {job_id}",
                details={"job_id": job_id},
            )
        return job

    def list_jobs(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResult:
        """分页查询训练任务历史"""
        with self._lock:
            all_jobs = list(self._jobs.values())

        # 按创建时间倒序排列
        all_jobs.sort(key=lambda j: j.created_time, reverse=True)

        total = len(all_jobs)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        start = (page - 1) * page_size
        end = start + page_size
        items = all_jobs[start:end]

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    def cancel_job(self, job_id: str) -> None:
        """
        取消训练任务。
        - 仅 pending/running 状态可取消
        - running 状态需终止训练进程
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise InvalidStateError(
                    error="job_not_found",
                    message=f"训练任务不存在: {job_id}",
                    details={"job_id": job_id},
                )

            if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                raise InvalidStateError(
                    error="invalid_state_transition",
                    message=f"无法取消状态为 {job.status.value} 的任务",
                    details={"current_status": job.status.value, "job_id": job_id},
                )

            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_time = datetime.now(timezone.utc).isoformat()
                # 从队列中移除
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                self._save_jobs()
            elif job.status == JobStatus.RUNNING:
                # 设置取消标志，_run_training 会检查
                cancel_event = self._cancel_flags.get(job_id)
                if cancel_event:
                    cancel_event.set()
                job.status = JobStatus.CANCELLED
                job.completed_time = datetime.now(timezone.utc).isoformat()
                self._save_jobs()

    # ── 内部训练执行 ──────────────────────────────────────────────────────

    def _run_training(self, job: TrainingJob) -> None:
        """
        在工作线程中执行训练（内部方法）。
        - 调用 DatasetPreparer 准备训练目录
        - 调用 YOLO model.train()
        - 记录每个 epoch 的 loss 和 mAP
        - 训练完成后注册模型到 Model Registry
        - 处理下一个队列中的任务
        """
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_flags[job.job_id] = cancel_event
            job.status = JobStatus.RUNNING
            self._current_job_id = job.job_id
            self._save_jobs()

        try:
            # 检查取消
            if cancel_event.is_set():
                return

            # 1. 准备训练目录
            yaml_path = self.dataset_preparer.prepare(
                dataset_name=job.dataset_name,
                job_id=job.job_id,
            )

            if cancel_event.is_set():
                return

            # 2. 加载 YOLO 模型并训练
            from ultralytics import YOLO

            model = YOLO(job.base_model)

            training_dir = os.path.join(self.base_dir, "training", job.job_id)
            project_dir = os.path.abspath(os.path.join(training_dir, "runs"))

            # 注册 epoch 回调，实时推送训练进度
            def _on_epoch_end(trainer):
                try:
                    epoch = trainer.epoch + 1
                    total_epochs = trainer.epochs
                    metrics = trainer.metrics or {}
                    train_loss = float(trainer.loss.mean().item()) if trainer.loss is not None else 0.0
                    map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
                    map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))

                    with self._lock:
                        job.current_epoch = epoch
                        metric = TrainingMetrics(
                            epoch=epoch - 1,
                            train_loss=train_loss,
                            map50=map50,
                            map50_95=map50_95,
                        )
                        job.metrics.append(metric)
                        if map50 > job.best_map50:
                            job.best_map50 = map50
                        self._save_jobs()

                    if self.on_progress:
                        self.on_progress(job.job_id, {
                            "type": "training_progress",
                            "job_id": job.job_id,
                            "dataset_name": job.dataset_name,
                            "current_epoch": epoch,
                            "total_epochs": total_epochs,
                            "train_loss": round(train_loss, 4),
                            "map50": round(map50, 4),
                            "map50_95": round(map50_95, 4),
                            "best_map50": round(job.best_map50, 4),
                            "status": "running",
                        })
                except Exception as e:
                    logger.debug("epoch 回调异常: %s", e)

            model.add_callback("on_fit_epoch_end", _on_epoch_end)

            # 推送训练开始事件
            if self.on_progress:
                self.on_progress(job.job_id, {
                    "type": "training_progress",
                    "job_id": job.job_id,
                    "dataset_name": job.dataset_name,
                    "current_epoch": 0,
                    "total_epochs": job.epochs,
                    "train_loss": 0,
                    "map50": 0,
                    "map50_95": 0,
                    "best_map50": 0,
                    "status": "running",
                })

            results = model.train(
                data=yaml_path,
                epochs=job.epochs,
                batch=job.batch_size,
                imgsz=job.image_size,
                project=project_dir,
                name="train",
                exist_ok=True,
                device="0" if torch.cuda.is_available() else "cpu",
            )

            if cancel_event.is_set():
                return

            # 3. 收集训练指标
            self._collect_metrics(job, results, project_dir)

            # 4. 找到 best.pt 并注册模型
            best_pt = os.path.join(project_dir, "train", "weights", "best.pt")
            if not os.path.exists(best_pt):
                # 尝试备选路径
                last_pt = os.path.join(project_dir, "train", "weights", "last.pt")
                if os.path.exists(last_pt):
                    best_pt = last_pt
                else:
                    raise FileNotFoundError(
                        f"训练完成但未找到模型权重文件: {best_pt}"
                    )

            # 构建训练配置快照
            training_config = {
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "image_size": job.image_size,
                "base_model": job.base_model,
            }

            # 构建指标摘要
            metrics_summary = {
                "map50": job.best_map50,
                "map50_95": 0.0,
            }
            if job.metrics:
                last_metric = job.metrics[-1]
                metrics_summary["map50"] = last_metric.map50
                metrics_summary["map50_95"] = last_metric.map50_95

            model_version = self.model_registry.register_model(
                job_id=job.job_id,
                weights_path=best_pt,
                metrics=metrics_summary,
                training_config=training_config,
                dataset_name=job.dataset_name,
                parent_model_id=job.parent_model_id,
            )

            with self._lock:
                job.output_model_id = model_version.model_id
                job.status = JobStatus.COMPLETED
                job.completed_time = datetime.now(timezone.utc).isoformat()
                self._save_jobs()

            if self.on_progress:
                self.on_progress(job.job_id, {
                    "type": "training_progress",
                    "job_id": job.job_id,
                    "dataset_name": job.dataset_name,
                    "current_epoch": job.current_epoch,
                    "total_epochs": job.epochs,
                    "best_map50": round(job.best_map50, 4),
                    "status": "completed",
                    "output_model_id": model_version.model_id,
                })

        except Exception as e:
            logger.error("训练任务 %s 失败: %s", job.job_id, e, exc_info=True)
            with self._lock:
                if job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    job.completed_time = datetime.now(timezone.utc).isoformat()
                self._save_jobs()

            if self.on_progress:
                self.on_progress(job.job_id, {
                    "type": "training_progress",
                    "job_id": job.job_id,
                    "dataset_name": job.dataset_name,
                    "status": "failed" if job.status == JobStatus.FAILED else "cancelled",
                    "error": str(e) if job.status == JobStatus.FAILED else None,
                })

        finally:
            with self._lock:
                self._cancel_flags.pop(job.job_id, None)
                self._current_job_id = None
                self._try_start_next()

    def _collect_metrics(self, job: TrainingJob, results, project_dir: str) -> None:
        """从训练结果中收集指标（仅在回调未收集时作为后备）"""
        # 如果回调已经收集了 metrics，跳过
        if job.metrics:
            return
        try:
            # 尝试从 results.csv 读取每个 epoch 的指标
            csv_path = os.path.join(project_dir, "train", "results.csv")
            if os.path.exists(csv_path):
                import csv
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        epoch_num = int(row.get("epoch", row.get("                  epoch", 0)))
                        train_loss = float(
                            row.get("train/box_loss", row.get("         train/box_loss", 0.0))
                        )
                        map50 = float(
                            row.get("metrics/mAP50(B)", row.get("       metrics/mAP50(B)", 0.0))
                        )
                        map50_95 = float(
                            row.get("metrics/mAP50-95(B)", row.get("    metrics/mAP50-95(B)", 0.0))
                        )
                        metric = TrainingMetrics(
                            epoch=epoch_num,
                            train_loss=train_loss,
                            map50=map50,
                            map50_95=map50_95,
                        )
                        job.metrics.append(metric)
                        job.current_epoch = epoch_num + 1
                        if map50 > job.best_map50:
                            job.best_map50 = map50
            else:
                # 如果没有 CSV，设置基本完成信息
                job.current_epoch = job.epochs
        except Exception as e:
            logger.warning("收集训练指标失败: %s", e)
            job.current_epoch = job.epochs

    # ── 队列管理 ──────────────────────────────────────────────────────────

    def _try_start_next(self) -> None:
        """尝试启动队列中的下一个任务（需在锁内调用）"""
        if self._current_job_id is not None:
            return

        while self._queue:
            next_job_id = self._queue.popleft()
            job = self._jobs.get(next_job_id)
            if job is None or job.status != JobStatus.PENDING:
                continue

            # 启动工作线程
            self._worker_thread = threading.Thread(
                target=self._run_training,
                args=(job,),
                daemon=True,
            )
            self._worker_thread.start()
            return

    # ── 持久化 ────────────────────────────────────────────────────────────

    def _load_jobs(self) -> None:
        """从 JSON 文件加载任务记录"""
        if not os.path.exists(self.jobs_json):
            return
        try:
            with open(self.jobs_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                job = self._job_from_dict(item)
                self._jobs[job.job_id] = job
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("加载任务记录失败: %s", e)

    def _save_jobs(self) -> None:
        """将任务记录保存到 JSON 文件（需在锁内调用）"""
        data = [self._job_to_dict(j) for j in self._jobs.values()]
        with open(self.jobs_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _job_to_dict(job: TrainingJob) -> dict:
        """将 TrainingJob 转换为可序列化的字典"""
        return {
            "job_id": job.job_id,
            "dataset_name": job.dataset_name,
            "status": job.status.value,
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "image_size": job.image_size,
            "base_model": job.base_model,
            "parent_model_id": job.parent_model_id,
            "current_epoch": job.current_epoch,
            "metrics": [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "map50": m.map50,
                    "map50_95": m.map50_95,
                }
                for m in job.metrics
            ],
            "best_map50": job.best_map50,
            "error_message": job.error_message,
            "created_time": job.created_time,
            "completed_time": job.completed_time,
            "output_model_id": job.output_model_id,
        }

    @staticmethod
    def _job_from_dict(d: dict) -> TrainingJob:
        """从字典恢复 TrainingJob"""
        metrics = [
            TrainingMetrics(
                epoch=m["epoch"],
                train_loss=m["train_loss"],
                map50=m.get("map50", 0.0),
                map50_95=m.get("map50_95", 0.0),
            )
            for m in d.get("metrics", [])
        ]
        return TrainingJob(
            job_id=d["job_id"],
            dataset_name=d["dataset_name"],
            status=JobStatus(d["status"]),
            epochs=d.get("epochs", 50),
            batch_size=d.get("batch_size", 16),
            image_size=d.get("image_size", 640),
            base_model=d.get("base_model", "yolov8n.pt"),
            parent_model_id=d.get("parent_model_id"),
            current_epoch=d.get("current_epoch", 0),
            metrics=metrics,
            best_map50=d.get("best_map50", 0.0),
            error_message=d.get("error_message"),
            created_time=d.get("created_time", ""),
            completed_time=d.get("completed_time"),
            output_model_id=d.get("output_model_id"),
        )
