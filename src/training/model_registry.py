"""模型注册中心 — 管理训练产出的模型版本、元数据和发布状态"""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import yaml

from .models import ModelInUseError, ModelNotFoundError, ModelVersion


class ModelRegistry:
    """模型版本管理和发布"""

    def __init__(
        self,
        base_dir: str = "data",
        config_path: str = "configs/cameras.yaml",
        on_model_published: Optional[Callable] = None,
    ):
        self.base_dir = base_dir
        self.config_path = config_path
        self.on_model_published = on_model_published

        # 确保目录存在
        self.models_dir = os.path.join(base_dir, "models")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.models_json = os.path.join(self.metadata_dir, "models.json")
        if not os.path.exists(self.models_json):
            self._save_all([])

    # ── 公开方法 ──────────────────────────────────────────────────────────

    def register_model(
        self,
        job_id: str,
        weights_path: str,
        metrics: dict,
        training_config: dict,
        dataset_name: str,
        parent_model_id: Optional[str] = None,
    ) -> ModelVersion:
        """注册新模型版本，保存权重和元数据"""
        models = self._load_all()
        model_id = str(uuid.uuid4())
        version = f"v{len(models) + 1}"

        # 创建模型目录并复制权重
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        dest_weights = os.path.join(model_dir, "best.pt")
        shutil.copy2(weights_path, dest_weights)

        model = ModelVersion(
            model_id=model_id,
            version=version,
            job_id=job_id,
            dataset_name=dataset_name,
            weights_path=dest_weights,
            metrics=metrics,
            training_config=training_config,
            parent_model_id=parent_model_id,
            published=False,
            created_time=datetime.now(timezone.utc).isoformat(),
        )

        # 保存 meta.json 到模型目录
        self._save_meta_json(model_dir, model)

        # 追加到全局 models.json
        models.append(self._to_dict(model))
        self._save_all(models)

        return model

    def list_models(self) -> List[ModelVersion]:
        """列出所有已注册模型"""
        return [self._from_dict(d) for d in self._load_all()]

    def get_model(self, model_id: str) -> Optional[ModelVersion]:
        """获取模型详情，不存在返回 None"""
        for d in self._load_all():
            if d["model_id"] == model_id:
                return self._from_dict(d)
        return None

    def publish_model(self, model_id: str) -> None:
        """
        发布模型到生产环境。
        1. 复制权重到根目录
        2. 更新 cameras.yaml 中的 model.path
        3. 调用 on_model_published 回调
        """
        models = self._load_all()
        idx = self._find_index(models, model_id)

        model_dict = models[idx]
        src_weights = model_dict["weights_path"]

        # 目标文件名：model_{model_id}.pt
        dest_filename = f"model_{model_id}.pt"
        dest_path = os.path.join(os.path.dirname(self.config_path), dest_filename)
        shutil.copy2(src_weights, dest_path)

        # 更新 cameras.yaml
        self._update_cameras_config(dest_filename)

        # 取消其他模型的 published 状态，设置当前模型为 published
        for m in models:
            m["published"] = m["model_id"] == model_id
        self._save_all(models)

        # 更新模型目录中的 meta.json
        model_dir = os.path.join(self.models_dir, model_id)
        if os.path.isdir(model_dir):
            self._save_meta_json(model_dir, self._from_dict(models[idx]))

        # 回调通知热重载
        if self.on_model_published:
            self.on_model_published(dest_path)

    def delete_model(self, model_id: str) -> None:
        """
        删除模型版本。
        - 当前生产模型拒绝删除
        - 删除权重文件和元数据
        """
        models = self._load_all()
        idx = self._find_index(models, model_id)

        if models[idx].get("published", False):
            raise ModelInUseError(
                error="model_in_use",
                message="无法删除当前生产环境使用的模型，请先切换到其他模型",
                details={"model_id": model_id},
            )

        # 删除模型目录（权重 + meta.json）
        model_dir = os.path.join(self.models_dir, model_id)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)

        # 从全局列表移除
        del models[idx]
        self._save_all(models)

    # ── 内部辅助 ──────────────────────────────────────────────────────────

    def _find_index(self, models: List[dict], model_id: str) -> int:
        for i, m in enumerate(models):
            if m["model_id"] == model_id:
                return i
        raise ModelNotFoundError(
            error="model_not_found",
            message=f"模型 {model_id} 不存在",
            details={"model_id": model_id},
        )

    def _load_all(self) -> List[dict]:
        if not os.path.exists(self.models_json):
            return []
        with open(self.models_json, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_all(self, models: List[dict]) -> None:
        with open(self.models_json, "w", encoding="utf-8") as f:
            json.dump(models, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _save_meta_json(model_dir: str, model: "ModelVersion") -> None:
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(ModelRegistry._to_dict(model), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _to_dict(model: ModelVersion) -> dict:
        return {
            "model_id": model.model_id,
            "version": model.version,
            "job_id": model.job_id,
            "dataset_name": model.dataset_name,
            "weights_path": model.weights_path,
            "metrics": model.metrics,
            "training_config": model.training_config,
            "parent_model_id": model.parent_model_id,
            "published": model.published,
            "created_time": model.created_time,
        }

    @staticmethod
    def _from_dict(d: dict) -> ModelVersion:
        return ModelVersion(
            model_id=d["model_id"],
            version=d["version"],
            job_id=d["job_id"],
            dataset_name=d["dataset_name"],
            weights_path=d["weights_path"],
            metrics=d["metrics"],
            training_config=d["training_config"],
            parent_model_id=d.get("parent_model_id"),
            published=d.get("published", False),
            created_time=d.get("created_time", ""),
        )

    def _update_cameras_config(self, model_filename: str) -> None:
        """更新 cameras.yaml 中的 model.path"""
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config and "model" in config:
            config["model"]["path"] = model_filename
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
