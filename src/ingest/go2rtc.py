"""
go2rtc 流管理模块 — 参考 Frigate go2rtc 集成

通过 go2rtc REST API 动态管理 RTSP 流：
- 添加/更新摄像头时自动注册 go2rtc stream
- 删除摄像头时自动移除 go2rtc stream
- 同时更新 go2rtc.yaml 配置文件持久化

go2rtc API:
- PUT /api/streams?name={name}&src={rtsp_url}  → 添加/更新流
- DELETE /api/streams?src={name}                → 删除流
- GET /api/streams                              → 列出所有流
"""

import logging
import requests
import yaml
import os

logger = logging.getLogger(__name__)

# go2rtc 默认配置
DEFAULT_GO2RTC_API = "http://127.0.0.1:1984"
DEFAULT_GO2RTC_RTSP_PORT = 8555
DEFAULT_GO2RTC_CONFIG = "configs/go2rtc.yaml"


class Go2RTCManager:
    """go2rtc 流管理器 — 通过 REST API + 配置文件双写"""

    def __init__(self, api_url: str = DEFAULT_GO2RTC_API,
                 rtsp_port: int = DEFAULT_GO2RTC_RTSP_PORT,
                 config_path: str = DEFAULT_GO2RTC_CONFIG):
        self.api_url = api_url
        self.rtsp_port = rtsp_port
        self.config_path = config_path

    def get_restream_url(self, stream_name: str) -> str:
        """获取 go2rtc restream 地址（供 ffmpeg detect 拉流）"""
        return f"rtsp://127.0.0.1:{self.rtsp_port}/{stream_name}"

    def add_stream(self, stream_name: str, rtsp_url: str) -> bool:
        """
        添加或更新 go2rtc 流 — 参考 Frigate go2rtc_add_stream。
        同时通过 API 热更新 + 写入配置文件持久化。
        """
        # 1. 通过 API 热更新（go2rtc 立即生效，无需重启）
        try:
            r = requests.put(
                f"{self.api_url}/api/streams",
                params={"name": stream_name, "src": rtsp_url},
                timeout=10,
            )
            if not r.ok:
                logger.error(f"go2rtc 添加流失败 {stream_name}: {r.text}")
                return False
            logger.info(f"go2rtc 流已添加: {stream_name}")
        except requests.RequestException as e:
            logger.warning(f"go2rtc API 不可用，仅写入配置文件: {e}")

        # 2. 写入配置文件持久化
        self._update_config_file(stream_name, rtsp_url)
        return True

    def remove_stream(self, stream_name: str) -> bool:
        """
        删除 go2rtc 流 — 参考 Frigate go2rtc_delete_stream。
        """
        # 1. 通过 API 热删除
        try:
            r = requests.delete(
                f"{self.api_url}/api/streams",
                params={"src": stream_name},
                timeout=10,
            )
            if not r.ok:
                logger.error(f"go2rtc 删除流失败 {stream_name}: {r.text}")
            else:
                logger.info(f"go2rtc 流已删除: {stream_name}")
        except requests.RequestException as e:
            logger.warning(f"go2rtc API 不可用: {e}")

        # 2. 从配置文件移除
        self._remove_from_config_file(stream_name)
        return True

    def _update_config_file(self, stream_name: str, rtsp_url: str):
        """更新 go2rtc.yaml 配置文件"""
        config = self._load_config()
        if "streams" not in config:
            config["streams"] = {}
        config["streams"][stream_name] = rtsp_url
        self._save_config(config)

    def _remove_from_config_file(self, stream_name: str):
        """从 go2rtc.yaml 移除流"""
        config = self._load_config()
        streams = config.get("streams", {})
        if stream_name in streams:
            del streams[stream_name]
            self._save_config(config)

    def _load_config(self) -> dict:
        """加载 go2rtc.yaml"""
        if os.path.isfile(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "streams": {},
            "rtsp": {"listen": f":{self.rtsp_port}"},
            "api": {"listen": ":1984"},
            "log": {"level": "info"},
        }

    def _save_config(self, config: dict):
        """保存 go2rtc.yaml"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
