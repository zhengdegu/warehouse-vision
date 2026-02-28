"""
事件数据库 — SQLite 持久化
参考 Frigate 的事件存储设计：结构化查询、事件生命周期、自动清理。
替代 JSONL 文件，支持高效分页查询和统计。
"""

import json
import os
import sqlite3
import threading
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventDatabase:
    """线程安全的 SQLite 事件数据库"""

    def __init__(self, db_path: str = "data/warehouse_vision.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()
        logger.info(f"事件数据库已初始化: {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                sub_type TEXT DEFAULT '',
                camera_id TEXT NOT NULL,
                track_id INTEGER DEFAULT -1,
                class_name TEXT DEFAULT '',
                confidence REAL DEFAULT 0,
                bbox TEXT DEFAULT '[]',
                detail TEXT DEFAULT '',
                screenshot TEXT DEFAULT '',
                extra TEXT DEFAULT '{}',
                timestamp REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_camera
                ON events(camera_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_ts
                ON events(timestamp DESC);

            CREATE TABLE IF NOT EXISTS camera_stats (
                camera_id TEXT PRIMARY KEY,
                total_in INTEGER DEFAULT 0,
                total_out INTEGER DEFAULT 0,
                alert_count INTEGER DEFAULT 0,
                last_event_time REAL DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS count_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                window_start REAL NOT NULL,
                window_end REAL NOT NULL,
                window_in INTEGER DEFAULT 0,
                window_out INTEGER DEFAULT 0,
                total_in INTEGER DEFAULT 0,
                total_out INTEGER DEFAULT 0,
                by_class TEXT DEFAULT '{}',
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_count_windows_camera
                ON count_windows(camera_id, timestamp DESC);

            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                fps REAL DEFAULT 0,
                detection_fps REAL DEFAULT 0,
                motion_boxes INTEGER DEFAULT 0,
                cpu_percent REAL DEFAULT 0,
                memory_mb REAL DEFAULT 0,
                timestamp REAL NOT NULL
            );
        """)
        conn.commit()

    # ── 事件写入 ──────────────────────────────────────────────────────────

    def insert_event(self, event: Dict[str, Any]) -> int:
        conn = self._get_conn()
        extra = {k: v for k, v in event.items()
                 if k not in ("type", "sub_type", "camera_id", "track_id",
                              "class_name", "confidence", "bbox", "detail",
                              "screenshot", "timestamp")}
        cur = conn.execute("""
            INSERT INTO events
                (event_type, sub_type, camera_id, track_id, class_name,
                 confidence, bbox, detail, screenshot, extra, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.get("type", "unknown"),
            event.get("sub_type", ""),
            event.get("camera_id", ""),
            event.get("track_id", -1),
            event.get("class_name", ""),
            event.get("confidence", 0),
            json.dumps(event.get("bbox", []), ensure_ascii=False),
            event.get("detail", ""),
            event.get("screenshot", ""),
            json.dumps(extra, ensure_ascii=False),
            event.get("timestamp", time.time()),
        ))
        conn.commit()
        return cur.lastrowid

    def insert_count_window(self, count: Dict[str, Any]):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO count_windows
                (camera_id, window_start, window_end, window_in, window_out,
                 total_in, total_out, by_class, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            count.get("camera_id", ""),
            count.get("window_start", 0),
            count.get("window_end", 0),
            count.get("window_in", 0),
            count.get("window_out", 0),
            count.get("total_in", 0),
            count.get("total_out", 0),
            json.dumps(count.get("by_class", {}), ensure_ascii=False),
            count.get("timestamp", time.time()),
        ))
        conn.commit()

    # ── 统计更新 ──────────────────────────────────────────────────────────

    def update_camera_stats(self, camera_id: str,
                            total_in: int = None, total_out: int = None,
                            increment_alert: bool = False):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO camera_stats (camera_id, total_in, total_out, alert_count, last_event_time)
            VALUES (?, 0, 0, 0, ?)
            ON CONFLICT(camera_id) DO NOTHING
        """, (camera_id, time.time()))

        if total_in is not None and total_out is not None:
            conn.execute("""
                UPDATE camera_stats
                SET total_in = ?, total_out = ?, last_event_time = ?,
                    updated_at = datetime('now', 'localtime')
                WHERE camera_id = ?
            """, (total_in, total_out, time.time(), camera_id))

        if increment_alert:
            conn.execute("""
                UPDATE camera_stats
                SET alert_count = alert_count + 1,
                    last_event_time = ?,
                    updated_at = datetime('now', 'localtime')
                WHERE camera_id = ?
            """, (time.time(), camera_id))

        conn.commit()

    def get_camera_stats(self, camera_id: str = None) -> Dict:
        conn = self._get_conn()
        if camera_id:
            row = conn.execute(
                "SELECT * FROM camera_stats WHERE camera_id = ?",
                (camera_id,)
            ).fetchone()
            if row:
                return dict(row)
            return {"total_in": 0, "total_out": 0, "alert_count": 0}
        rows = conn.execute("SELECT * FROM camera_stats").fetchall()
        return {r["camera_id"]: dict(r) for r in rows}

    # ── 事件查询 ──────────────────────────────────────────────────────────

    def query_events(self, camera_id: str = None,
                     event_type: str = None,
                     limit: int = 50, offset: int = 0,
                     start_time: float = None,
                     end_time: float = None) -> List[Dict]:
        conn = self._get_conn()
        conditions = []
        params = []

        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM events
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["bbox"] = json.loads(d.get("bbox", "[]"))
            d["extra"] = json.loads(d.get("extra", "{}"))
            # 映射数据库字段名到前端期望的字段名
            d["type"] = d.pop("event_type", "unknown")
            results.append(d)
        return results

    def count_events(self, camera_id: str = None,
                     event_type: str = None,
                     start_time: float = None,
                     end_time: float = None) -> int:
        conn = self._get_conn()
        conditions = []
        params = []
        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        where = " AND ".join(conditions) if conditions else "1=1"
        row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM events WHERE {where}", params
        ).fetchone()
        return row["cnt"] if row else 0

    def get_event_summary(self, camera_id: str = None,
                          hours: int = 24) -> Dict:
        """获取事件统计摘要"""
        conn = self._get_conn()
        since = time.time() - hours * 3600
        params = [since]
        cam_filter = ""
        if camera_id:
            cam_filter = "AND camera_id = ?"
            params.append(camera_id)

        rows = conn.execute(f"""
            SELECT event_type, sub_type, COUNT(*) as cnt
            FROM events
            WHERE timestamp >= ? {cam_filter}
            GROUP BY event_type, sub_type
            ORDER BY cnt DESC
        """, params).fetchall()

        summary = {}
        for r in rows:
            key = r["event_type"]
            if r["sub_type"]:
                key += f"/{r['sub_type']}"
            summary[key] = r["cnt"]
        return summary

    # ── 系统统计 ──────────────────────────────────────────────────────────

    def insert_system_stats(self, camera_id: str, fps: float,
                            detection_fps: float, motion_boxes: int):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO system_stats
                (camera_id, fps, detection_fps, motion_boxes, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (camera_id, fps, detection_fps, motion_boxes, time.time()))
        conn.commit()

    # ── 清理 ──────────────────────────────────────────────────────────────

    def cleanup_old_events(self, retain_days: int = 30):
        """清理过期事件"""
        conn = self._get_conn()
        cutoff = time.time() - retain_days * 86400
        deleted = conn.execute(
            "DELETE FROM events WHERE timestamp < ?", (cutoff,)
        ).rowcount
        conn.execute(
            "DELETE FROM count_windows WHERE timestamp < ?", (cutoff,)
        )
        conn.execute(
            "DELETE FROM system_stats WHERE timestamp < ?",
            (time.time() - 7 * 86400,)  # 系统统计只保留 7 天
        )
        conn.commit()
        if deleted:
            logger.info(f"已清理 {deleted} 条过期事件 (>{retain_days}天)")
