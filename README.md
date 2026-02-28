# 仓储安防视频分析系统

基于 YOLO + ByteTrack 的多路摄像头实时视频分析系统，支持入侵检测、越线计数、行为异常检测、在线训练等功能。

## 系统架构

```
RTSP 摄像头 → 拉流(ffmpeg) → 运动预过滤 → YOLO 检测 + ByteTrack 跟踪
                                                    ↓
                                              规则引擎（入侵/越线/计数/异常/存在）
                                                    ↓
                                         事件存储(SQLite) + 截图 + WebSocket 推送
                                                    ↓
                                         React 前端（实时画面 / 事件 / 配置 / 训练）
```

核心设计参考 Frigate：
- 运动检测作为预过滤，仅在有运动时运行 AI 推理，大幅降低 GPU/CPU 负载
- 每路摄像头独立线程，多路共享同一个 YOLO 检测器实例
- 看门狗自动恢复崩溃的摄像头管线（指数退避）

## 功能一览

| 功能 | 说明 |
|------|------|
| 入侵检测 | ROI 多边形区域内目标检测，支持确认帧数和冷却时间 |
| 越线检测 | Tripwire 绊线，支持方向判定（左→右 / 右→左） |
| 流量计数 | 进/出计数，滑动窗口统计（默认 60 秒） |
| 滞留检测 | 目标在区域内停留超时告警 |
| 聚集检测 | 区域内人数超过阈值告警 |
| 人车过近 | 人与车辆距离过近告警 |
| 打架检测 | 基于 Pose 姿态估计 + 运动速度判定 |
| 跌倒检测 | 基于 Pose 宽高比突变 + Y 轴下降 |
| 存在检测 | 目标出现/消失通知 |
| 在线训练 | 数据集管理、标注、YOLO 训练、模型热重载 |

## 工程结构

```
warehouse-vision/
├── configs/
│   └── cameras.yaml            # 摄像头与规则配置（核心配置文件）
├── src/
│   ├── app.py                  # 主程序调度（多线程管线 + 看门狗）
│   ├── config/schema.py        # Pydantic 配置模型
│   ├── ingest/rtsp_reader.py   # RTSP 拉流（ffmpeg + TCP）
│   ├── vision/
│   │   ├── detector.py         # YOLO 检测 + ByteTrack 跟踪 + Pose
│   │   └── motion.py           # 运动检测预过滤
│   ├── rules/
│   │   ├── geometry.py         # 几何运算（点线关系、多边形判定）
│   │   ├── intrusion.py        # 入侵检测
│   │   ├── tripwire.py         # 越线检测
│   │   ├── counting.py         # 滑动窗口流量计数
│   │   ├── anomaly.py          # 行为异常检测引擎
│   │   └── presence.py         # 存在检测
│   ├── events/
│   │   ├── database.py         # SQLite 事件存储
│   │   ├── es_store.py         # Elasticsearch 存储（可选）
│   │   ├── evidence.py         # 事件截图 + 中文标注
│   │   └── logger.py           # JSONL 日志
│   ├── training/               # 在线训练子系统
│   │   ├── api.py              # 训练 REST API
│   │   ├── sample_manager.py   # 样本管理
│   │   ├── annotation_engine.py# 标注引擎
│   │   ├── training_engine.py  # YOLO 训练引擎
│   │   └── model_registry.py   # 模型注册 + 发布
│   └── web/server.py           # FastAPI Web 服务
├── web/                        # React + Vite 前端
│   └── src/
│       ├── pages/
│       │   ├── LivePage.tsx    # 实时画面（MJPEG 流）
│       │   ├── EventsPage.tsx  # 事件查询（分页/过滤）
│       │   ├── ConfigPage.tsx  # 摄像头配置（ROI/绊线/规则）
│       │   ├── SystemPage.tsx  # 系统监控（性能/健康）
│       │   └── TrainingPage.tsx# 在线训练（数据集/标注/训练/模型）
│       └── components/         # 通用组件
├── scripts/
│   ├── run_all.py              # 启动全部服务（后端 + Web）
│   ├── run_single_cam.py       # 单摄像头测试
│   └── roi_selector.py         # ROI 点选工具
├── Dockerfile                  # 多阶段构建（前端 + 后端 + GPU）
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10+（推荐 3.12）
- Node.js 18+（前端构建）
- ffmpeg（RTSP 拉流）
- GPU（可选，有 CUDA 显卡可加速推理）

## 本地运行

### 1. 安装 Python 依赖

```bash
# 使用 conda 环境（推荐）
conda create -n py312 python=3.12
conda activate py312

# 安装 PyTorch（GPU 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 安装前端依赖

```bash
cd web
npm install
```

### 3. 配置摄像头

编辑 `configs/cameras.yaml`，配置 RTSP 地址和检测规则（详见下方配置说明）。

### 4. 启动后端

```bash
python scripts/run_all.py
```

后端启动后访问 http://localhost:8000

### 5. 启动前端（开发模式）

```bash
cd web
npm run dev
```

前端开发服务器 http://localhost:5173，API 请求自动代理到后端。

### 6. 构建前端（生产模式）

```bash
cd web
npm run build
```

构建产物输出到 `web/dist/`，后端会自动托管静态文件，直接访问 http://localhost:8000 即可。

## Docker 部署

### docker-compose（推荐）

```bash
# 构建并启动（GPU 模式，需要 nvidia-container-toolkit）
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

`docker-compose.yml` 已配置：
- 端口映射 `8000:8000`（前端 + 后端统一入口）
- 挂载卷：`configs/`（配置）、`data/`（数据库/模型/样本）、`events/`（截图/日志）
- GPU 资源预留

如不需要 GPU，删除 `docker-compose.yml` 中 `deploy` 段即可。

### docker run（手动）

```bash
# CPU 模式
docker build -t warehouse-vision .
docker run -d -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/events:/app/events \
  warehouse-vision

# GPU 模式
docker run -d --gpus all -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/events:/app/events \
  warehouse-vision
```

启动后访问 http://localhost:8000 即可使用（前端已内置在镜像中）。

## 配置说明

所有配置集中在 `configs/cameras.yaml`，主要分为以下几个部分：

### 摄像头配置

```yaml
cameras:
  - id: cam01                    # 唯一标识
    name: 停车场入口              # 显示名称
    url: rtsp://...              # RTSP 地址
    width: 1280
    height: 720
    fps: 15                      # 拉流帧率
    roi:                         # ROI 多边形顶点（入侵检测区域）
      - [360, 236]
      - [45, 310]
      - [123, 661]
      - [751, 362]
    tripwires:                   # 绊线配置（越线检测）
      - id: tw01
        name: 入口线1
        p1: [149, 710]           # 起点
        p2: [1105, 362]          # 终点
        direction: left_to_right # 正方向
        cooldown: 2.0            # 冷却时间（秒）
    motion:                      # 运动检测预过滤
      enabled: true
      threshold: 40
      contour_area: 200
      frame_alpha: 0.02
    rules:                       # 检测规则
      alert_types:               # 告警类型过滤（空=全部告警）
        - intrusion
        - tripwire
        - anomaly/dwell
        - anomaly/crowd
        - anomaly/proximity
        - anomaly/fight
        - anomaly/fall
        - presence
      intrusion:
        enabled: true
        confirm_frames: 2        # 确认帧数（防误报）
        cooldown: 5.0            # 冷却时间
      tripwire:
        enabled: true
      counting:
        enabled: true
        window_seconds: 60       # 滑动窗口大小（秒）
      anomaly:
        dwell:
          enabled: true
          max_seconds: 120       # 滞留超时
        crowd:
          enabled: true
          max_count: 3           # 最大人数
          radius: 300            # 聚集半径
        proximity:
          enabled: true
          min_distance: 50       # 最小安全距离
        fight:
          enabled: true
          min_speed: 60          # 运动速度阈值
          proximity_radius: 150
        fall:
          enabled: true
          ratio_threshold: 1.0   # 宽高比阈值
```

### 模型配置

```yaml
model:
  path: yolo26m.pt               # YOLO 模型路径
  confidence: 0.3                # 置信度阈值
  analyze_fps: 5                 # 分析帧率（节流）
  classes: [0, 1, 2, 3, 5, 7]   # 检测类别（COCO: person, bicycle, car, motorcycle, bus, truck）
  pose:
    enabled: true                # 启用姿态估计（打架/跌倒增强）
    path: yolo26m-pose.pt
    confidence: 0.3
```

### 系统配置

```yaml
web:
  host: 0.0.0.0
  port: 8000

database:
  path: data/warehouse_vision.db  # SQLite 数据库路径

elasticsearch:                    # 可选，告警推送到 ES
  enabled: false
  host: http://localhost:9222
  index_prefix: warehouse-alerts

events:
  output_dir: events              # 事件截图输出目录
  draw_bbox: true                 # 截图绘制检测框
  draw_roi: true                  # 截图绘制 ROI
  draw_tripwire: true             # 截图绘制绊线
```

## 前端页面

| 页面 | 路径 | 功能 |
|------|------|------|
| 实时画面 | `/` | MJPEG 视频流，实时查看各摄像头画面（含检测框、ROI、绊线叠加） |
| 事件查询 | `/events` | 分页浏览事件记录，支持按摄像头、类型、时间范围过滤 |
| 摄像头配置 | `/config` | 添加/编辑/删除摄像头，可视化配置 ROI 和绊线 |
| 系统监控 | `/system` | 查看系统性能（FPS、检测率、跳帧率）、健康状态、系统配置 |
| 在线训练 | `/training` | 数据集管理、样本上传、标注编辑、YOLO 训练、模型发布 |

## API 接口

### 摄像头管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/cameras` | 获取所有摄像头列表 |
| GET | `/api/cameras/{id}` | 获取摄像头详细配置 |
| POST | `/api/cameras` | 添加摄像头 |
| PUT | `/api/cameras/{id}` | 更新摄像头配置 |
| DELETE | `/api/cameras/{id}` | 删除摄像头 |

### 事件与计数

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/events` | 查询事件（支持分页、过滤） |
| GET | `/api/events/summary` | 事件统计摘要 |
| GET | `/api/counts` | 获取各摄像头计数数据 |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/system/stats` | 系统性能统计 |
| GET | `/api/system/health` | 健康检查 |
| GET | `/api/system/config` | 获取系统配置 |
| PUT | `/api/system/config` | 更新系统配置（模型路径变更自动热重载） |

### 视频流

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/stream/{camera_id}` | MJPEG 实时视频流 |
| GET | `/snapshot/{camera_id}` | 单帧快照 |
| WS | `/ws/events` | WebSocket 实时事件推送 |

### 训练

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/training/datasets` | 获取数据集列表 |
| POST | `/api/training/datasets` | 创建数据集 |
| POST | `/api/training/samples/upload` | 上传样本图片 |
| GET/PUT | `/api/training/samples/{id}/annotations` | 获取/保存标注 |
| POST | `/api/training/datasets/{name}/auto-annotate` | 自动标注 |
| POST | `/api/training/jobs` | 创建训练任务 |
| GET | `/api/training/models` | 获取模型列表 |
| POST | `/api/training/models/{id}/publish` | 发布模型（热重载到检测管线） |

## 工具脚本

### ROI 点选工具

```bash
python scripts/roi_selector.py
```

鼠标左键点击添加多边形顶点，按 `s` 保存坐标，按 `q` 退出。

### 单摄像头测试

```bash
python scripts/run_single_cam.py
```

## 输出文件

| 路径 | 说明 |
|------|------|
| `data/warehouse_vision.db` | SQLite 事件数据库 |
| `events/*.jpg` | 事件截图 |
| `events/events.jsonl` | 事件日志（JSONL 格式） |
| `events/counts.jsonl` | 计数统计日志 |

## 常见问题

### ffmpeg 未安装

```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

安装方法：
- Windows: `choco install ffmpeg` 或从 https://ffmpeg.org 下载
- Linux: `apt install ffmpeg`
- macOS: `brew install ffmpeg`

### RTSP 连接失败

1. 检查摄像头 IP 和端口是否正确
2. 检查用户名密码
3. 用 VLC 测试 RTSP 地址是否可用
4. 确认网络可达：`ping <摄像头IP>`

### YOLO 模型下载失败

首次运行会自动下载模型，如网络不通可手动下载 `.pt` 文件放到项目根目录。

### 端口被占用

修改 `configs/cameras.yaml` 中 `web.port` 配置项。
