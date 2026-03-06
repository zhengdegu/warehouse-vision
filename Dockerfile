# ── Stage 1: 构建前端 ──
FROM node:22-alpine AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ .
RUN npm run build

# ── Stage 2: 下载 go2rtc 二进制 ──
FROM debian:bookworm-slim AS go2rtc
ARG TARGETARCH
ARG GO2RTC_VERSION=1.9.14
ADD --chmod=755 "https://github.com/AlexxIT/go2rtc/releases/download/v${GO2RTC_VERSION}/go2rtc_linux_${TARGETARCH}" /usr/local/bin/go2rtc

# ── Stage 3: Python 后端 (GPU via PyTorch CUDA) ──
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# go2rtc 二进制
COPY --from=go2rtc /usr/local/bin/go2rtc /usr/local/bin/go2rtc

# PyTorch GPU 版 — pip 包自带 CUDA 运行时，无需 nvidia 基础镜像
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 源码
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# 模型文件
COPY yolo26m.pt .
COPY yolo26m-pose.pt .

# 前端构建产物
COPY --from=frontend /web/dist web/dist

# 数据目录
RUN mkdir -p events data/samples data/labels data/models data/training data/metadata

EXPOSE 8000 1984 8555

# NVIDIA Container Runtime 环境变量
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "scripts/run_all.py"]
