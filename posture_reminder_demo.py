#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 姿态坐姿提醒 Demo
==========================
基于 MediaPipe PoseLandmarker（Task API）实时提取人体 33 个关键点，
用规则判断含胸驼背、头部前倾、身体歪斜等不良坐姿，并给出提示与统计。

运行环境：
    - Python 3.10 ~ 3.12
    - 依赖：mediapipe, opencv-python（本地有摄像头）/ opencv-python-headless（无显示）
    - 模型：models/pose_landmarker.task（缺失时脚本自动下载）

操作：
    - 默认摄像头；也可传入视频文件路径做离线调试。
    - 按 q 或 Esc 退出。

判断思路（基于归一化关键点坐标，规则可在阈值区调整）：
    - 前倾/含胸：鼻子相对肩膀连线中点明显靠前（y 更小，即更靠近画面上方之外的前倾投影），
      以及肩-髋躯干垂线倾斜角过大。
    - 驼背：肩膀连线中点相对髋部中点向前偏移、躯干弯曲。
    - 歪斜：左右肩高度差过大（身体侧向倾斜）。
    - 头部前倾：鼻子相对肩中点的水平/垂直偏移超出阈值。

说明：本期为规则化 Demo，阈值为经验默认值；正式使用前应按个人坐姿基准校准。
"""

import os
import sys
import time
import math
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as draw_utils
from mediapipe.tasks.python.vision import pose_landmarker

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker.task")
MODEL_URL = "https://github.com/dsjerry/mediapipe-demo/releases/download/models-v1/pose_landmarker.task"

# 关键点索引（MediaPipe Pose 33 点）
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_EAR, RIGHT_EAR = 7, 8

# 姿态置信度低于该值不参与判断
MIN_POSE_CONFIDENCE = 0.5

# 规则阈值（归一化坐标，按实际画面与坐姿校准）
SHOULDER_TILT_MAX = 0.12      # 左右肩 y 差超过此值视为歪斜
TRUNK_ANGLE_MAX = 25.0         # 躯干相对垂线倾角（度）超过视为前倾/含胸
HEAD_FORWARD_X = 0.10          # 鼻尖相对肩中点水平偏移阈值
HEAD_DROP_Y = 0.10             # 鼻尖低于肩中点（低头）超过此值阈值
SMOOTH_FRAMES = 5              # 滑动平滑窗口

# 状态文案
STATUS_GOOD = "坐姿良好"
STATUS_BAD = "坐姿异常"

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_LARGE = ImageFont.truetype(FONT_PATH, 36)
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 32)
FONT_SMALL = ImageFont.truetype(FONT_PATH, 22)


def put_text(frame, text, pos, font, color_bgr):
    """用 Pillow 绘制文字（cv2.putText 不支持中文）。"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=font, fill=color_bgr[::-1])
    frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"[模型] 未找到模型，正在下载到 {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[模型] 下载完成。")
    return MODEL_PATH


def angle_between(v1, v2):
    """计算两个二维向量夹角（度）。"""
    dx1, dy1 = v1
    dx2, dy2 = v2
    norm1 = math.hypot(dx1, dy1)
    norm2 = math.hypot(dx2, dy2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2)))
    return math.degrees(math.acos(cos))


def analyze_posture(landmarks):
    """
    输入归一化关键点列表，返回 (状态, 问题列表, 关键指标字典)。
    landmarks: list of (x, y, visibility) 风格的关键点对象。
    """
    def lm(idx):
        p = landmarks[idx]
        return p.x, p.y, getattr(p, "visibility", 1.0)

    nose = lm(NOSE)
    ls = lm(LEFT_SHOULDER)
    rs = lm(RIGHT_SHOULDER)
    lh = lm(LEFT_HIP)
    rh = lm(RIGHT_HIP)

    # 关键点可见性不足，不判断
    for p in (nose, ls, rs, lh, rh):
        if p[2] < MIN_POSE_CONFIDENCE:
            return "未检测到可靠姿态", [], {}

    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    problems = []

    # 1. 身体歪斜：左右肩高度差
    shoulder_diff = abs(ls[1] - rs[1])
    if shoulder_diff > SHOULDER_TILT_MAX:
        problems.append("身体歪斜")

    # 2. 躯干倾角：肩中点 -> 髋中点 向量，相对竖直向下方向的夹角
    trunk_vec = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
    trunk_angle = angle_between(trunk_vec, (0, 1))  # 竖直向下参考
    if trunk_angle > TRUNK_ANGLE_MAX:
        problems.append("含胸/前倾")

    # 3. 头部前倾/低头：鼻尖相对肩中点左右偏移过大，或鼻尖下沉到肩线以下过多
    head_dx = nose[0] - shoulder_mid[0]
    head_dy = nose[1] - shoulder_mid[1]  # 正值表示鼻尖低于肩中点（低头）
    if abs(head_dx) > HEAD_FORWARD_X or head_dy > HEAD_DROP_Y:
        problems.append("头部前倾/低头")

    metrics = {
        "肩高差": shoulder_diff,
        "躯干倾角": trunk_angle,
        "头部偏移x": head_dx,
        "头部偏移y": head_dy,
    }

    status = STATUS_BAD if problems else STATUS_GOOD
    return status, problems, metrics


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[错误] 无法打开输入源：{source}")
        sys.exit(1)

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    connections = pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS

    # 统计与平滑
    status_history = deque(maxlen=SMOOTH_FRAMES)
    total_frames = 0
    good_frames = 0
    last_status = "检测中"

    print("[提示] 摄像头已启动，按 q 或 Esc 退出。")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[提示] 视频读取结束。")
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        status = "未检测到人"
        problems = []
        metrics = {}

        if result.pose_landmarks:
            pose_lm = result.pose_landmarks[0]
            draw_utils.draw_landmarks(
                frame, pose_lm, connections,
                draw_utils.DrawingSpec(color=(0, 200, 120), thickness=2, circle_radius=3),
                draw_utils.DrawingSpec(color=(255, 255, 255), thickness=1),
            )
            status, problems, metrics = analyze_posture(pose_lm)

        # 滑动平滑：多数帧为良好才判良好，避免单帧抖动
        status_history.append(status)
        smoothed = max(set(status_history), key=status_history.count)
        last_status = smoothed

        total_frames += 1
        if smoothed == STATUS_GOOD:
            good_frames += 1

        good_rate = (good_frames / total_frames * 100) if total_frames else 0

        # 界面渲染
        color = (0, 200, 120) if smoothed == STATUS_GOOD else (0, 0, 255)
        put_text(frame, f"状态: {smoothed}", (20, 10), FONT_LARGE, color)
        if problems:
            put_text(frame, "问题: " + "、".join(problems), (20, 55), FONT_MEDIUM, (0, 0, 255))
        put_text(frame, f"坐姿良好率: {good_rate:.1f}%", (20, 95), FONT_MEDIUM, (255, 160, 0))

        if metrics:
            info = "  ".join(f"{k}: {v:.2f}" for k, v in metrics.items())
            put_text(frame, info, (20, frame.shape[0] - 35), FONT_SMALL, (180, 180, 180))

        cv2.imshow("Posture Reminder Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    # 结束后输出汇总
    print(f"\n[汇总] 总帧数 {total_frames}，良好帧 {good_frames}，良好率 {good_rate:.1f}%")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
