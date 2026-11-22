#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 人脸注意力分析 Demo
============================
基于 MediaPipe FaceLandmarker（Task API）检测人脸，借助头部姿态变换矩阵
解算偏航（yaw）、俯仰（pitch）、翻滚（roll），判断是否面向屏幕，
并汇总注意力（面向屏幕）占比与偏离次数。

隐私说明：
    - 全程端侧本地推理，不上传、不保存原始画面。
    - 仅输出朝向状态与统计指标，不做人脸身份识别、不做情绪识别。

运行环境：
    - Python 3.10 ~ 3.12
    - 依赖：mediapipe, opencv-python（本地有摄像头）/ opencv-python-headless（无显示）
    - 模型：models/face_landmarker.task（缺失时脚本自动下载）

操作：
    - 默认摄像头；也可传入视频文件路径做离线调试。
    - 按 q 或 Esc 退出。
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

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = "https://github.com/dsjerry/mediapipe-demo/releases/download/models-v1/face_landmarker.task"

# 朝向判定阈值（度）：超出范围视为未看向屏幕
YAW_THRESHOLD = 25.0      # 左右转头（偏航）
PITCH_THRESHOLD = 25.0    # 抬头/低头（俯仰）
ROLL_THRESHOLD = 20.0     # 左右歪头（翻滚）

SMOOTH_FRAMES = 5         # 滑动平滑窗口
MIN_FACE_CONFIDENCE = 0.5


FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_LARGE = ImageFont.truetype(FONT_PATH, 36)
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 28)


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


def rotation_matrix_to_euler(R):
    """
    从 3x3 旋转矩阵解算欧拉角（度）。
    返回 (pitch, yaw, roll)，对应俯仰、偏航、翻滚。
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll),
    )


def head_pose_from_matrix(matrix):
    """
    4x4 头部变换矩阵 -> 欧拉角。
    MediaPipe 给出的矩阵为列优先 flattened 列表或嵌套结构。
    """
    arr = np.array(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(4, 4)
    R = arr[:3, :3]
    return rotation_matrix_to_euler(R)


def judge_attention(pitch, yaw, roll):
    """根据欧拉角判断是否面向屏幕，返回 (状态, 偏离原因列表)。"""
    reasons = []
    if abs(yaw) > YAW_THRESHOLD:
        reasons.append("左右转头")
    if abs(pitch) > PITCH_THRESHOLD:
        reasons.append("抬头/低头")
    if abs(roll) > ROLL_THRESHOLD:
        reasons.append("歪头")

    if reasons:
        return "未面向屏幕", reasons
    return "面向屏幕", []


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[错误] 无法打开输入源：{source}")
        sys.exit(1)

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=MIN_FACE_CONFIDENCE,
        min_face_presence_confidence=MIN_FACE_CONFIDENCE,
        min_tracking_confidence=MIN_FACE_CONFIDENCE,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    status_history = deque(maxlen=SMOOTH_FRAMES)
    total_frames = 0
    focused_frames = 0
    off_count = 0
    prev_focused = True

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
        reasons = []
        pitch = yaw = roll = 0.0
        faces = 0

        if result.face_landmarks:
            faces = len(result.face_landmarks)
            # 取第一张脸做演示；多人场景可遍历统计
            if result.facial_transformation_matrixes:
                pitch, yaw, roll = head_pose_from_matrix(
                    result.facial_transformation_matrixes[0]
                )
                status, reasons = judge_attention(pitch, yaw, roll)

        # 平滑
        status_history.append(status)
        smoothed = max(set(status_history), key=status_history.count)

        total_frames += 1
        focused = smoothed == "面向屏幕"
        if focused:
            focused_frames += 1
        if not focused and prev_focused:
            off_count += 1
        prev_focused = focused

        focus_rate = (focused_frames / total_frames * 100) if total_frames else 0

        # 界面渲染
        color = (0, 200, 120) if focused else (0, 0, 255)
        put_text(frame, f"状态: {smoothed}", (20, 10), FONT_LARGE, color)
        if reasons:
            put_text(frame, "偏离: " + "、".join(reasons), (20, 55), FONT_MEDIUM, (0, 0, 255))
        put_text(frame, f"人数: {faces}  专注率: {focus_rate:.1f}%  偏离次数: {off_count}",
                 (20, 95), FONT_MEDIUM, (255, 160, 0))
        cv2.putText(frame, f"pitch {pitch:.1f}  yaw {yaw:.1f}  roll {roll:.1f}",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Attention Analysis Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    print(f"\n[汇总] 总帧数 {total_frames}，面向屏幕帧 {focused_frames}，"
          f"专注率 {focus_rate:.1f}%，偏离 {off_count} 次")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
