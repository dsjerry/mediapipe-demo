#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 手势隔空控制 Demo
===========================
基于 MediaPipe GestureRecognizer（Task API）实时识别手势，并把手势映射为可演示动作。

运行环境：
    - Python 3.10 ~ 3.12
    - 依赖：mediapipe, opencv-python（本地有摄像头用 opencv-python；服务器无摄像头可用 headless）
    - 模型：models/gesture_recognizer.task（首次运行脚本会自动下载，也可手动放置）

操作：
    - 摄像头实时画面上叠加手部骨架与识别到的手势。
    - 按 q 或 Esc 退出。
    - 无摄像头时可传入视频文件路径作为输入，便于离线演示与调试。

手势 -> 动作映射（可在 GESTURE_ACTION 中自行修改）：
    Open_Palm   张开手掌  -> 下一页 / 播放
    Closed_Fist 握拳      -> 上一页 / 暂停
    Pointing_Up 食指向上  -> 音量 +
    Thumb_Down  拇指向下  -> 音量 -
    Victory     剪刀手    -> 切换镜头
    ILoveYou    我爱你    -> 自定义触发
"""

import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as draw_utils
from mediapipe.tasks.python.vision import hand_landmarker

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")
MODEL_URL = "https://github.com/dsjerry/mediapipe-demo/releases/download/models-v1/gesture_recognizer.task"

# 识别置信度低于该值时不展示手势
MIN_CONFIDENCE = 0.6

# 手势类别 -> 演示动作说明（这里只做界面提示，后续可接翻页/播放器/系统控制）
GESTURE_ACTION = {
    "Open_Palm": "下一页 / 播放",
    "Closed_Fist": "上一页 / 暂停",
    "Pointing_Up": "音量 +",
    "Thumb_Down": "音量 -",
    "Victory": "切换镜头",
    "ILoveYou": "自定义触发",
}

# 中文展示名
GESTURE_CN = {
    "Open_Palm": "张开手掌",
    "Closed_Fist": "握拳",
    "Pointing_Up": "食指向上",
    "Thumb_Up": "拇指向上",
    "Thumb_Down": "拇指向下",
    "Victory": "剪刀手",
    "ILoveYou": "我爱你",
    "None": "未识别",
}


def ensure_model() -> str:
    """模型不存在时自动下载。"""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"[模型] 未找到模型，正在下载到 {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[模型] 下载完成。")
    return MODEL_PATH


def draw_landmarks(frame, hand_landmarks):
    """在画面上绘制手部骨架。"""
    # mediapipe 新版已移除 mp.solutions，改用 tasks.vision 自带的绘制工具与连接常量
    draw_utils.draw_landmarks(
        frame,
        hand_landmarks,
        hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
        draw_utils.DrawingSpec(color=(0, 200, 120), thickness=2, circle_radius=3),
        draw_utils.DrawingSpec(color=(255, 255, 255), thickness=1),
    )


FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_LARGE = ImageFont.truetype(FONT_PATH, 36)
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 32)


def put_text(frame, text, pos, font, color_bgr):
    """用 Pillow 绘制文字（cv2.putText 不支持中文）。"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=font, fill=color_bgr[::-1])
    frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def recognize(recognizer, frame, timestamp_ms):
    """把 BGR 帧转为 MediaPipe Image 并识别，返回结果。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return recognizer.recognize_for_video(mp_image, timestamp_ms)


def main():
    # 输入源：命令行传入视频文件，否则默认摄像头（index 0）
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[错误] 无法打开输入源：{source}")
        sys.exit(1)

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    print("[提示] 摄像头已启动，按 q 或 Esc 退出。")
    prev_action = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[提示] 视频读取结束。")
            break

        frame = cv2.flip(frame, 1)  # 镜像，符合自拍习惯
        timestamp_ms = int(time.time() * 1000)
        result = recognize(recognizer, frame, timestamp_ms)

        gesture_label = "None"
        action_text = ""

        # 遍历检测到的手
        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                draw_landmarks(frame, hand_lm)

        if result.gestures:
            # 取置信度最高的一组手势
            top = result.gestures[0]
            if top and top[0].score >= MIN_CONFIDENCE:
                gesture_label = top[0].category_name
                action_text = GESTURE_ACTION.get(gesture_label, "未绑定动作")

        # 界面提示
        cn = GESTURE_CN.get(gesture_label, gesture_label)
        put_text(frame, f"手势: {cn}", (20, 10), FONT_LARGE, (0, 200, 120))
        if action_text:
            put_text(frame, f"动作: {action_text}", (20, 55), FONT_MEDIUM, (255, 160, 0))

        # 动作触发（演示阶段只打印；后续可接翻页/播放器/系统快捷键）
        if gesture_label != "None" and gesture_label != prev_action:
            print(f"[触发] {cn} -> {action_text}")
            prev_action = gesture_label
        elif gesture_label == "None":
            prev_action = None

        cv2.imshow("MediaPipe Gesture Control Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
