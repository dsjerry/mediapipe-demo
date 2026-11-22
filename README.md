# MediaPipe 端侧实时视觉 Demo 合集

基于谷歌开源 MediaPipe（Task API）的端侧实时视觉 Demo，所有推理均在本地完成，不上传摄像头画面。包含三个独立脚本，分别对应创新提案中的三个模块：

| 模块 | 脚本 | 说明 | 详细文档 |
| --- | --- | --- | --- |
| 手势隔空控制 | `gesture_control_demo.py` | 识别手势并映射为演示动作 | [README_手势Demo.md](README_手势Demo.md) |
| 姿态坐姿提醒 | `posture_reminder_demo.py` | 基于关键点规则判断含胸、前倾、歪斜等不良坐姿 | [README_坐姿Demo.md](README_坐姿Demo.md) |
| 人脸与注意力分析 | `attention_analysis_demo.py` | 解算头部姿态，判断是否面向屏幕并统计专注率 | [README_注意力Demo.md](README_注意力Demo.md) |

## 环境要求

- Python 3.10 ~ 3.12
- 依赖安装：

```bash
pip install -r requirements.txt
```

- 界面中文文字使用 Windows 自带的微软雅黑字体（`C:\Windows\Fonts\msyh.ttc`），因此目前脚本仅在 Windows 上测试。在其他系统运行时，需要修改脚本中的 `FONT_PATH`。

## 运行方式

```bash
python gesture_control_demo.py
python posture_reminder_demo.py
python attention_analysis_demo.py
```

- 默认调用本机摄像头；也可以传入视频文件路径做离线演示，例如 `python posture_reminder_demo.py demo.mp4`。
- 按 `q` 或 `Esc` 退出。
- 模型文件在首次运行时会自动下载，下载地址见下方「模型文件」。

## 模型文件

运行脚本时，如果 `models/` 目录下缺少对应模型，脚本会自动下载。也可以手动下载后放到项目的 `models/` 目录下：

| 模型文件 | 对应脚本 | 下载地址 |
| --- | --- | --- |
| `gesture_recognizer.task` | `gesture_control_demo.py` | [下载](https://github.com/dsjerry/mediapipe-demo/releases/download/models-v1/gesture_recognizer.task) |
| `pose_landmarker.task` | `posture_reminder_demo.py` | [下载](https://github.com/dsjerry/mediapipe-demo/releases/download/models-v1/pose_landmarker.task) |
| `face_landmarker.task` | `attention_analysis_demo.py` | [下载](https://github.com/dsjerry/mediapipe-demo/releases/download/models-v1/face_landmarker.task) |

所有模型也可以在 Release 页面统一下载：https://github.com/dsjerry/mediapipe-demo/releases/tag/models-v1

## 目录结构

```
.
├── gesture_control_demo.py      # 手势隔空控制
├── posture_reminder_demo.py     # 姿态坐姿提醒
├── attention_analysis_demo.py   # 人脸与注意力分析
├── requirements.txt
├── README_手势Demo.md
├── README_坐姿Demo.md
└── README_注意力Demo.md
```

## 说明

- 本项目为演示用途，规则阈值为经验默认值，正式使用前请按实际摄像头角度与坐姿进行校准。
- 注意力分析仅输出朝向状态与统计指标，不做人脸身份识别或情绪识别。
