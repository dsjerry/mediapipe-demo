# MediaPipe 手势隔空控制 Demo

基于谷歌开源 MediaPipe 的端侧实时手势识别 Demo，对应创新提案中的「手势隔空控制」模块。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `gesture_control_demo.py` | 主程序：摄像头采集 → 手势识别 → 骨架可视化 → 动作提示 |
| `models/gesture_recognizer.task` | 手势识别模型（脚本首次运行缺失时会自动下载） |

## 环境要求

- Python 3.10 ~ 3.12
- 依赖安装：

```bash
pip install mediapipe opencv-python numpy
```

> 纯服务器、无显示环境装 `opencv-python-headless`；本地笔记本有摄像头和窗口界面时装 `opencv-python`。

## 运行方式

```bash
# 默认调用本机摄像头
python gesture_control_demo.py

# 或传入一段视频文件做离线演示/调试
python gesture_control_demo.py demo.mp4
```

- 画面为镜像自拍视角，叠加手部骨架。
- 按 `q` 或 `Esc` 退出。

## 手势与动作映射

识别结果来自模型内置手势分类，脚本在 `GESTURE_ACTION` 中映射为演示动作：

| 手势 | 类别名 | 演示动作 |
| --- | --- | --- |
| 张开手掌 | Open_Palm | 下一页 / 播放 |
| 握拳 | Closed_Fist | 上一页 / 暂停 |
| 食指向上 | Pointing_Up | 音量 + |
| 拇指向下 | Thumb_Down | 音量 - |
| 剪刀手 | Victory | 切换镜头 |
| 我爱你 | ILoveYou | 自定义触发 |

当前阶段只在界面上提示动作并在终端打印触发日志；后续可把触发回调接到翻页、播放器控制、系统快捷键或业务页面。

## 可调参数

- `MIN_CONFIDENCE`：手势置信度阈值，默认 0.6，调低更灵敏、调高更稳定。
- `num_hands`：同时识别的手数量，默认 2。
- `GESTURE_ACTION` / `GESTURE_CN`：动作映射与中文展示名。

## 已验证情况

- 云主机环境：mediapipe 1.2.3、opencv 5.1.0，模型加载与推理链路通过；云主机无摄像头，实时画面需在本地电脑运行。
- 识别鲁棒性受光线、手部遮挡、距离影响；正式演示前请在充足光线下、手距镜头 30~60cm 测试。

## 后续可扩展

- 接入翻页/媒体控制等真实动作回调。
- 增加手势连续帧防抖与触发冷却，避免一次手势重复触发。
- 迁移到浏览器端（`@mediapipe/tasks-vision`）或桌面端打包。
