"""
GestureWar - 手部追踪模块 (MediaPipe 0.10.35+ 重构版)
基于 MediaPipe Tasks API 和 OpenCV 的实时手部关键点检测

重构说明:
  - 旧版 mp.solutions API 在 0.10.30+ 已移除
  - 新版使用 mediapipe.tasks.python.vision.HandLandmarker
  - 数据结构、UDP 发送保持与旧版兼容
  - 集成手势分类器 (gesture_classifier.py)
"""

import cv2
import numpy as np
import time
import sys
import json
import socket
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode
)
from mediapipe import Image, ImageFormat

# 手势分类器
from gesture_classifier import GestureClassifier, PerKeypointKalmanSmoother


# ---------------------------------------------------------------------------
# 绘制工具 (替代旧版 mp.solutions.drawing_utils)
# ---------------------------------------------------------------------------

# 新版 MediaPipe 不再提供内置绘图函数，自己画

HAND_CONNECTION_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17),              # 手掌横向
]

LANDMARK_COLOR = (0, 255, 0)       # 关键点: 绿色
CONNECTION_COLOR = (255, 255, 255) # 连接线: 白色


def draw_landmarks_on_image(rgb_image, hand_landmarks_list):
    """
    在图像上绘制手部关键点和骨骼连接线。
    直接修改传入的图像 (in-place)。

    参数:
        rgb_image: RGB 格式的 numpy 图像
        hand_landmarks_list: HandLandmarkerResult.hand_landmarks
    """
    h, w, _ = rgb_image.shape

    for hand_landmarks in hand_landmarks_list:
        # 收集像素坐标
        points = []
        for lm in hand_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))

        # 画连接线
        for start_idx, end_idx in HAND_CONNECTION_PAIRS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(rgb_image, points[start_idx], points[end_idx],
                         CONNECTION_COLOR, 2, cv2.LINE_AA)

        # 画关键点
        for px, py in points:
            cv2.circle(rgb_image, (px, py), 4, LANDMARK_COLOR, cv2.FILLED)


# ---------------------------------------------------------------------------
# HandTracker - 使用 MediaPipe Tasks API
# ---------------------------------------------------------------------------

class HandTracker:
    """手部追踪器类 - 适配 MediaPipe 0.10.35+ Tasks API"""

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5,
                 enable_gesture=True, use_kalman=False, use_per_keypoint_kalman=False):
        """
        初始化手部追踪器

        参数:
            max_hands: 最大检测手部数量
            detection_confidence: 检测置信度阈值
            tracking_confidence: 追踪置信度阈值
            enable_gesture: 是否启用手势分类
            use_kalman: 是否使用 Kalman 滤波 (默认 EMA)
            use_per_keypoint_kalman: 是否使用坐标级 Kalman (默认 False)
        """
        print("正在初始化手部追踪器 (MediaPipe 0.10.35+ Tasks API)...")

        # 构建选项
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path='hand_landmarker.task'
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # 手部关键点名称映射 (MediaPipe 标准 21 点)
        self.hand_landmark_names = {
            0: "WRIST",
            1: "THUMB_CMC",
            2: "THUMB_MCP",
            3: "THUMB_IP",
            4: "THUMB_TIP",
            5: "INDEX_FINGER_MCP",
            6: "INDEX_FINGER_PIP",
            7: "INDEX_FINGER_DIP",
            8: "INDEX_FINGER_TIP",
            9: "MIDDLE_FINGER_MCP",
            10: "MIDDLE_FINGER_PIP",
            11: "MIDDLE_FINGER_DIP",
            12: "MIDDLE_FINGER_TIP",
            13: "RING_FINGER_MCP",
            14: "RING_FINGER_PIP",
            15: "RING_FINGER_DIP",
            16: "RING_FINGER_TIP",
            17: "PINKY_MCP",
            18: "PINKY_PIP",
            19: "PINKY_DIP",
            20: "PINKY_TIP"
        }

        # 手势分类器
        self.enable_gesture = enable_gesture
        self.use_kalman = use_kalman
        self.use_per_keypoint_kalman = use_per_keypoint_kalman
        self.gesture_classifier = GestureClassifier(
            use_kalman=use_kalman,
            use_per_keypoint_kalman=use_per_keypoint_kalman,
        ) if enable_gesture else None

        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.frame_id = 0          # 全局递增帧序号 (用于 UDP 丢包检测)
        self.start_time = time.time()

        print("手部追踪器初始化完成！")

    def process_frame(self, frame):
        """
        处理单帧图像

        参数:
            frame: OpenCV 图像帧 (BGR格式)

        返回:
            processed_frame: 处理后的图像帧 (BGR)
            hand_data: 手部关键点数据字典 (与旧版格式兼容)
        """
        # 镜像处理，更符合交互直觉
        frame = cv2.flip(frame, 1)

        # 转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建 MediaPipe Image
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

        # 运行手部检测
        result = self.landmarker.detect(mp_image)

        # 初始化手部数据 (格式与旧版保持兼容)
        self.frame_id += 1
        hand_data = {
            "timestamp": time.time(),    # Unix 秒, 毫秒精度 — 用于延迟测量
            "frame_id": self.frame_id,   # 递增序号 — 用于丢包检测
            "num_hands": 0,
            "hands": []
        }

        h, w, _ = frame.shape

        if result.hand_landmarks:
            hand_data["num_hands"] = len(result.hand_landmarks)

            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                # 收集手部关键点数据
                hand_info = {
                    "id": hand_idx,
                    "landmarks": [],
                    "bounding_box": None
                }

                x_coords = []
                y_coords = []

                for landmark in hand_landmarks:
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    x_coords.append(px)
                    y_coords.append(py)

                    # 保存关键点信息 (与旧版格式一致)
                    hand_info["landmarks"].append({
                        "x": landmark.x,     # 归一化坐标 (0-1)
                        "y": landmark.y,
                        "z": landmark.z,
                        "pixel_x": px,       # 像素坐标
                        "pixel_y": py
                    })

                # 计算边界框
                if x_coords and y_coords:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    hand_info["bounding_box"] = {
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "width": x_max - x_min,
                        "height": y_max - y_min
                    }

                    # 绘制边界框
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                  (0, 255, 0), 2)

                # 手势分类
                if self.enable_gesture and self.gesture_classifier:
                    gesture_name, gesture_conf = self.gesture_classifier.classify(hand_landmarks)
                    hand_info["gesture"] = gesture_name
                    hand_info["gesture_confidence"] = gesture_conf

                # 高亮显示食指尖端 (索引 8)
                if len(hand_landmarks) > 8:
                    index_tip = hand_landmarks[8]
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                hand_data["hands"].append(hand_info)

            # 绘制手部关键点和连接线
            draw_landmarks_on_image(frame, result.hand_landmarks)

        # 计算 FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

        # 显示性能信息
        cv2.putText(frame, f'FPS: {int(self.fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(frame, f'Hands: {hand_data["num_hands"]}', (10, 140),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # 显示手势识别结果 (在画面上绘制手势名称)
        # 手势颜色映射
        gesture_colors = {
            "shoot": (0, 165, 255),         # 橙色
            "aim": (255, 255, 0),            # 青色
            "grenade": (0, 0, 255),          # 红色
            "reload": (255, 0, 255),         # 品红
            "melee": (0, 255, 255),          # 黄色
            "switch_weapon": (255, 128, 0),  # 浅蓝
        }
        gesture_labels_cn = {
            "shoot": "射击",
            "aim": "瞄准",
            "grenade": "手雷",
            "reload": "换弹",
            "melee": "近战",
            "switch_weapon": "切换武器",
            "none": "",
        }

        for hand in hand_data["hands"]:
            gesture = hand.get("gesture", "none")
            confidence = hand.get("gesture_confidence", 0)
            if gesture == "none" or confidence < 0.3:
                continue

            bbox = hand.get("bounding_box")
            if bbox:
                label = f"{gesture_labels_cn.get(gesture, gesture)} ({confidence:.0%})"
                color = gesture_colors.get(gesture, (0, 255, 0))
                # 在边界框上方绘制手势标签
                label_y = bbox["y_min"] - 15
                if label_y < 25:
                    label_y = bbox["y_max"] + 25
                cv2.putText(frame, label,
                            (bbox["x_min"], label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 绘制时序反馈面板 (张开度条、速度、状态机)
        if self.enable_gesture and self.gesture_classifier and hand_data["hands"]:
            self._draw_feedback_panel(frame, hand_data)

        return frame, hand_data

    def _draw_feedback_panel(self, frame, hand_data):
        """
        在画面底部绘制调试面板:
          - 张开度条 (带低/高阈值线)
          - 手腕瞬时速度
          - 换弹状态机状态
        """
        clf = self.gesture_classifier
        fb = clf.get_feedback()
        h, w = frame.shape[:2]

        # 面板参数
        bar_w = 160
        bar_h = 10
        panel_x = 10
        panel_y = h - 100

        openness = fb["openness"]
        vel = fb["velocity"]
        inst_vel = fb["instant_velocity"]
        rl_state = fb["reload_state"]

        # ---- 张开度条 ----
        bar_x, bar_y = panel_x, panel_y
        # 背景
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)
        # 填充 (颜色随张开度变化: 红→黄→绿)
        fill_w = int(bar_w * min(openness, 1.0))
        if openness <= clf.openness_low:
            bar_color = (0, 80, 255)       # 红 = 握拳
        elif openness >= clf.openness_high:
            bar_color = (0, 220, 80)       # 绿 = 张开
        else:
            bar_color = (0, 200, 220)      # 黄 = 过渡
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        # 边框
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (180, 180, 180), 1)
        # 阈值标记线
        lo_x = bar_x + int(bar_w * clf.openness_low)
        hi_x = bar_x + int(bar_w * clf.openness_high)
        cv2.line(frame, (lo_x, bar_y - 3), (lo_x, bar_y + bar_h + 3),
                 (100, 100, 255), 1)
        cv2.line(frame, (hi_x, bar_y - 3), (hi_x, bar_y + bar_h + 3),
                 (100, 255, 100), 1)

        cv2.putText(frame, f"Openness: {openness:.0%}",
                    (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # ---- 速度显示 ----
        line2_y = bar_y + bar_h + 18
        peak_vel = fb.get("peak_velocity", 0)
        vel_color = (0, 255, 200) if peak_vel > clf.melee_velocity * 0.6 else (180, 180, 180)
        cv2.putText(frame,
                    f"Vel: {vel:.2f}  |  Inst: {inst_vel:.2f}  |  Peak: {peak_vel:.2f}  (thresh: {clf.melee_velocity})",
                    (bar_x, line2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, vel_color, 1)

        # ---- 状态机状态 ----
        line3_y = line2_y + 16
        state_names = {"idle": "待机", "closed": "握拳", "open": "张开"}
        state_cn = state_names.get(rl_state, rl_state)
        if rl_state != "idle":
            state_color = (0, 255, 255)
            state_text = f">>> Reload: {state_cn} <<<"
        else:
            state_color = (140, 140, 140)
            state_text = f"Reload: {state_cn}"
        cv2.putText(frame, state_text, (bar_x, line3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1)

        # ---- 平滑器类型 ----
        line4_y = line3_y + 16
        smoother_info = clf.get_smoother_info()
        if smoother_info["type"] == "PerKeypointKalman":
            s_text = "Smoother: PK-KF (21pt grouped q)"
            s_color = (100, 200, 255)  # 淡蓝 = 坐标级 Kalman
        elif smoother_info["type"] == "Kalman":
            s_text = f"Smoother: KF (q={clf._kalman_q}, r={clf._kalman_r})"
            s_color = (255, 200, 100)  # 金色 = Kalman
        else:
            s_text = f"Smoother: EMA (a_o={clf._ema_alphas[0]}, a_w={clf._ema_alphas[1]})"
            s_color = (150, 200, 150)  # 淡绿 = EMA
        cv2.putText(frame, s_text, (bar_x, line4_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, s_color, 1)

    def release(self):
        """释放资源"""
        print("释放手部追踪器资源...")
        self.landmarker.close()
        print("资源释放完成！")


# ---------------------------------------------------------------------------
# UDPSender - 保持不变
# ---------------------------------------------------------------------------

class UDPSender:
    """UDP 数据发送器"""

    def __init__(self, host='127.0.0.1', port=5005):
        """
        初始化 UDP 发送器

        参数:
            host: 目标主机地址 (默认: 127.0.0.1)
            port: 目标端口号 (默认: 5005)
        """
        self.host = host
        self.port = port

        # 创建 UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        print(f"UDP 发送器初始化完成，目标: {host}:{port}")

    def send_hand_data(self, hand_data):
        """
        发送手部数据

        参数:
            hand_data: 手部关键点数据字典
        """
        try:
            # 转换为 JSON 字符串
            data_str = json.dumps(hand_data)

            # 发送数据
            self.sock.sendto(data_str.encode('utf-8'), (self.host, self.port))

            # 调试信息 (含手势)
            gestures = []
            for hand in hand_data.get("hands", []):
                g = hand.get("gesture", "none")
                if g != "none":
                    gestures.append(f"{g}({hand.get('gesture_confidence', 0):.0%})")
            gesture_str = ", ".join(gestures) if gestures else "无手势"
            print(f"发送数据: {hand_data['num_hands']} 只手 | 手势: {gesture_str}")

        except Exception as e:
            print(f"发送数据时出错: {e}")

    def close(self):
        """关闭连接"""
        self.sock.close()
        print("UDP 发送器已关闭")


# ---------------------------------------------------------------------------
# main - 保持不变
# ---------------------------------------------------------------------------

def main():
    """主函数 - 完整的手部追踪 + UDP 发送程序"""
    print("=" * 60)
    print("GestureWar - 手部追踪 + UDP 发送 (MediaPipe 0.10.35+)")
    print("=" * 60)
    print("功能说明:")
    print("1. 实时摄像头手部关键点检测")
    print("2. 绘制手部骨架和边界框")
    print("3. 手势分类 (射击/瞄准/手雷/换弹/近战/切换武器)")
    print("4. 通过 UDP 发送手部数据到 Unity")
    print("5. 显示实时 FPS、手部数量和手势")
    print("")
    print("控制说明:")
    print("  [q] 退出程序")
    print("  [s] 保存当前帧为图片")
    print("  [i] 显示详细信息")
    print("  [u] 切换 UDP 发送 (开/关)")
    print("  [k] 三模式循环: EMA → Kalman(特征) → PK-KF(坐标级)")
    print("  [p] 直接切换坐标级 Kalman 开/关")
    print("=" * 60)

    # 初始化手部追踪器
    tracker = HandTracker(
        max_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.5
    )

    # 初始化 UDP 发送器
    sender = UDPSender(host='127.0.0.1', port=5005)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        sys.exit(1)

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("摄像头已打开，开始手部追踪...")
    print("UDP 数据发送到: 127.0.0.1:5005")
    print("")

    # 状态变量
    udp_enabled = True
    frame_save_counter = 0
    last_debug_time = time.time()

    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("警告: 无法读取摄像头帧")
                break

            # 处理帧
            processed_frame, hand_data = tracker.process_frame(frame)

            # 发送手部数据 (如果启用)
            if udp_enabled:
                sender.send_hand_data(hand_data)

            # 显示手部信息
            if hand_data["num_hands"] > 0:
                for hand in hand_data["hands"]:
                    if hand["bounding_box"]:
                        bbox = hand["bounding_box"]
                        info_text = f"Hand {hand['id']}: {bbox['width']}x{bbox['height']}"
                        cv2.putText(processed_frame, info_text,
                                    (bbox['x_min'], bbox['y_min'] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示 UDP 状态
            udp_status = "ON" if udp_enabled else "OFF"
            cv2.putText(processed_frame, f"UDP: {udp_status}", (10, 210),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            # 显示窗口
            cv2.imshow('GestureWar - Hand Tracking (0.10.35)', processed_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户请求退出...")
                break
            elif key == ord('s'):
                # 保存当前帧
                filename = f"hand_tracking_frame_{frame_save_counter:04d}.png"
                cv2.imwrite(filename, processed_frame)
                print(f"已保存帧到: {filename}")
                frame_save_counter += 1
            elif key == ord('i'):
                # 显示详细信息
                current_time = time.time()
                if current_time - last_debug_time > 0.5:  # 防抖动
                    print("=" * 40)
                    print(f"当前帧: {tracker.frame_count}")
                    print(f"FPS: {tracker.fps:.1f}")
                    print(f"检测到手部数量: {hand_data['num_hands']}")
                    for hand in hand_data.get("hands", []):
                        g = hand.get("gesture", "none")
                        gc = hand.get("gesture_confidence", 0)
                        print(f"  手 {hand['id']}: 手势={g} ({gc:.0%})")
                    print(f"UDP 发送: {'启用' if udp_enabled else '禁用'}")
                    # 分类器调试信息
                    if tracker.gesture_classifier:
                        di = tracker.gesture_classifier.debug_info
                        if di:
                            print(f"分类器调试: {di}")
                        si = tracker.gesture_classifier.get_smoother_info()
                        print(f"平滑器: {si['type']} | 参数: {si}")
                    last_debug_time = current_time
            elif key == ord('u'):
                # 切换 UDP 发送
                udp_enabled = not udp_enabled
                status = "启用" if udp_enabled else "禁用"
                print(f"UDP 发送已{status}")
            elif key == ord('k'):
                # 三模式循环: EMA → Kalman(特征) → PerKeypointKalman → EMA → ...
                if tracker.gesture_classifier:
                    clf = tracker.gesture_classifier
                    if clf._use_pk_kalman:
                        # 当前是坐标级 Kalman → 切回 EMA (特征级)
                        clf._use_pk_kalman = False
                        clf._pk_smoother = None
                        clf.use_kalman = False
                        clf._init_smoothers()
                        tracker.use_per_keypoint_kalman = False
                        tracker.use_kalman = False
                        print("平滑器: EMA (特征级)")
                    elif clf.use_kalman:
                        # 当前是特征级 Kalman → 切到坐标级 Kalman
                        clf._use_pk_kalman = True
                        clf._pk_smoother = PerKeypointKalmanSmoother()
                        tracker.use_per_keypoint_kalman = True
                        print("平滑器: PerKeypointKalman (坐标级, 21点分组)")
                    else:
                        # 当前是 EMA → 切到特征级 Kalman
                        clf.use_kalman = True
                        clf._init_smoothers()
                        tracker.use_kalman = True
                        print("平滑器: Kalman (特征级)")
                    info = clf.get_smoother_info()
                    print(f"  参数: {info}")
            elif key == ord('p'):
                # 直接切换坐标级 Kalman 开/关 (快捷方式)
                if tracker.gesture_classifier:
                    new_mode = tracker.gesture_classifier.toggle_per_keypoint_kalman()
                    tracker.use_per_keypoint_kalman = tracker.gesture_classifier._use_pk_kalman
                    label = "坐标级 Kalman (21点分组)" if tracker.use_per_keypoint_kalman else "特征级平滑"
                    print(f"平滑模式已切换为: {label}")
                    info = tracker.gesture_classifier.get_smoother_info()
                    print(f"  参数: {info}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")
    finally:
        # 释放资源
        cap.release()
        tracker.release()
        sender.close()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    main()
