"""
GestureWar - 程序运行管理模块
整合手部追踪和 UDP 数据发送（含握拳检测）
"""

import cv2
import time
import sys
import json
from src.hand_tracking import HandTracker
from src.UDP_sender import UDPSender


def detect_fist(hand_landmarks, threshold=0.05):
    """
    检测手部是否为握拳
    参数:
        hand_landmarks: 手部关键点列表，每个元素含 x, y (归一化)
        threshold: 指尖到指根距离的阈值
    返回:
        "fist" 或 "open"
    """
    # 指尖和指根索引 (MediaPipe 0.10.21)
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]

    for tip_idx, mcp_idx in zip(finger_tips, finger_mcps):
        tip = hand_landmarks[tip_idx]
        mcp = hand_landmarks[mcp_idx]
        dx = tip["x"] - mcp["x"]
        dy = tip["y"] - mcp["y"]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist >= threshold:
            return "open"
    return "fist"


def main():
    """主函数 - 手部追踪 + UDP 发送 + 手势识别（握拳）"""
    print("=" * 60)
    print("GestureWar - 手部追踪 + UDP 发送 (含握拳检测)")
    print("=" * 60)
    print("功能说明:")
    print("1. 实时摄像头手部关键点检测")
    print("2. 识别握拳手势并发送到 Unity")
    print("3. 通过 UDP 发送手部数据到 Unity")
    print("4. 显示实时 FPS 和手势状态")
    print("")
    print("控制说明:")
    print("  [q] 退出程序")
    print("  [s] 保存当前帧为图片")
    print("  [i] 显示详细信息")
    print("  [u] 切换 UDP 发送 (开/关)")
    print("=" * 60)

    # 初始化手部追踪器
    hand_tracker = HandTracker(
        max_hands=1,
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

    print("摄像头已打开，开始追踪...")
    print("UDP 数据发送到: 127.0.0.1:5005")
    print("")

    # 状态变量
    udp_enabled = True
    frame_save_counter = 0
    last_debug_time = time.time()

    print("按 'q' 退出...")

    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("警告: 无法读取摄像头帧")
                break

            # 统一翻转一次，方便交互
            frame = cv2.flip(frame, 1)

            # 手部追踪处理
            processed_frame, hand_data = hand_tracker.process_frame(frame, already_flipped=True)

            # ---- 握拳检测 ----
            gesture = "open"
            if hand_data["num_hands"] > 0:
                hand_landmarks = hand_data["hands"][0]["landmarks"]
                gesture = detect_fist(hand_landmarks, threshold=0.05)

            # ---- 发送追踪数据（包含手势） ----
            if udp_enabled and hand_data["num_hands"] > 0:
                data_packet = {
                    "num_hands": hand_data["num_hands"],
                    "hands": hand_data["hands"],
                    "gesture": gesture
                }
                json_str = json.dumps(data_packet)
                sender.sock.sendto(json_str.encode('utf-8'), (sender.host, sender.port))

            # 显示手部信息
            if hand_data["num_hands"] > 0:
                for hand in hand_data["hands"]:
                    if hand["bounding_box"]:
                        bbox = hand["bounding_box"]
                        info_text = f"Hand {hand['id']}: {bbox['width']}x{bbox['height']}"
                        cv2.putText(processed_frame, info_text,
                                    (bbox['x_min'], bbox['y_min'] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示手势状态
            cv2.putText(processed_frame, f"Gesture: {gesture.upper()}", (10, 180),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

            # 显示 UDP 状态
            udp_status = "ON" if udp_enabled else "OFF"
            cv2.putText(processed_frame, f"UDP: {udp_status}", (10, 210),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            # 显示窗口
            cv2.imshow('GestureWar - Hand Tracking', processed_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户请求退出...")
                break
            elif key == ord('s'):
                filename = f"tracking_frame_{frame_save_counter:04d}.png"
                cv2.imwrite(filename, processed_frame)
                print(f"已保存帧到: {filename}")
                frame_save_counter += 1
            elif key == ord('i'):
                current_time = time.time()
                if current_time - last_debug_time > 0.5:
                    print(f"手部追踪帧: {hand_tracker.frame_count}, FPS: {hand_tracker.fps:.1f}")
                    print(f"检测到手部数量: {hand_data['num_hands']}")
                    print(f"手势: {gesture}")
                    print(f"UDP 发送: {'启用' if udp_enabled else '禁用'}")
                    last_debug_time = current_time
            elif key == ord('u'):
                udp_enabled = not udp_enabled
                status = "启用" if udp_enabled else "禁用"
                print(f"UDP 发送已{status}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")
    finally:
        cap.release()
        hand_tracker.release()
        sender.close()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    main