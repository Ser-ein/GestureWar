"""
GestureWar - 手部追踪模块 (MediaPipe 0.10.21 版本)
基于 MediaPipe 0.10.21 和 OpenCV 的实时手部关键点检测
"""

import cv2
import mediapipe as mp
import time
import sys
import json
import socket
import threading

class HandTracker:
    """手部追踪器类 - 适配 MediaPipe 0.10.21"""
    
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        """
        初始化手部追踪器
        
        参数:
            max_hands: 最大检测手部数量
            detection_confidence: 检测置信度阈值
            tracking_confidence: 追踪置信度阈值
        """
        print("正在初始化手部追踪器 (MediaPipe 0.10.21)...")
        
        # MediaPipe 0.10.21 版本的 API 结构
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # 绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 手部关键点名称映射
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
        
        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("手部追踪器初始化完成！")
    
    def process_frame(self, frame):
        """
        处理单帧图像
        
        参数:
            frame: OpenCV 图像帧 (BGR格式)
            
        返回:
            processed_frame: 处理后的图像帧
            hand_data: 手部关键点数据字典
        """
        # 镜像处理，更符合交互直觉
        frame = cv2.flip(frame, 1)
        
        # MediaPipe 需要 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 运行手部检测
        results = self.hands.process(frame_rgb)
        
        # 初始化手部数据
        hand_data = {
            "num_hands": 0,
            "hands": []
        }
        
        # 绘制检测结果
        if results.multi_hand_landmarks:
            hand_data["num_hands"] = len(results.multi_hand_landmarks)
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # 收集手部关键点数据
                hand_info = {
                    "id": hand_idx,
                    "landmarks": [],
                    "bounding_box": None
                }
                
                # 获取图像尺寸
                h, w, c = frame.shape
                
                # 计算边界框
                x_coords = []
                y_coords = []
                
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    x_coords.append(x)
                    y_coords.append(y)
                    
                    # 保存关键点信息
                    hand_info["landmarks"].append({
                        "x": landmark.x,  # 归一化坐标 (0-1)
                        "y": landmark.y,
                        "z": landmark.z,
                        "pixel_x": x,     # 像素坐标
                        "pixel_y": y
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
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 高亮显示食指尖端 (ID 8)
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                
                # 添加手部信息
                hand_data["hands"].append(hand_info)
        
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
        
        return frame, hand_data
    
    def release(self):
        """释放资源"""
        print("释放手部追踪器资源...")
        self.hands.close()
        print("资源释放完成！")


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
            
            # 调试信息
            print(f"发送数据: {hand_data['num_hands']} 只手")
            
        except Exception as e:
            print(f"发送数据时出错: {e}")
    
    def close(self):
        """关闭连接"""
        self.sock.close()
        print("UDP 发送器已关闭")


def main():
    """主函数 - 完整的手部追踪 + UDP 发送程序"""
    print("=" * 60)
    print("GestureWar - 手部追踪 + UDP 发送 (MediaPipe 0.10.21)")
    print("=" * 60)
    print("功能说明:")
    print("1. 实时摄像头手部关键点检测")
    print("2. 绘制手部骨架和边界框")
    print("3. 通过 UDP 发送手部数据到 Unity")
    print("4. 显示实时 FPS 和手部数量")
    print("")
    print("控制说明:")
    print("  [q] 退出程序")
    print("  [s] 保存当前帧为图片")
    print("  [i] 显示详细信息")
    print("  [u] 切换 UDP 发送 (开/关)")
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
    
    print("按任意键开始，按 'q' 退出...")
    
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
            cv2.imshow('GestureWar - Hand Tracking', processed_frame)
            
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
                    print(f"当前帧: {tracker.frame_count}")
                    print(f"FPS: {tracker.fps:.1f}")
                    print(f"检测到手部数量: {hand_data['num_hands']}")
                    print(f"UDP 发送: {'启用' if udp_enabled else '禁用'}")
                    last_debug_time = current_time
            elif key == ord('u'):
                # 切换 UDP 发送
                udp_enabled = not udp_enabled
                status = "启用" if udp_enabled else "禁用"
                print(f"UDP 发送已{status}")
    
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
