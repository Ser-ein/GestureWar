import cv2
import mediapipe as mp
# 显式导入 solutions 模块，防止某些环境下的 AttributeError
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import time

def main():
    # 初始化 MediaPipe 手部模型
    # 设置检测参数
    # static_image_mode=False 表示处理视频流
    # max_num_hands=2 最多检测两只手
    # min_detection_confidence=0.7 检测置信度阈值
    # min_tracking_confidence=0.5 追踪置信度阈值
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 打开摄像头 (0 通常是笔记本自带摄像头)
    cap = cv2.VideoCapture(0)
    
    p_time = 0  # 用于计算 FPS

    print("正在启动摄像头，按 'q' 键退出程序...")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法读取摄像头帧")
            break

        # 1. 图像处理
        # 镜像处理，更符合交互直觉
        image = cv2.flip(image, 1)
        
        # MediaPipe 需要 RGB 格式，而 OpenCV 默认是 BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 运行手部检测
        results = hands.process(image_rgb)

        # 3. 绘制检测结果
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点 (Landmarks) 和连接线 (Connections)
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4), # 关键点颜色
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2) # 连接线颜色
                )
                
                # 打印特定点的坐标示例 (例如：食指尖端 Index Finger Tip, ID 是 8)
                # 坐标是归一化后的 (0.0 - 1.0)，需要乘以宽高得到像素坐标
                h, w, c = image.shape
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        # 4. 计算并显示 FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(image, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # 5. 显示窗口
        cv2.imshow('GestureWar - Hand Tracking', image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
