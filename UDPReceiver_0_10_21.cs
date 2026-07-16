"""
GestureWar - Unity 端 UDP 接收器 (C# 脚本)
适配 MediaPipe 0.10.21 版本发送的数据格式
"""

using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using System;

public class UDPReceiver_0_10_21 : MonoBehaviour
{
    [Header("UDP 设置")]
    [Tooltip("监听端口号 (与 Python 发送端口一致)")]
    public int port = 5005;
    
    [Header("调试信息")]
    [Tooltip("是否显示调试信息")]
    public bool showDebug = true;
    
    [Tooltip("是否在控制台打印接收的数据")]
    public bool printData = false;
    
    [Tooltip("是否显示手部追踪可视化")]
    public bool showVisualization = true;
    
    [Header("手部可视化设置")]
    [Tooltip("手部关键点预制体")]
    public GameObject landmarkPrefab;
    
    [Tooltip("手部连接线材质")]
    public Material connectionMaterial;
    
    [Tooltip("手部颜色 (按手部ID分配)")]
    public Color[] handColors = {
        new Color(1f, 0f, 0f, 1f),    // 红色 - 手部0
        new Color(0f, 1f, 0f, 1f),    // 绿色 - 手部1
        new Color(0f, 0f, 1f, 1f),    // 蓝色 - 手部2
        new Color(1f, 1f, 0f, 1f)     // 黄色 - 手部3
    };
    
    // UDP 相关
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isReceiving = false;
    
    // 手部数据
    private HandData currentHandData;
    private List<GameObject> handVisualizations = new List<GameObject>();
    
    // 性能统计
    private int framesReceived = 0;
    private float startTime;
    private float lastDataTime;
    
    // 手部数据结构 (与 Python 端保持一致)
    [System.Serializable]
    public class HandData
    {
        public float timestamp;       // Unix 秒 — Python 端发送时间
        public int frame_id;          // 递增帧序号 — 用于丢包检测
        public int num_hands;
        public List<HandInfo> hands;
    }

    [System.Serializable]
    public class HandInfo
    {
        public int id;
        public List<Landmark> landmarks;
        public BoundingBox bounding_box;
        public string gesture;           // 手势名称 (shoot/aim/grenade/reload/melee/switch_weapon/none)
        public float gesture_confidence; // 手势置信度 (0~1)
    }
    
    [System.Serializable]
    public class Landmark
    {
        public float x;        // 归一化坐标 (0-1)
        public float y;
        public float z;
        public int pixel_x;    // 像素坐标
        public int pixel_y;
    }
    
    [System.Serializable]
    public class BoundingBox
    {
        public int x_min;
        public int x_max;
        public int y_min;
        public int y_max;
        public int width;
        public int height;
    }
    
    void Start()
    {
        // 初始化手部数据
        currentHandData = new HandData
        {
            num_hands = 0,
            hands = new List<HandInfo>()
        };
        
        // 启动 UDP 接收线程
        StartUDPReceiver();
        
        startTime = Time.time;
        lastDataTime = Time.time;
        
        if (showDebug)
            Debug.Log($"UDP 接收器已启动，监听端口: {port}");
    }
    
    void StartUDPReceiver()
    {
        try
        {
            udpClient = new UdpClient(port);
            isReceiving = true;
            
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError($"启动 UDP 接收器失败: {e.Message}");
        }
    }
    
    void ReceiveData()
    {
        while (isReceiving)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string jsonString = Encoding.UTF8.GetString(data);
                
                // 解析 JSON 数据
                HandData newHandData = JsonUtility.FromJson<HandData>(jsonString);
                
                // 更新当前手部数据
                lock (currentHandData)
                {
                    currentHandData = newHandData;
                    framesReceived++;
                    lastDataTime = Time.time;
                }
                
                if (printData)
                    Debug.Log($"收到手部数据: {newHandData.num_hands} 只手");
            }
            catch (Exception e)
            {
                if (isReceiving)
                    Debug.LogWarning($"接收数据时出错: {e.Message}");
            }
        }
    }
    
    void Update()
    {
        // 更新手部可视化
        if (showVisualization)
            UpdateHandVisualization();
        
        // 显示调试信息
        if (showDebug && Time.time - startTime > 1f)
        {
            float elapsedTime = Time.time - startTime;
            float fps = framesReceived / elapsedTime;
            
            // 检查数据是否超时
            float timeSinceLastData = Time.time - lastDataTime;
            string timeoutStatus = timeSinceLastData > 1f ? " (超时)" : "";
            
            string gestureInfo = "";
            if (currentHandData.hands.Count > 0)
            {
                var h = currentHandData.hands[0];
                gestureInfo = $", 手势: {h.gesture}({h.gesture_confidence:P0})";
            }
            int lost = GetPacketLossCount();
            string lossInfo = lost > 0 ? $", 丢包: {lost}" : "";
            Debug.Log($"接收帧率: {fps:F1} FPS, 手部数量: {currentHandData.num_hands}"
                      + $"{gestureInfo}{lossInfo}{timeoutStatus}");
            
            // 重置统计
            framesReceived = 0;
            startTime = Time.time;
        }
    }
    
    void UpdateHandVisualization()
    {
        // 清理旧的视觉对象
        foreach (GameObject obj in handVisualizations)
        {
            Destroy(obj);
        }
        handVisualizations.Clear();
        
        // 为每个检测到的手部创建可视化
        for (int handIdx = 0; handIdx < currentHandData.hands.Count; handIdx++)
        {
            HandInfo handInfo = currentHandData.hands[handIdx];
            
            // 创建手部容器对象
            GameObject handContainer = new GameObject($"Hand_{handInfo.id}");
            handContainer.transform.SetParent(transform);
            handVisualizations.Add(handContainer);
            
            // 获取手部颜色
            Color handColor = handColors[handIdx % handColors.Length];
            
            // 创建关键点
            List<GameObject> landmarkObjects = new List<GameObject>();
            foreach (Landmark landmark in handInfo.landmarks)
            {
                GameObject landmarkObj;
                if (landmarkPrefab != null)
                {
                    landmarkObj = Instantiate(landmarkPrefab, handContainer.transform);
                }
                else
                {
                    landmarkObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    landmarkObj.transform.SetParent(handContainer.transform);
                    landmarkObj.GetComponent<Renderer>().material.color = handColor;
                }
                
                // 设置位置 (归一化坐标转换为世界坐标)
                // 假设屏幕映射到 Unity 的 10x10 区域，原点在中心
                float worldX = (landmark.x - 0.5f) * 10f;
                float worldY = (0.5f - landmark.y) * 10f; // Y轴反转
                float worldZ = landmark.z * 2f; // 深度信息
                
                landmarkObj.transform.position = new Vector3(worldX, worldY, worldZ);
                landmarkObj.transform.localScale = Vector3.one * 0.1f;
                
                landmarkObjects.Add(landmarkObj);
            }
            
            // 创建连接线
            if (connectionMaterial != null && landmarkObjects.Count >= 21)
            {
                // 手部关键点连接关系 (基于 MediaPipe 的 HAND_CONNECTIONS)
                int[,] connections = {
                    {0, 1}, {1, 2}, {2, 3}, {3, 4},     // 拇指
                    {0, 5}, {5, 6}, {6, 7}, {7, 8},     // 食指
                    {0, 9}, {9, 10}, {10, 11}, {11, 12}, // 中指
                    {0, 13}, {13, 14}, {14, 15}, {15, 16}, // 无名指
                    {0, 17}, {17, 18}, {18, 19}, {19, 20}, // 小指
                    {5, 9}, {9, 13}, {13, 17}           // 手掌连接
                };
                
                for (int i = 0; i < connections.GetLength(0); i++)
                {
                    int startIdx = connections[i, 0];
                    int endIdx = connections[i, 1];
                    
                    if (startIdx < landmarkObjects.Count && endIdx < landmarkObjects.Count)
                    {
                        CreateConnectionLine(
                            landmarkObjects[startIdx].transform.position,
                            landmarkObjects[endIdx].transform.position,
                            handColor,
                            handContainer.transform
                        );
                    }
                }
            }
        }
    }
    
    void CreateConnectionLine(Vector3 start, Vector3 end, Color color, Transform parent)
    {
        GameObject lineObj = new GameObject("ConnectionLine");
        lineObj.transform.SetParent(parent);
        
        LineRenderer lineRenderer = lineObj.AddComponent<LineRenderer>();
        lineRenderer.material = connectionMaterial;
        lineRenderer.startColor = color;
        lineRenderer.endColor = color;
        lineRenderer.startWidth = 0.02f;
        lineRenderer.endWidth = 0.02f;
        lineRenderer.SetPosition(0, start);
        lineRenderer.SetPosition(1, end);
    }
    
    void OnApplicationQuit()
    {
        StopUDPReceiver();
    }
    
    void StopUDPReceiver()
    {
        isReceiving = false;
        
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(100);
        }
        
        if (udpClient != null)
        {
            udpClient.Close();
        }
        
        if (showDebug)
            Debug.Log("UDP 接收器已停止");
    }
    
    // 公开方法：获取当前手部数据
    public HandData GetCurrentHandData()
    {
        lock (currentHandData)
        {
            return currentHandData;
        }
    }

    // 公开方法：获取当前识别的手势 (取第一只手)
    public string GetCurrentGesture()
    {
        lock (currentHandData)
        {
            if (currentHandData.hands.Count > 0)
                return currentHandData.hands[0].gesture ?? "none";
            return "none";
        }
    }

    // 公开方法：获取当前手势及其置信度
    public (string gesture, float confidence) GetCurrentGestureWithConfidence()
    {
        lock (currentHandData)
        {
            if (currentHandData.hands.Count > 0)
            {
                var hand = currentHandData.hands[0];
                return (hand.gesture ?? "none", hand.gesture_confidence);
            }
            return ("none", 0f);
        }
    }

    // 公开方法：获取数据延迟 (秒) — 需 Unity 端也有时间基准
    public float GetDataLatency()
    {
        lock (currentHandData)
        {
            if (currentHandData.timestamp > 0)
                return Time.time - currentHandData.timestamp;
            return -1f;
        }
    }

    // 公开方法：检测丢包 (需要记录上一次的 frame_id)
    private int lastFrameId = -1;
    public int GetPacketLossCount()
    {
        lock (currentHandData)
        {
            if (lastFrameId < 0)
            {
                lastFrameId = currentHandData.frame_id;
                return 0;
            }
            int gap = currentHandData.frame_id - lastFrameId - 1;
            lastFrameId = currentHandData.frame_id;
            return gap > 0 ? gap : 0;
        }
    }
    
    // 公开方法：获取特定手部的食指尖端坐标
    public Vector3 GetIndexFingerPosition(int handIndex = 0)
    {
        if (currentHandData.hands.Count > handIndex)
        {
            HandInfo handInfo = currentHandData.hands[handIndex];
            if (handInfo.landmarks.Count > 8)
            {
                Landmark indexFinger = handInfo.landmarks[8];
                // 转换为 Unity 世界坐标
                return new Vector3(
                    (indexFinger.x - 0.5f) * 10f,
                    (0.5f - indexFinger.y) * 10f,
                    indexFinger.z * 2f
                );
            }
        }
        return Vector3.zero;
    }
    
    // 公开方法：获取所有手部的食指尖端坐标
    public List<Vector3> GetAllIndexFingerPositions()
    {
        List<Vector3> positions = new List<Vector3>();
        
        foreach (HandInfo handInfo in currentHandData.hands)
        {
            if (handInfo.landmarks.Count > 8)
            {
                Landmark indexFinger = handInfo.landmarks[8];
                positions.Add(new Vector3(
                    (indexFinger.x - 0.5f) * 10f,
                    (0.5f - indexFinger.y) * 10f,
                    indexFinger.z * 2f
                ));
            }
        }
        
        return positions;
    }
    
    // 公开方法：检查数据是否超时
    public bool IsDataTimeout(float timeoutSeconds = 1f)
    {
        return (Time.time - lastDataTime) > timeoutSeconds;
    }
}
