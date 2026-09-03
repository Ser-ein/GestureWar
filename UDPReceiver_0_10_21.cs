/*
GestureWar - Unity 端 UDP 接收器 (C# 脚本)
适配 MediaPipe 0.10.21 版本发送的数据格式
职责：接收 Python 端发送的 MediaPipe 手部关键点数据、手势信息，并传递给 HandModelDriver。
修改为：握拳触发换武器
*/

using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using System;
using UnityEngine.Events;

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
    
    [Header("手部模型驱动")]
    [Tooltip("引用 HandModelDriver 脚本，用于驱动 3D 手部模型")]
    public HandModelDriver handModelDriver;

    [Header("换武器设置")]
    [Tooltip("换武器冷却时间（秒），防止连续触发")]
    public float switchCooldown = 0.5f;
    
    [Header("事件绑定")]
    public UnityEvent onSwitchWeapon; // 在 Inspector 中绑定换武器方法

    // UDP 相关
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isReceiving = false;
    
    // 手部数据
    private HandData currentHandData;
    
    // 性能统计
    private int framesReceived = 0;
    private float startTime;
    private float lastDataTime;
    
    // 换武器控制
    private float lastSwitchTime = -999f;
    private string lastGesture = "open"; // 记录上一帧手势，用于上升沿检测
    
    // 手部数据结构 (与 Python 端保持一致)
    [System.Serializable]
    public class HandData
    {
        public int num_hands;
        public List<HandInfo> hands;
        public string gesture; // "fist" 或 "open"
    }
    
    [System.Serializable]
    public class HandInfo
    {
        public int id;
        public List<Landmark> landmarks;
        public BoundingBox bounding_box;
    }
    
    [System.Serializable]
    public class Landmark
    {
        public float x;
        public float y;
        public float z;
        public int pixel_x;
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
        currentHandData = new HandData
        {
            num_hands = 0,
            hands = new List<HandInfo>(),
            gesture = "open"
        };
        
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
                
                HandData newHandData = JsonUtility.FromJson<HandData>(jsonString);
                
                lock (currentHandData)
                {
                    currentHandData = newHandData;
                    framesReceived++;
                    lastDataTime = Time.time;
                }
                
                if (printData)
                    Debug.Log($"收到手部数据: {newHandData.num_hands} 只手, 手势: {newHandData.gesture}");
            }
            catch (Exception e)
            {
                // 忽略解析错误
            }
        }
    }
    
    void Update()
    {
        // 将手部数据传递给 HandModelDriver（如果有）
        if (handModelDriver != null)
        {
            handModelDriver.UpdateHandPose(currentHandData);
        }
        
        // ---- 握拳换武器检测 ----
        if (currentHandData != null && !string.IsNullOrEmpty(currentHandData.gesture))
        {
            string currentGesture = currentHandData.gesture;
            // 上升沿检测：只有当手势从 "open" 变为 "fist" 时才触发，且满足冷却时间
            if (lastGesture == "open" && currentGesture == "fist" && Time.time - lastSwitchTime >= switchCooldown)
            {
                SwitchWeapon();
                lastSwitchTime = Time.time;
            }
            lastGesture = currentGesture;
        }
        
        // 调试信息
        if (showDebug && Time.time - startTime > 1f)
        {
            float elapsedTime = Time.time - startTime;
            float fps = framesReceived / elapsedTime;
            
            float timeSinceLastData = Time.time - lastDataTime;
            string timeoutStatus = timeSinceLastData > 1f ? " (超时)" : "";
            
            Debug.Log($"接收帧率: {fps:F1} FPS, 手部数量: {currentHandData.num_hands}, 手势: {currentHandData.gesture}{timeoutStatus}");
            
            framesReceived = 0;
            startTime = Time.time;
        }
    }
    
    // ---- 换武器方法 ----
    void SwitchWeapon()
    {
        // 触发 UnityEvent，在 Inspector 中绑定实际的换武器函数
        onSwitchWeapon?.Invoke();
        Debug.Log("🔄 换武器！");
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
    
    // 公开方法：获取特定手部的食指尖端坐标 (示例)
    public Vector3 GetIndexFingerPosition(int handIndex = 0)
    {
        if (currentHandData.hands.Count > handIndex)
        {
            HandInfo handInfo = currentHandData.hands[handIndex];
            if (handInfo.landmarks.Count > 8)
            {
                Landmark indexFinger = handInfo.landmarks[8];
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