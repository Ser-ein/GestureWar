"""
GestureWar - 手势分类模块
基于手部 21 个关键点的几何关系，进行规则式手势识别。

支持的手势:
  - 静态: 射击 (shoot) / 瞄准 (aim) / 手雷 (grenade)
  - 时序: 换弹 (reload) / 近战 (melee) / 切换武器 (switch_weapon)

用法:
    classifier = GestureClassifier()
    gesture, confidence = classifier.classify(hand_landmarks)
"""

import time
import math
from collections import deque

# ---------------------------------------------------------------------------
# 关键点索引 (MediaPipe 标准 21 点)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# 手指定义: (名称, TIP, PIP, MCP)
FINGERS = [
    ("thumb",  THUMB_TIP,  THUMB_IP,  THUMB_MCP),
    ("index",  INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
    ("middle", MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    ("ring",   RING_TIP,   RING_PIP,   RING_MCP),
    ("pinky",  PINKY_TIP,  PINKY_DIP,  PINKY_MCP),
]

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _distance(p1, p2):
    """两点之间的欧氏距离 (2D/3D)"""
    dim = min(len(p1), len(p2))
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(dim)))


def _angle_between(v1, v2):
    """两向量夹角 (度)"""
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def _vector(p_from, p_to):
    """从 p_from 指向 p_to 的向量"""
    return (p_to[0] - p_from[0], p_to[1] - p_from[1], p_to[2] - p_from[2])


# ---------------------------------------------------------------------------
# EMA 平滑器 — 滤除摄像头抖动，让时序手势检测更稳定
# ---------------------------------------------------------------------------

class EMASmoother:
    """
    指数移动平均 (Exponential Moving Average) 平滑器。

    原理: smoothed = alpha * raw + (1 - alpha) * previous_smoothed
    - alpha 越大 → 响应越快，但保留更多噪声
    - alpha 越小 → 越平滑，但响应延迟增大
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


# ---------------------------------------------------------------------------
# GestureClassifier
# ---------------------------------------------------------------------------

class GestureClassifier:
    """
    基于规则的实时手势分类器。

    可调参数 (通过 __init__ 传入):
        finger_extend_ratio: 手指延伸判定比例 (默认 1.25)
        pinch_threshold:     捏合距离阈值 (默认 0.06)
        openness_high:       张开阈值 (默认 0.8, 即 4/5 手指延伸)
        openness_low:        闭合阈值 (默认 0.2, 即 1/5 手指延伸)
        reload_timeout:      换弹序列时限 (秒, 默认 1.5)
        melee_velocity:      近战速度阈值 (归一化坐标/秒, 默认 1.5)
        switch_angle:        翻转角度阈值 (度, 默认 60)
    """

    # 手势名称常量
    NONE = "none"
    SHOOT = "shoot"
    AIM = "aim"
    GRENADE = "grenade"
    RELOAD = "reload"
    MELEE = "melee"
    SWITCH_WEAPON = "switch_weapon"

    # 换弹状态机
    _RELOAD_IDLE = "idle"
    _RELOAD_CLOSED = "closed"
    _RELOAD_OPEN = "open"

    def __init__(
        self,
        finger_extend_ratio=1.25,     # (保留, 用于向后兼容)
        pinch_threshold=0.06,
        openness_high=0.8,
        openness_low=0.2,
        curl_angle_threshold=40.0,    # PIP 关节角度阈值 (度): <40°=伸直, ≥40°=弯曲
        reload_timeout=2.0,
        melee_velocity=0.9,            # 近战速度阈值 — 需要明显的出拳动作
        switch_angle=40.0,             # 翻转角度阈值 (度)
        melee_cooldown=0.8,
        reload_cooldown=1.0,
        switch_cooldown=0.6,
        smooth_alpha_openness=0.35,
        smooth_alpha_wrist=0.5,
        smooth_alpha_angle=0.3,
        reload_fault_tolerance=3,
    ):
        # ---- 静态手势参数 ----
        self.finger_extend_ratio = finger_extend_ratio
        self.pinch_threshold = pinch_threshold
        self.curl_angle_threshold = curl_angle_threshold

        # ---- 时序手势参数 ----
        self.openness_high = openness_high
        self.openness_low = openness_low
        self.reload_timeout = reload_timeout
        self.melee_velocity = melee_velocity
        self.switch_angle = switch_angle
        self.melee_cooldown = melee_cooldown
        self.reload_cooldown = reload_cooldown
        self.switch_cooldown = switch_cooldown

        # ---- EMA 平滑器 ----
        self._smooth_openness = EMASmoother(alpha=smooth_alpha_openness)
        self._smooth_wrist_x = EMASmoother(alpha=smooth_alpha_wrist)
        self._smooth_wrist_y = EMASmoother(alpha=smooth_alpha_wrist)
        self._smooth_wrist_z = EMASmoother(alpha=smooth_alpha_wrist)
        self._smooth_angle = EMASmoother(alpha=smooth_alpha_angle)

        # 保存平滑后的最新值 (供 get_feedback 使用)
        self._smoothed_openness = 0.0
        self._smoothed_wrist = (0.0, 0.0, 0.0)
        self._smoothed_angle = 0.0

        # ---- 历史记录 (存平滑后的值) ----
        self._openness_history = deque(maxlen=60)
        self._wrist_history = deque(maxlen=30)
        self._palm_angle_history = deque(maxlen=45)

        # ---- 时序状态机 ----
        self._reload_state = self._RELOAD_IDLE
        self._reload_state_time = 0.0
        self._reload_fault_counter = 0
        self._reload_fault_tolerance = reload_fault_tolerance
        self._last_melee_time = 0.0
        self._last_reload_time = 0.0
        self._last_switch_time = 0.0
        self._peak_velocity = 0.0            # 近战速度峰值 (带衰减)
        self._peak_angle_change = 0.0        # 翻转角度峰值 (带衰减)

        # ---- 调试 ----
        self.debug_info = {}

    # -------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------

    def classify(self, hand_landmarks, timestamp=None):
        """
        对单手的关键点进行分类。

        时序手势优先 — 换弹/近战/切换武器是主动动作，
        即使同时匹配了静态手势(如出拳时手握拳=射击),
        也应该优先输出时序手势。

        参数:
            hand_landmarks: 21 个 NormalizedLandmark 的列表
            timestamp:      当前时间戳 (秒)，用于时序手势。默认取 time.time()

        返回:
            (gesture_name: str, confidence: float)
        """
        if timestamp is None:
            timestamp = time.time()

        # 提取 (x, y, z) 元组
        pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

        # 更新历史 (必须先更新，时序检测依赖历史数据)
        self._update_history(pts, timestamp)

        # ---- 时序手势优先 ----
        # 主动动作优先级高于静态姿势
        temporal = []

        r = self._check_reload(timestamp)
        if r:
            temporal.append(r)

        r = self._check_melee(timestamp)
        if r:
            temporal.append(r)

        r = self._check_switch_weapon(timestamp)
        if r:
            temporal.append(r)

        if temporal:
            temporal.sort(key=lambda x: x[1], reverse=True)
            return temporal[0]

        # ---- 静态手势 (仅在无时序动作时) ----
        candidates = []

        r = self._check_shoot(pts)
        if r:
            candidates.append(r)

        r = self._check_aim(pts)
        if r:
            candidates.append(r)

        r = self._check_grenade(pts)
        if r:
            candidates.append(r)

        if not candidates:
            return (self.NONE, 0.0)

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    # -------------------------------------------------------------------
    # 手指状态判定
    # -------------------------------------------------------------------

    def _finger_curl_angle(self, pts, mcp_idx, pip_idx, tip_idx):
        """
        计算 PIP 关节的弯曲角度 (度)。
        0° = 手指完全伸直 (MCP→PIP 和 PIP→TIP 方向一致)
        90°+ = 手指弯曲 (两段向量夹角大)

        这个指标只依赖手指自身结构，不受手掌/手指长度比例影响！
        """
        mcp = pts[mcp_idx]
        pip = pts[pip_idx]
        tip = pts[tip_idx]

        v1 = (pip[0] - mcp[0], pip[1] - mcp[1], pip[2] - mcp[2])
        v2 = (tip[0] - pip[0], tip[1] - pip[1], tip[2] - pip[2])

        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        if mag1 < 0.001 or mag2 < 0.001:
            return 0.0

        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    def _is_finger_extended(self, pts, tip_idx, pip_idx, mcp_idx):
        """
        判断一根手指是否伸直。
        使用 PIP 关节角度 — 不受手型比例影响。
        角度 < 40° → 伸直，≥ 40° → 弯曲。
        """
        angle = self._finger_curl_angle(pts, mcp_idx, pip_idx, tip_idx)
        return angle < self.curl_angle_threshold

    def _get_extended_fingers(self, pts):
        """返回哪些手指是伸直的。返回 {finger_name: bool}"""
        result = {}
        for name, tip, pip, mcp in FINGERS:
            result[name] = self._is_finger_extended(pts, tip, pip, mcp)
        return result

    def _compute_openness(self, pts):
        """
        计算手部张开度 (0.0 ~ 1.0，连续值)。

        四指 (食指~小指):  用 PIP 关节角度映射
          角度 ≤ 20° → 1.0 (完全伸直)
          角度 ≥ 100° → 0.0 (完全弯曲)
          中间线性插值

        拇指: 用指尖到食指 MCP 的距离映射 (拇指解剖结构特殊)
          距离大 → 拇指外展 (张开)
          距离小 → 拇指内收 (握拳)

        关节角度与手型比例无关，手掌长/手指短都不影响！
        """
        total = 0.0

        # 四指: 用 PIP 角度
        four_fingers = [
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ]
        for tip_idx, pip_idx, mcp_idx in four_fingers:
            angle = self._finger_curl_angle(pts, mcp_idx, pip_idx, tip_idx)
            if angle <= 20.0:
                score = 1.0
            elif angle >= 100.0:
                score = 0.0
            else:
                score = 1.0 - (angle - 20.0) / 80.0
            total += score

        # 拇指: 用拇指尖到食指 MCP 的距离
        thumb_tip = pts[THUMB_TIP]
        index_mcp = pts[INDEX_MCP]
        thumb_dist = _distance(thumb_tip, index_mcp)
        # 典型值: 张开 ~0.25, 握拳 ~0.08
        thumb_score = max(0.0, min(1.0, (thumb_dist - 0.06) / 0.18))
        total += thumb_score

        return total / 5.0

    # -------------------------------------------------------------------
    # 静态手势检测
    # -------------------------------------------------------------------

    def _check_shoot(self, pts):
        """
        射击手势: 食指伸直 + 其余三指弯曲。
        拇指不做要求 (自然伸展或弯曲均可)。
        """
        ext = self._get_extended_fingers(pts)

        shoot_pattern = (
            ext["index"] and
            not ext["middle"] and
            not ext["ring"] and
            not ext["pinky"]
        )

        if not shoot_pattern:
            return None

        # 置信度: 看食指延伸的程度，以及其余手指弯曲的程度
        index_ratio = _distance(pts[INDEX_TIP], pts[WRIST]) / max(0.01, _distance(pts[INDEX_MCP], pts[WRIST]))

        # 中指弯曲程度 (tip 离 wrist 越近越弯曲)
        middle_ratio = _distance(pts[MIDDLE_TIP], pts[WRIST]) / max(0.01, _distance(pts[MIDDLE_MCP], pts[WRIST]))
        ring_ratio = _distance(pts[RING_TIP], pts[WRIST]) / max(0.01, _distance(pts[RING_MCP], pts[WRIST]))
        pinky_ratio = _distance(pts[PINKY_TIP], pts[WRIST]) / max(0.01, _distance(pts[PINKY_MCP], pts[WRIST]))

        # 食指越直越高，其余越弯越高
        confidence = (
            0.35 * min(index_ratio / self.finger_extend_ratio, 2.0) +
            0.25 * (1.0 - min(middle_ratio, 1.0)) +
            0.20 * (1.0 - min(ring_ratio, 1.0)) +
            0.20 * (1.0 - min(pinky_ratio, 1.0))
        )
        confidence = min(confidence, 1.0)

        self.debug_info["shoot"] = {
            "index_ratio": round(index_ratio, 2),
            "middle_ratio": round(middle_ratio, 2),
            "ring_ratio": round(ring_ratio, 2),
            "pinky_ratio": round(pinky_ratio, 2),
        }

        return (self.SHOOT, round(confidence, 3))

    def _check_aim(self, pts):
        """
        瞄准手势: 拇指指尖与食指尖端捏合 (距离很小)。

        带迟滞 (hysteresis) 防抖:
          - 距离 < pinch_threshold → 进入瞄准
          - 距离 > pinch_threshold × 1.3 → 退出瞄准
          - 中间区域 → 保持当前状态不变
        这样指尖坐标的正常抖动不会导致瞄准状态闪烁。
        """
        dist = _distance(pts[THUMB_TIP], pts[INDEX_TIP])
        release_threshold = self.pinch_threshold * 1.3  # ~0.078

        # 迟滞判断
        was_aiming = getattr(self, '_aim_active', False)
        if was_aiming:
            if dist > release_threshold:
                self._aim_active = False
                return None
            # 还在保持区 → 继续瞄准
        else:
            if dist > self.pinch_threshold:
                return None
            self._aim_active = True

        # 距离越小置信度越高
        confidence = 1.0 - (dist / release_threshold)
        confidence = max(0.35, min(confidence, 1.0))

        self.debug_info["aim"] = {
            "pinch_dist": round(dist, 4),
            "hysteresis": was_aiming,
        }

        return (self.AIM, round(confidence, 3))

    def _check_grenade(self, pts):
        """
        手雷/特殊技能: 五指全部张开。
        """
        ext = self._get_extended_fingers(pts)

        if not all(ext.values()):
            return None

        # 所有手指都伸直 → 计算平均延伸比例作为置信度
        ratios = []
        for name, tip, pip, mcp in FINGERS:
            r = _distance(pts[tip], pts[WRIST]) / max(0.01, _distance(pts[mcp], pts[WRIST]))
            ratios.append(r)

        avg_ratio = sum(ratios) / len(ratios)
        confidence = min(avg_ratio / (self.finger_extend_ratio * 1.3), 1.0)
        confidence = max(0.5, confidence)

        self.debug_info["grenade"] = {
            "avg_ratio": round(avg_ratio, 2),
        }

        return (self.GRENADE, round(confidence, 3))

    # -------------------------------------------------------------------
    # 时序手势检测
    # -------------------------------------------------------------------

    def _update_history(self, pts, timestamp):
        """维护时序历史数据 — 先 EMA 平滑再存，滤除摄像头抖动"""
        # 张开度 → 平滑
        raw_openness = self._compute_openness(pts)
        self._smoothed_openness = self._smooth_openness.update(raw_openness)
        self._openness_history.append((timestamp, self._smoothed_openness))

        # 手腕位置 → 逐轴平滑
        wrist = pts[WRIST]
        sx = self._smooth_wrist_x.update(wrist[0])
        sy = self._smooth_wrist_y.update(wrist[1])
        sz = self._smooth_wrist_z.update(wrist[2])
        self._smoothed_wrist = (sx, sy, sz)
        self._wrist_history.append((timestamp, sx, sy, sz))

        # 手掌角度 → 平滑
        raw_angle = self._compute_palm_angle(pts)
        self._smoothed_angle = self._smooth_angle.update(raw_angle)
        self._palm_angle_history.append((timestamp, self._smoothed_angle))

        # 清理过期数据
        self._prune_history(timestamp)

        # 更新状态机 (用平滑后的张开度)
        self._update_reload_state(timestamp, self._smoothed_openness)

    def _prune_history(self, now):
        """移除超过 max_history_time 的旧数据"""
        cutoff = now - 2.0
        while self._openness_history and self._openness_history[0][0] < cutoff:
            self._openness_history.popleft()
        while self._wrist_history and self._wrist_history[0][0] < cutoff:
            self._wrist_history.popleft()
        while self._palm_angle_history and self._palm_angle_history[0][0] < cutoff:
            self._palm_angle_history.popleft()

    def _compute_palm_angle(self, pts):
        """
        估算手掌法向量 (用于检测翻转)。
        使用 wrist(0), middle_mcp(9), ring_mcp(13) 三点构成平面。
        返回法向量在 xz 平面上的角度 (近似)。
        """
        w = pts[WRIST]
        m = pts[MIDDLE_MCP]
        r = pts[RING_MCP]

        # 两条边
        v1 = _vector(w, m)  # wrist → middle MCP
        v2 = _vector(w, r)  # wrist → ring MCP

        # 叉积得到法向量
        normal = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )

        # 只取 x 分量 (近似左右翻转的角度指标)
        # 归一化后取 x
        mag = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if mag < 0.001:
            return 0.0

        return normal[0] / mag

    # --- 换弹 (Reload) ---

    def _update_reload_state(self, timestamp, openness):
        """
        换弹状态机 (带容错): IDLE → CLOSED → OPEN → TRIGGER

        容错设计: 每帧偏离阈值时累积 fault_counter，
        只有连续 N 帧偏离才重置状态，避免单帧抖动打断序列。
        """
        now = timestamp
        tol = self._reload_fault_tolerance

        if self._reload_state == self._RELOAD_IDLE:
            if openness <= self.openness_low:
                self._reload_state = self._RELOAD_CLOSED
                self._reload_state_time = now
                self._reload_fault_counter = 0

        elif self._reload_state == self._RELOAD_CLOSED:
            if now - self._reload_state_time > self.reload_timeout:
                # 超时
                self._reload_state = self._RELOAD_IDLE
                self._reload_fault_counter = 0
            elif openness >= self.openness_high:
                self._reload_state = self._RELOAD_OPEN
                self._reload_state_time = now
                self._reload_fault_counter = 0
            elif openness > self.openness_low:
                # 暂时偏离但仍低于 high → 累积容错
                self._reload_fault_counter += 1
                if self._reload_fault_counter > tol:
                    self._reload_state = self._RELOAD_IDLE
                    self._reload_fault_counter = 0

        elif self._reload_state == self._RELOAD_OPEN:
            if now - self._reload_state_time > self.reload_timeout:
                # 超时
                self._reload_state = self._RELOAD_IDLE
                self._reload_fault_counter = 0
            elif openness <= self.openness_low:
                # 完成 闭→开→闭 序列！
                self._reload_state = self._RELOAD_IDLE
                self._last_reload_time = now
                self._reload_fault_counter = 0
            elif openness < self.openness_high:
                # 暂时偏离但仍高于 low → 累积容错
                self._reload_fault_counter += 1
                if self._reload_fault_counter > tol:
                    self._reload_state = self._RELOAD_IDLE
                    self._reload_fault_counter = 0

    def _check_reload(self, timestamp):
        """检测换弹是否刚刚完成 + 冷却时间"""
        elapsed = timestamp - self._last_reload_time
        # 只在检测到的瞬间返回 (窗口 0.15 秒)
        if elapsed < 0.15 and elapsed >= 0:
            if timestamp - self._last_reload_time < self.reload_cooldown:
                # 冷却中，不重复触发
                pass
            confidence = min(0.9, 1.0 - elapsed / 0.15)
            self.debug_info["reload"] = {"triggered": True}
            return (self.RELOAD, round(confidence, 3))

        return None

    # --- 近战 (Melee) ---

    def _check_melee(self, timestamp):
        """
        检测快速挥拳: 峰值速度记忆 + 触发保持。

        问题: 拳头朝摄像头快速移动时，MediaPipe 追踪质量下降，
              等追踪恢复时速度峰值已过，导致近战一闪而过。

        方案:
          1. 峰值记忆 — 维护一个慢衰减的速度峰值，错过的高峰也能补触发
          2. 触发保持 — 触发后持续返回近战 0.35s，不会闪一下就消失
        """
        MELEE_HOLD = 0.35  # 触发后保持显示的时间 (秒)
        PEAK_DECAY = 0.75  # 峰值每帧衰减系数 (越低越快忘记)

        elapsed = timestamp - self._last_melee_time

        # ---- 触发保持: 还在保持窗口内就继续返回近战 ----
        if 0 < elapsed < MELEE_HOLD:
            hold_confidence = 1.0 - (elapsed / MELEE_HOLD) * 0.5
            return (self.MELEE, round(hold_confidence, 3))

        # ---- 冷却检查 (保持窗口结束后) ----
        if elapsed < self.melee_cooldown:
            return None

        # ---- 计算当前速度 ----
        if len(self._wrist_history) < 3:
            return None

        recent = list(self._wrist_history)[-7:]
        if len(recent) < 2:
            return None

        t0, x0, y0, z0 = recent[0]
        tn, xn, yn, zn = recent[-1]
        dt = tn - t0
        if dt < 0.01:
            return None

        dist = _distance((x0, y0, z0), (xn, yn, zn))
        current_vel = dist / dt

        # 瞬时速度 (最近两帧)
        if len(recent) >= 2:
            pt, px, py, pz = recent[-2]
            inst_dist = _distance((xn, yn, zn), (px, py, pz))
            inst_dt = tn - pt
            inst_vel = inst_dist / inst_dt if inst_dt > 0.001 else 0.0
        else:
            inst_vel = 0.0

        # ---- 峰值记忆: 取当前速度, 旧峰值衰减后比较 ----
        self._peak_velocity = max(
            max(current_vel, inst_vel),
            self._peak_velocity * PEAK_DECAY
        )

        self.debug_info["melee"] = {
            "velocity": round(max(current_vel, inst_vel), 3),
            "peak": round(self._peak_velocity, 3),
            "threshold": self.melee_velocity,
        }

        if self._peak_velocity < self.melee_velocity:
            return None

        # ---- 触发! ----
        triggered_peak = self._peak_velocity
        self._last_melee_time = timestamp
        self._peak_velocity = 0.0  # 重置峰值

        confidence = min(triggered_peak / (self.melee_velocity * 2.5), 1.0)
        return (self.MELEE, round(confidence, 3))

    # --- 切换武器 (Switch Weapon) ---

    def _check_switch_weapon(self, timestamp):
        """
        检测手掌翻转: 峰值角度变化记忆 + 触发保持。

        和近战一样的问题 — 摄像头对手掌翻转时的追踪不稳定，
        翻转过程中可能丢帧，导致角度变化峰值被错过。
        用相同的"峰值记忆 + 保持窗口"策略解决。
        """
        SWITCH_HOLD = 0.35   # 触发后保持显示 (秒)
        PEAK_DECAY = 0.85    # 峰值每帧衰减系数

        elapsed = timestamp - self._last_switch_time

        # ---- 触发保持 ----
        if 0 < elapsed < SWITCH_HOLD:
            hold_confidence = 1.0 - (elapsed / SWITCH_HOLD) * 0.5
            return (self.SWITCH_WEAPON, round(hold_confidence, 3))

        # ---- 冷却检查 ----
        if elapsed < self.switch_cooldown:
            return None

        if len(self._palm_angle_history) < 5:
            return None

        # 只看最近 ~0.5 秒的角度变化 (避免正常晃动累积)
        recent_all = list(self._palm_angle_history)
        cutoff_t = timestamp - 0.5
        recent = [(t, a) for t, a in recent_all if t >= cutoff_t]
        if len(recent) < 3:
            return None

        a0 = recent[0][1]

        max_change = 0.0
        for _t, angle in recent:
            change = abs(angle - a0)
            if change > max_change:
                max_change = change

        # ---- 峰值记忆 ----
        self._peak_angle_change = max(
            max_change,
            getattr(self, '_peak_angle_change', 0.0) * PEAK_DECAY
        )

        threshold = self.switch_angle / 180.0  # 角度 → 归一化值

        self.debug_info["switch"] = {
            "angle_change": round(max_change, 3),
            "peak_change": round(self._peak_angle_change, 3),
            "threshold": round(threshold, 3),
        }

        if self._peak_angle_change < threshold:
            return None

        # ---- 触发! ----
        triggered_peak = self._peak_angle_change
        self._last_switch_time = timestamp
        self._peak_angle_change = 0.0

        confidence = min(triggered_peak / (threshold * 2.0), 1.0)
        return (self.SWITCH_WEAPON, round(confidence, 3))

    # -------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------

    def reset(self):
        """重置所有状态 (切换玩家、重新开始等)"""
        self._openness_history.clear()
        self._wrist_history.clear()
        self._palm_angle_history.clear()
        self._smooth_openness.reset()
        self._smooth_wrist_x.reset()
        self._smooth_wrist_y.reset()
        self._smooth_wrist_z.reset()
        self._smooth_angle.reset()
        self._smoothed_openness = 0.0
        self._smoothed_wrist = (0.0, 0.0, 0.0)
        self._smoothed_angle = 0.0
        self._reload_state = self._RELOAD_IDLE
        self._reload_state_time = 0.0
        self._reload_fault_counter = 0
        self._peak_velocity = 0.0
        self._peak_angle_change = 0.0
        self._aim_active = False
        self.debug_info.clear()

    def get_openness(self):
        """获取当前平滑后的手部张开度 (供外部调试/显示)"""
        return self._smoothed_openness

    def get_feedback(self):
        """
        返回可视化反馈数据 (供主程序绘制调试面板)。

        返回 dict:
            openness:      当前平滑张开度 (0~1)
            velocity:      当前手腕速度 (归一化坐标/秒)
            reload_state:  换弹状态机当前状态 (idle/closed/open)
            palm_angle:    当前平滑手掌法向量 x 分量
        """
        vel = 0.0
        wrist_list = list(self._wrist_history)
        if len(wrist_list) >= 2:
            t0, x0, y0, z0 = wrist_list[0]
            tn, xn, yn, zn = wrist_list[-1]
            dt = tn - t0
            if dt > 0.01:
                vel = _distance((x0, y0, z0), (xn, yn, zn)) / dt

        # 瞬时速度
        inst_vel = 0.0
        if len(wrist_list) >= 2:
            pt, px, py, pz = wrist_list[-2]
            tn, xn, yn, zn = wrist_list[-1]
            dt_inst = tn - pt
            if dt_inst > 0.001:
                inst_vel = _distance((xn, yn, zn), (px, py, pz)) / dt_inst

        return {
            "openness": round(self._smoothed_openness, 3),
            "velocity": round(vel, 3),
            "instant_velocity": round(inst_vel, 3),
            "peak_velocity": round(self._peak_velocity, 3),
            "reload_state": self._reload_state,
            "palm_angle": round(self._smoothed_angle, 3),
            "fault_counter": self._reload_fault_counter,
        }


# ---------------------------------------------------------------------------
# 独立测试 (直接运行此文件)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("Gesture Classifier - 独立单元测试")
    print("=" * 50)

    # 测试: 用虚拟关键点模拟手势
    # 模拟食指伸直、其余握拳 (射击手势)
    import numpy as np

    class FakeLandmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z

    def make_hand(is_index_extended, is_others_extended):
        """生成假的手部关键点"""
        lm = [FakeLandmark(0.5, 0.8, 0)] * 21  # 默认全部在手腕附近

        # 手腕
        lm[WRIST] = FakeLandmark(0.5, 0.85, 0)

        # 食指 MCP
        lm[INDEX_MCP] = FakeLandmark(0.45, 0.7, 0)
        lm[INDEX_PIP] = FakeLandmark(0.43, 0.55, 0)
        lm[INDEX_DIP] = FakeLandmark(0.42, 0.42, 0)
        if is_index_extended:
            lm[INDEX_TIP] = FakeLandmark(0.41, 0.25, 0)  # 伸直
        else:
            lm[INDEX_TIP] = FakeLandmark(0.40, 0.75, 0)  # 弯曲

        # 中指 MCP
        lm[MIDDLE_MCP] = FakeLandmark(0.50, 0.68, 0)
        lm[MIDDLE_PIP] = FakeLandmark(0.50, 0.52, 0)
        lm[MIDDLE_DIP] = FakeLandmark(0.50, 0.40, 0)
        if is_others_extended:
            lm[MIDDLE_TIP] = FakeLandmark(0.50, 0.22, 0)
        else:
            lm[MIDDLE_TIP] = FakeLandmark(0.50, 0.73, 0)

        # 无名指 MCP
        lm[RING_MCP] = FakeLandmark(0.55, 0.69, 0)
        lm[RING_PIP] = FakeLandmark(0.56, 0.55, 0)
        lm[RING_DIP] = FakeLandmark(0.57, 0.43, 0)
        if is_others_extended:
            lm[RING_TIP] = FakeLandmark(0.58, 0.26, 0)
        else:
            lm[RING_TIP] = FakeLandmark(0.57, 0.74, 0)

        # 小指 MCP
        lm[PINKY_MCP] = FakeLandmark(0.58, 0.71, 0)
        lm[PINKY_PIP] = FakeLandmark(0.60, 0.58, 0)
        lm[PINKY_DIP] = FakeLandmark(0.61, 0.48, 0)
        if is_others_extended:
            lm[PINKY_TIP] = FakeLandmark(0.62, 0.32, 0)
        else:
            lm[PINKY_TIP] = FakeLandmark(0.63, 0.76, 0)

        # 拇指 (默认自然状态)
        lm[THUMB_CMC] = FakeLandmark(0.42, 0.80, 0)
        lm[THUMB_MCP] = FakeLandmark(0.38, 0.74, 0)
        lm[THUMB_IP] = FakeLandmark(0.35, 0.68, 0)
        lm[THUMB_TIP] = FakeLandmark(0.33, 0.63, 0)

        return lm

    clf = GestureClassifier()

    # 测试 1: 射击手势
    print("\n[测试 1] 射击手势 (食指伸直, 其余握拳)")
    shoot_hand = make_hand(is_index_extended=True, is_others_extended=False)
    g, c = clf.classify(shoot_hand)
    print(f"  结果: {g} (置信度: {c})")
    print(f"  预期: shoot")
    print(f"  [PASS] {'OK' if g == 'shoot' else 'FAIL'}")

    # 测试 2: 手雷手势
    print("\n[测试 2] 手雷手势 (五指张开)")
    grenade_hand = make_hand(is_index_extended=True, is_others_extended=True)
    g, c = clf.classify(grenade_hand)
    print(f"  结果: {g} (置信度: {c})")
    print(f"  预期: grenade")
    print(f"  [PASS] {'OK' if g == 'grenade' else 'FAIL'}")

    # 测试 3: 张开度
    print("\n[测试 3] 张开度计算")
    clf.reset()
    o_shoot = clf._compute_openness([(lm.x, lm.y, lm.z) for lm in shoot_hand])
    o_grenade = clf._compute_openness([(lm.x, lm.y, lm.z) for lm in grenade_hand])
    print(f"  射击手势张开度: {o_shoot:.2f} (预期: ~0.2)")
    print(f"  手雷手势张开度: {o_grenade:.2f} (预期: ~1.0)")

    print("\n" + "=" * 50)
    print("测试完成!")
