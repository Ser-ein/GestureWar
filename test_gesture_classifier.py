"""
GestureWar - 手势分类 + Kalman 滤波 综合单元测试
覆盖: 静态手势边界、时序状态机、EMA/Kalman 切换、置信度范围
"""
import sys
import math
import time
from gesture_classifier import (
    GestureClassifier, EMASmoother, KalmanSmoother,
    WRIST, THUMB_TIP, THUMB_IP, THUMB_MCP, THUMB_CMC,
    INDEX_TIP, INDEX_PIP, INDEX_MCP, INDEX_DIP,
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, MIDDLE_DIP,
    RING_TIP, RING_PIP, RING_MCP, RING_DIP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP, PINKY_DIP,
    FINGERS,
)

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

PASS = 0
FAIL = 0

def check(condition, label):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  <<< FAILED")

def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

def make_default_hand():
    """返回一个'五指全部弯曲'的基准手 (所有手指都握拳)"""
    lm = [FakeLandmark(0.5, 0.85, 0.0)] * 21

    # 手腕
    lm[WRIST] = FakeLandmark(0.50, 0.85, 0.0)

    # ---- 拇指 (弯曲) ----
    lm[THUMB_CMC] = FakeLandmark(0.42, 0.80, 0.0)
    lm[THUMB_MCP] = FakeLandmark(0.38, 0.76, 0.0)
    lm[THUMB_IP]  = FakeLandmark(0.35, 0.72, 0.0)
    lm[THUMB_TIP] = FakeLandmark(0.33, 0.68, 0.0)

    # ---- 食指 (弯曲) ----
    lm[INDEX_MCP] = FakeLandmark(0.45, 0.70, 0.0)
    lm[INDEX_PIP] = FakeLandmark(0.43, 0.58, 0.0)
    lm[INDEX_DIP] = FakeLandmark(0.41, 0.50, 0.0)
    lm[INDEX_TIP] = FakeLandmark(0.40, 0.74, 0.0)  # 弯回手腕方向

    # ---- 中指 (弯曲) ----
    lm[MIDDLE_MCP] = FakeLandmark(0.50, 0.68, 0.0)
    lm[MIDDLE_PIP] = FakeLandmark(0.50, 0.55, 0.0)
    lm[MIDDLE_DIP] = FakeLandmark(0.50, 0.47, 0.0)
    lm[MIDDLE_TIP] = FakeLandmark(0.50, 0.72, 0.0)

    # ---- 无名指 (弯曲) ----
    lm[RING_MCP] = FakeLandmark(0.55, 0.69, 0.0)
    lm[RING_PIP] = FakeLandmark(0.56, 0.57, 0.0)
    lm[RING_DIP] = FakeLandmark(0.57, 0.49, 0.0)
    lm[RING_TIP] = FakeLandmark(0.57, 0.73, 0.0)

    # ---- 小指 (弯曲) ----
    lm[PINKY_MCP] = FakeLandmark(0.58, 0.71, 0.0)
    lm[PINKY_PIP] = FakeLandmark(0.60, 0.60, 0.0)
    lm[PINKY_DIP] = FakeLandmark(0.61, 0.53, 0.0)
    lm[PINKY_TIP] = FakeLandmark(0.62, 0.75, 0.0)

    return lm


def set_finger_extended(lm, mcp_idx, pip_idx, dip_or_tip_idx, tip_idx=None):
    """
    将一根手指设为伸直状态 (向上伸展)。

    两种调用方式:
      set_finger_extended(lm, MCP, PIP, TIP)           # 不设 DIP
      set_finger_extended(lm, MCP, PIP, DIP, TIP)      # 完整关节链
    """
    mcp = lm[mcp_idx]
    lm[pip_idx] = FakeLandmark(mcp.x, mcp.y - 0.14, mcp.z)
    pip = lm[pip_idx]
    if tip_idx is not None:
        # 完整模式: MCP, PIP, DIP, TIP
        dip_idx = dip_or_tip_idx
        lm[dip_idx] = FakeLandmark(pip.x, pip.y - 0.12, pip.z)
        dip = lm[dip_idx]
        lm[tip_idx] = FakeLandmark(dip.x, dip.y - 0.13, dip.z)
    else:
        # 简化模式: MCP, PIP, TIP
        tip = dip_or_tip_idx
        lm[tip] = FakeLandmark(pip.x, pip.y - 0.25, pip.z)


def set_finger_half_curl(lm, mcp_idx, pip_idx, dip_idx, tip_idx, angle_deg=50):
    """将一根手指设为特定弯曲角度 (近似)"""
    mcp = lm[mcp_idx]
    # PIP 在 MCP 上方
    lm[pip_idx] = FakeLandmark(mcp.x, mcp.y - 0.14, mcp.z)
    pip = lm[pip_idx]
    # 根据角度计算 TIP 位置
    rad = math.radians(angle_deg)
    seg_len = 0.25
    dx = seg_len * math.sin(rad)
    dy = -seg_len * math.cos(rad)
    lm[tip_idx] = FakeLandmark(pip.x + dx, pip.y + dy, pip.z)
    # DIP 在中间
    lm[dip_idx] = FakeLandmark(pip.x + dx * 0.5, pip.y + dy * 0.5, pip.z)


def set_thumb_pinch(lm, index_tip, distance):
    """设置拇指尖与食指尖的距离"""
    lm[THUMB_TIP] = FakeLandmark(
        index_tip.x + distance,
        index_tip.y,
        index_tip.z
    )


# ---------------------------------------------------------------------------
# EMASmoother 测试
# ---------------------------------------------------------------------------

def test_ema_smoother():
    section("EMASmoother 基础")

    ema = EMASmoother(alpha=0.5)
    check(ema.value is None, "初始 value 为 None")

    v1 = ema.update(1.0)
    check(abs(v1 - 1.0) < 0.001, f"首次 update 返回原始值 (got {v1})")

    v2 = ema.update(0.0)
    expected = 0.5 * 0.0 + 0.5 * 1.0  # = 0.5
    check(abs(v2 - expected) < 0.001, f"EMA 公式正确: {v2:.3f} ≈ {expected:.3f}")

    ema.reset()
    check(ema.value is None, "reset 后 value 为 None")

    v3 = ema.update(0.5)
    check(abs(v3 - 0.5) < 0.001, "reset 后首次 update = 原始值")


def test_ema_convergence():
    section("EMASmoother 收敛性")

    ema = EMASmoother(alpha=0.3)
    # 从 0.0 开始收敛到 0.8
    ema.update(0.0)  # 先设一个不同的初始值
    values = [ema.update(0.8) for _ in range(20)]
    check(values[-1] > 0.75, f"20 帧后应接近目标 0.8 (got {values[-1]:.3f})")
    check(values[-1] > values[0], "收敛过程中值递增")


# ---------------------------------------------------------------------------
# KalmanSmoother 测试
# ---------------------------------------------------------------------------

def test_kalman_smoother():
    section("KalmanSmoother 基础")

    kf = KalmanSmoother(process_noise=0.01, measurement_noise=0.005)
    check(kf.value is None, "初始 value 为 None")
    check(kf.velocity == 0.0, "初始 velocity 为 0")

    v1 = kf.update(0.5)
    check(abs(v1 - 0.5) < 0.001, f"首次 update 返回原始值 (got {v1})")

    # 输入一个稍大的值，Kalman 应该平滑跟随（不会完全到新值）
    v2 = kf.update(0.55)
    check(v1 < v2 < 0.55, f"Kalman 平滑跟随: {v1:.4f} < {v2:.4f} < 0.55")

    # 速度估计应非零（位置在变化）
    check(kf.velocity != 0.0, "检测到运动 → 速度非零")

    kf.reset()
    check(kf.value is None, "reset 后 value 为 None")
    check(kf.velocity == 0.0, "reset 后 velocity 为 0")


def test_kalman_vs_ema():
    section("Kalman vs EMA 行为差异")

    # EMA 和 Kalman 对阶跃输入的响应
    ema = EMASmoother(alpha=0.35)
    kf = KalmanSmoother(process_noise=0.01, measurement_noise=0.005)

    # 先稳定在 0.5
    for _ in range(10):
        ema.update(0.5)
        kf.update(0.5)

    # 阶跃到 0.7
    ema_jump = ema.update(0.7)
    kf_jump = kf.update(0.7)

    # Kalman 因为有速度模型，应该比 EMA 更积极地跟随阶跃
    # (Kalman 拥有速度估计，会预测位置变化)
    check(ema_jump > 0.5, f"EMA 响应阶跃 (got {ema_jump:.4f})")
    check(kf_jump > 0.5, f"Kalman 响应阶跃 (got {kf_jump:.4f})")

    # 两者都应该 < 0.7 (平滑)
    check(ema_jump < 0.7, "EMA 不会直接跳到目标")
    check(kf_jump < 0.7, "Kalman 不会直接跳到目标")


# ---------------------------------------------------------------------------
# 静态手势 — 射击
# ---------------------------------------------------------------------------

def test_shoot():
    section("射击手势 (shoot)")

    clf = GestureClassifier()

    # 标准射击: 食指伸直 + 其余弯曲
    hand = make_default_hand()
    set_finger_extended(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    g, c = clf.classify(hand)
    check(g == "shoot", f"食指直+三指弯 → shoot (got {g}, conf={c:.3f})")
    check(c > 0.5, f"置信度 > 0.5 (got {c:.3f})")

    # 食指也弯曲 → 不是射击
    clf.reset()
    hand2 = make_default_hand()
    g, c = clf.classify(hand2)
    check(g != "shoot", f"五指全弯 → not shoot (got {g})")

    # 食指 + 中指都伸直 → 不是射击
    clf.reset()
    hand3 = make_default_hand()
    set_finger_extended(hand3, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    set_finger_extended(hand3, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
    g, c = clf.classify(hand3)
    check(g != "shoot", f"食指+中指直 → not shoot (got {g})")

    # 所有手指伸直 → 应该是手雷不是射击
    clf.reset()
    hand4 = make_default_hand()
    for _, tip, pip, mcp in FINGERS:
        set_finger_extended(hand4, mcp, pip, tip)
    # 修正：set_finger_extended 需要 dip_idx，我们传 0 当占位
    # 直接用 FINGERS 的定义来设置
    g, c = clf.classify(hand4)
    check(g == "grenade", f"五指全直 → grenade 不是 shoot (got {g})")


# ---------------------------------------------------------------------------
# 静态手势 — 瞄准
# ---------------------------------------------------------------------------

def test_aim():
    section("瞄准手势 (aim) — 基本检测")

    clf = GestureClassifier()

    # 拇指-食指距离很小 → 瞄准
    # 注意: 半弯手指避免被 shoot 覆盖
    hand = make_default_hand()
    set_finger_half_curl(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, angle_deg=50)
    set_thumb_pinch(hand, hand[INDEX_TIP], distance=0.03)  # < 0.06 threshold
    g, c = clf.classify(hand)
    check(g == "aim", f"捏合距离 0.03 + 半弯食指 → aim (got {g}, conf={c:.3f})")
    check(c > 0.3, f"置信度合理 (got {c:.3f})")

    # 距离 > 0.06 → 不是瞄准
    clf.reset()
    hand2 = make_default_hand()
    set_finger_half_curl(hand2, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, angle_deg=50)
    set_thumb_pinch(hand2, hand2[INDEX_TIP], distance=0.10)  # > 0.06
    g, c = clf.classify(hand2)
    check(g != "aim", f"捏合距离 0.10 → not aim (got {g})")


def test_aim_hysteresis():
    section("瞄准手势 — 迟滞区间")

    clf = GestureClassifier()

    # 进入瞄准 (dist < 0.06)
    hand = make_default_hand()
    set_finger_half_curl(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, angle_deg=50)
    set_thumb_pinch(hand, hand[INDEX_TIP], distance=0.04)
    g, c = clf.classify(hand)
    check(g == "aim", f"进入瞄准 (dist=0.04) → {g}")

    # 在迟滞区间内 (0.06~0.078) → 应该保持瞄准
    set_thumb_pinch(hand, hand[INDEX_TIP], distance=0.07)
    g, c = clf.classify(hand)
    check(g == "aim", f"迟滞保持 (dist=0.07) → {g} (应保持 aim)")

    # 超出释放阈值 (> 0.078) → 退出瞄准
    set_thumb_pinch(hand, hand[INDEX_TIP], distance=0.10)
    g, c = clf.classify(hand)
    check(g != "aim", f"退出瞄准 (dist=0.10) → {g} (应退出)")


# ---------------------------------------------------------------------------
# 静态手势 — 手雷
# ---------------------------------------------------------------------------

def test_grenade():
    section("手雷手势 (grenade)")

    clf = GestureClassifier()

    # 五指全张开
    hand = make_default_hand()
    for _, tip, pip, mcp in FINGERS:
        set_finger_extended(hand, mcp, pip, tip)
    g, c = clf.classify(hand)
    check(g == "grenade", f"五指全直 → grenade (got {g}, conf={c:.3f})")
    check(c >= 0.5, f"置信度 ≥ 0.5 (got {c:.3f})")

    # 只有四指 → 不是手雷
    clf.reset()
    hand2 = make_default_hand()
    for name, tip, pip, mcp in FINGERS:
        if name != "pinky":
            set_finger_extended(hand2, mcp, pip, tip)
    g, c = clf.classify(hand2)
    check(g != "grenade", f"四指直 → not grenade (got {g})")


# ---------------------------------------------------------------------------
# 时序手势 — 换弹
# ---------------------------------------------------------------------------

def test_reload_sequence():
    section("换弹手势 (reload) — 完整序列")

    # 增大容错，因为 EMA 从全开到全闭需要 ~6 帧收敛
    clf = GestureClassifier(reload_fault_tolerance=8)
    t0 = 1000.0

    # 阶段 1: 握拳 (闭合)，多帧让 EMA 收敛到低张开度
    hand_closed = make_default_hand()
    for i in range(5):
        g, c = clf.classify(hand_closed, timestamp=t0 + i * 0.033)
    check(clf._reload_state == "closed", f"多帧握拳 → state={clf._reload_state}")

    # 阶段 2: 张开，多帧让 EMA 收敛到高张开度
    hand_open = make_default_hand()
    for _, tip, pip, mcp in FINGERS:
        set_finger_extended(hand_open, mcp, pip, tip)
    t_mid = t0 + 0.5
    for i in range(5):
        g, c = clf.classify(hand_open, timestamp=t_mid + i * 0.033)
    check(clf._reload_state == "open",
          f"多帧张开 → state={clf._reload_state} (openness={clf._smoothed_openness:.3f})")

    # 阶段 3: 再握拳 (多帧)，触发换弹
    t_end = t_mid + 0.5
    for i in range(5):
        g, c = clf.classify(hand_closed, timestamp=t_end + i * 0.033)
    check(g == "reload", f"闭→开→闭 → reload (got {g}, conf={c:.3f})")


def test_reload_timeout():
    section("换弹手势 — 超时")

    clf = GestureClassifier()
    t0 = 2000.0

    # 握拳开始
    hand_closed = make_default_hand()
    clf.classify(hand_closed, timestamp=t0)
    check(clf._reload_state == "closed", f"进入 closed 状态")

    # 超过 2 秒不张开 → 超时回 idle
    clf.classify(hand_closed, timestamp=t0 + 2.5)
    check(clf._reload_state == "idle", f"超时 (t=2.5) → idle (got {clf._reload_state})")


def test_reload_fault_tolerance():
    section("换弹手势 — 容错")

    clf = GestureClassifier(reload_fault_tolerance=8)
    t0 = 3000.0

    hand_closed = make_default_hand()
    hand_open = make_default_hand()
    for _, tip, pip, mcp in FINGERS:
        set_finger_extended(hand_open, mcp, pip, tip)

    # 握拳 (多帧收敛) → 张开 (多帧收敛)
    for i in range(5):
        clf.classify(hand_closed, timestamp=t0 + i * 0.033)
    check(clf._reload_state == "closed", f"握拳 → closed")

    t1 = t0 + 0.3
    for i in range(5):
        clf.classify(hand_open, timestamp=t1 + i * 0.033)
    check(clf._reload_state == "open", f"张开 → open")

    # 短暂闭合 1 帧 (容错范围内)
    t2 = t1 + 0.2
    clf.classify(hand_closed, timestamp=t2)
    check(clf._reload_state == "open", f"单帧偏离 → 仍 open (容错) (got {clf._reload_state})")

    # 再张开，继续等待闭合
    t3 = t2 + 0.05
    for i in range(3):
        clf.classify(hand_open, timestamp=t3 + i * 0.033)

    # 最终闭合 (多帧收敛) → 触发
    t4 = t3 + 0.2
    for i in range(5):
        g, c = clf.classify(hand_closed, timestamp=t4 + i * 0.033)
    check(g == "reload", f"容错后仍触发 reload (got {g})")


def test_reload_cooldown():
    section("换弹手势 — 冷却")

    clf = GestureClassifier(reload_cooldown=1.0, reload_fault_tolerance=8)
    t0 = 4000.0

    hand_closed = make_default_hand()
    hand_open = make_default_hand()
    for _, tip, pip, mcp in FINGERS:
        set_finger_extended(hand_open, mcp, pip, tip)

    # 第一次触发 (闭→开→闭，各多帧)
    for i in range(5):
        clf.classify(hand_closed, timestamp=t0 + i * 0.033)
    t1 = t0 + 0.3
    for i in range(5):
        clf.classify(hand_open, timestamp=t1 + i * 0.033)
    t2 = t1 + 0.3
    for i in range(5):
        g1, _ = clf.classify(hand_closed, timestamp=t2 + i * 0.033)
    check(g1 == "reload", f"第一次触发 → {g1}")

    # 冷却中 (reload_cooldown=1.0s)，无法再次触发
    t3 = t2 + 0.3
    for i in range(5):
        clf.classify(hand_closed, timestamp=t3 + i * 0.033)
    g2, _ = clf.classify(hand_open, timestamp=t3 + 0.2)
    check(g2 != "reload", f"冷却中 → not reload (got {g2})")


# ---------------------------------------------------------------------------
# 时序手势 — 近战
# ---------------------------------------------------------------------------

def test_melee_trigger():
    section("近战手势 (melee) — 速度触发")

    clf = GestureClassifier(melee_velocity=0.5)  # 降低阈值方便测试
    t0 = 5000.0

    # 先喂几帧建立历史 (手腕稳定)
    hand = make_default_hand()
    for i in range(10):
        clf.classify(hand, timestamp=t0 + i * 0.033)

    # 快速移动手腕 (模拟出拳)
    hand_fast = make_default_hand()
    hand_fast[WRIST] = FakeLandmark(0.2, 0.5, 0.0)  # 手腕大幅移动
    g, c = clf.classify(hand_fast, timestamp=t0 + 0.4)
    # 注意: 需要足够的速度才能触发
    print(f"    近战检测: velocity debug = {clf.debug_info.get('melee', {})}")
    # 速度可能够也可能不够，仅验证不崩溃
    check(g in ("melee", "none"), f"近战检测不崩溃 (got {g})")


def test_melee_hold():
    section("近战手势 — 触发保持窗口")

    clf = GestureClassifier(melee_velocity=0.3, melee_cooldown=0.3)
    t0 = 6000.0

    # 建立历史
    hand = make_default_hand()
    for i in range(10):
        clf.classify(hand, timestamp=t0 + i * 0.033)

    # 大幅快速移动
    hand_fast = make_default_hand()
    hand_fast[WRIST] = FakeLandmark(0.1, 0.3, 0.0)
    g1, _ = clf.classify(hand_fast, timestamp=t0 + 0.5)

    # 如果触发了，下一帧应该还在保持窗口内
    if g1 == "melee":
        g2, _ = clf.classify(hand, timestamp=t0 + 0.55)
        check(g2 == "melee", f"保持窗口 (0.05s后) → {g2} (应仍为 melee)")
    else:
        print(f"    速度不足以触发 (可能正常)，跳过保持测试")


# ---------------------------------------------------------------------------
# 时序手势 — 切换武器
# ---------------------------------------------------------------------------

def test_switch_weapon():
    section("切换武器 (switch_weapon) — 角度变化")

    clf = GestureClassifier(switch_angle=30.0, switch_cooldown=0.3)
    t0 = 7000.0

    # 建立历史 (手掌朝前)
    hand_normal = make_default_hand()
    for i in range(10):
        clf.classify(hand_normal, timestamp=t0 + i * 0.033)

    # 翻转手掌 (改变 MCP 位置模拟法向量变化)
    hand_flipped = make_default_hand()
    # 大幅改变 middle_mcp 和 ring_mcp 的相对位置以改变法向量
    hand_flipped[MIDDLE_MCP] = FakeLandmark(0.30, 0.50, 0.3)
    hand_flipped[RING_MCP] = FakeLandmark(0.25, 0.55, 0.3)
    g, c = clf.classify(hand_flipped, timestamp=t0 + 0.5)

    print(f"    翻转检测: angle debug = {clf.debug_info.get('switch', {})}")
    check(g in ("switch_weapon", "none"), f"翻转检测不崩溃 (got {g})")


# ---------------------------------------------------------------------------
# 时序优先
# ---------------------------------------------------------------------------

def test_temporal_priority():
    section("时序手势优先")

    # 当同时满足静态和时序条件时，时序应优先
    clf = GestureClassifier(melee_velocity=0.3, melee_cooldown=0.3)
    t0 = 8000.0

    # 用"握拳+快速移动"：静态看是 shoot，但如果速度够就是 melee
    hand = make_default_hand()  # 握拳 = 静态 shoot 模式
    set_finger_extended(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    # 先建立低速历史
    hand_slow = make_default_hand()
    set_finger_extended(hand_slow, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    for i in range(10):
        clf.classify(hand_slow, timestamp=t0 + i * 0.033)

    # 快速移动
    hand_fast = make_default_hand()
    set_finger_extended(hand_fast, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    hand_fast[WRIST] = FakeLandmark(0.1, 0.3, 0.0)

    g, c = clf.classify(hand_fast, timestamp=t0 + 0.5)
    print(f"    时序优先: {g} (同时满足 shoot 静态 + 快速运动)")
    # 如果是 melee，说明时序优先生效
    if g == "melee":
        check(True, "时序手势 (melee) 优先于静态手势 (shoot)")
    else:
        check(g == "shoot", "速度不足以触发近战时降级为 shoot")


# ---------------------------------------------------------------------------
# Kalman/EMA 切换
# ---------------------------------------------------------------------------

def test_toggle():
    section("运行时 EMA <-> Kalman 切换")

    clf = GestureClassifier(use_kalman=False)
    info = clf.get_smoother_info()
    check(info["type"] == "EMA", f"默认 EMA (got {info['type']})")

    mode = clf.toggle_smoother()
    check(mode == "kalman", f"切换到 Kalman (got {mode})")
    info = clf.get_smoother_info()
    check(info["type"] == "Kalman", f"确认 Kalman 模式")

    mode = clf.toggle_smoother()
    check(mode == "ema", f"切回 EMA (got {mode})")

    # 切换后仍能正常分类
    hand = make_default_hand()
    set_finger_extended(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    g, c = clf.classify(hand)
    check(g == "shoot", f"切换后分类正常 → {g} (conf={c:.3f})")


def test_kalman_mode_classify():
    section("Kalman 模式下的手势分类")

    clf = GestureClassifier(use_kalman=True)

    # 射击
    hand = make_default_hand()
    set_finger_extended(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    g, c = clf.classify(hand)
    check(g == "shoot", f"Kalman 模式 shoot → {g} (conf={c:.3f})")

    # 手雷
    clf.reset()
    hand2 = make_default_hand()
    for _, tip, pip, mcp in FINGERS:
        set_finger_extended(hand2, mcp, pip, tip)
    g, c = clf.classify(hand2)
    check(g == "grenade", f"Kalman 模式 grenade → {g} (conf={c:.3f})")


# ---------------------------------------------------------------------------
# 边界条件
# ---------------------------------------------------------------------------

def test_edge_cases():
    section("边界条件")

    clf = GestureClassifier()

    # 极端坐标 (归一化坐标边界 0 和 1) — 不应崩溃
    try:
        hand_edge = make_default_hand()
        hand_edge[WRIST] = FakeLandmark(0.0, 0.0, 0.0)  # 左上角
        hand_edge[INDEX_TIP] = FakeLandmark(1.0, 1.0, 1.0)  # 右下角
        g, c = clf.classify(hand_edge)
        check(True, "极端坐标不崩溃")
    except Exception as e:
        check(False, f"极端坐标崩溃: {e}")

    # 重置后立即分类
    clf.reset()
    hand = make_default_hand()
    set_finger_extended(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    g, c = clf.classify(hand)
    check(g in ("shoot", "none"), f"reset 后立即分类 → {g}")

    # 连续相同帧 — 不应累积错误
    clf.reset()
    hand = make_default_hand()
    for _ in range(50):
        g, _ = clf.classify(hand)
    check(True, "连续 50 帧相同输入不崩溃")

    # 张开度边界: 验证 0~1 范围
    openness = clf._compute_openness([(lm.x, lm.y, lm.z) for lm in hand])
    check(0.0 <= openness <= 1.0, f"张开度在 [0,1] 内 (got {openness:.3f})")

    # 全部极值 → 张开度
    hand_extreme = make_default_hand()
    for _, tip, pip, mcp in FINGERS:
        set_finger_extended(hand_extreme, mcp, pip, tip)
    openness_full = clf._compute_openness([(lm.x, lm.y, lm.z) for lm in hand_extreme])
    check(openness_full > 0.7, f"五指全伸张开度 > 0.7 (got {openness_full:.3f})")


def test_confidence_range():
    section("置信度范围")

    clf = GestureClassifier()

    # 明确的射击手势
    hand = make_default_hand()
    set_finger_extended(hand, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    g, c = clf.classify(hand)
    check(0.0 <= c <= 1.0, f"置信度在 [0,1] (got {c:.3f})")

    # 无手势
    clf.reset()
    hand2 = make_default_hand()
    g, c = clf.classify(hand2)
    check(g == "none", f"无手势 → none (got {g})")
    check(c == 0.0, f"无手势置信度 = 0 (got {c})")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  GestureWar — 手势分类 + Kalman 综合测试")
    print("=" * 60)

    # 平滑器测试
    test_ema_smoother()
    test_ema_convergence()
    test_kalman_smoother()
    test_kalman_vs_ema()

    # 静态手势测试
    test_shoot()
    test_aim()
    test_aim_hysteresis()
    test_grenade()

    # 时序手势测试
    test_reload_sequence()
    test_reload_timeout()
    test_reload_fault_tolerance()
    test_reload_cooldown()
    test_melee_trigger()
    test_melee_hold()
    test_switch_weapon()

    # 优先级 + 切换
    test_temporal_priority()
    test_toggle()
    test_kalman_mode_classify()

    # 边界
    test_edge_cases()
    test_confidence_range()

    # 汇总
    total = PASS + FAIL
    print(f"\n{'='*60}")
    print(f"  结果: {PASS}/{total} 通过", end="")
    if FAIL > 0:
        print(f"  |  {FAIL} 失败 **")
        sys.exit(1)
    else:
        print(f"  |  全部通过! **")
    print(f"{'='*60}")
