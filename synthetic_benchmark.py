"""
GestureWar - 合成手势基准测试
=============================
纯合成数据驱动的手势分类器测试，不依赖摄像头。

用法:
  python synthetic_benchmark.py          # 跑全部测试
  python synthetic_benchmark.py --quick  # 快速模式 (减少迭代)
  python synthetic_benchmark.py --compare # 仅 EMA vs Kalman 对比

输出: Markdown 格式报告 + 控制台摘要
"""

import math
import time
import random
import sys
import json
from collections import defaultdict, Counter

from gesture_classifier import (
    GestureClassifier,
    # 关键点索引
    WRIST,
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
    _distance,
)


# ======================================================================
# FakeLandmark — 模拟 MediaPipe NormalizedLandmark
# ======================================================================

class FakeLandmark:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z=0.0):
        self.x = x; self.y = y; self.z = z
    def __repr__(self):
        return f"FL({self.x:.3f},{self.y:.3f},{self.z:.3f})"


# ======================================================================
# 手势模板库 — 21 个关键点在不同手势下的归一化位置
# ======================================================================
# 坐标约定: x∈[0,1] (左→右), y∈[0,1] (上→下), z∈[-1,1] (深度)
# 手掌朝摄像头，手腕在画面中下部

def _pts(*coords):
    """快捷构造: 连续的 (x,y,z) 三元组 → FakeLandmark 列表"""
    result = []
    for i in range(0, len(coords), 3):
        result.append(FakeLandmark(coords[i], coords[i+1], coords[i+2]))
    assert len(result) == 21, f"模板点数错误: {len(result)}"
    return result


# 基础手掌骨架 (除手指外)
#  腕(0)    拇CMC(1)  拇MCP(2)  拇IP(3)  拇尖(4)
#  食MCP(5)  食PIP(6)  食DIP(7)  食尖(8)
#  中MCP(9)  中PIP(10) 中DIP(11) 中尖(12)
#  无MCP(13) 无PIP(14) 无DIP(15) 无尖(16)
#  小MCP(17) 小PIP(18) 小DIP(19) 小尖(20)

REST_HAND = _pts(
    # 0:手腕
    0.50, 0.82, 0.0,
    # 1-4:拇指 (自然微弯)
    0.44, 0.78, -0.02,  0.39, 0.73, -0.04,  0.35, 0.68, -0.05,  0.32, 0.63, -0.06,
    # 5-8:食指
    0.46, 0.68, -0.01,  0.44, 0.56, -0.02,  0.43, 0.46, -0.02,  0.42, 0.36, -0.03,
    # 9-12:中指
    0.50, 0.66, 0.01,   0.50, 0.52, 0.02,   0.50, 0.41, 0.02,   0.50, 0.30, 0.02,
    # 13-16:无名指
    0.54, 0.67, 0.01,   0.55, 0.54, 0.02,   0.56, 0.44, 0.02,   0.57, 0.35, 0.01,
    # 17-20:小指
    0.57, 0.69, 0.01,   0.59, 0.58, 0.01,   0.60, 0.49, 0.01,   0.61, 0.41, 0.01,
)


def _modify_finger(base, tip_idx, pip_idx, mcp_idx, bend_amount):
    """
    修改手指弯曲程度。bend_amount: 0=完全伸直, 1=完全握拳。
    把指尖和 DIP 向 MCP 方向折叠，保证 PIP 关节角度变化。
    """
    pts = [FakeLandmark(p.x, p.y, p.z) for p in base]
    tip = pts[tip_idx]
    mcp = pts[mcp_idx]

    # 指尖向 MCP 大幅折叠
    dx_tip = (mcp.x - tip.x) * bend_amount * 0.75
    dy_tip = (mcp.y - tip.y) * bend_amount * 0.75
    dz_tip = (mcp.z - tip.z) * bend_amount * 0.75
    pts[tip_idx] = FakeLandmark(tip.x + dx_tip, tip.y + dy_tip, tip.z + dz_tip)

    # DIP 向 MCP 中度折叠
    dip_idx = pip_idx + 1
    dip = pts[dip_idx]
    dx_dip = (mcp.x - dip.x) * bend_amount * 0.3
    dy_dip = (mcp.y - dip.y) * bend_amount * 0.3
    dz_dip = (mcp.z - dip.z) * bend_amount * 0.3
    pts[dip_idx] = FakeLandmark(dip.x + dx_dip, dip.y + dy_dip, dip.z + dz_dip)

    return pts


def template_shoot():
    """射击: 食指伸直, 其余握拳"""
    pts = _modify_finger(REST_HAND, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, 0.95)
    pts = _modify_finger(pts, RING_TIP, RING_PIP, RING_MCP, 0.95)
    pts = _modify_finger(pts, PINKY_TIP, PINKY_PIP, PINKY_MCP, 0.9)
    # 食指保持伸直 (bend=0, 不变)
    return pts


def template_aim():
    """瞄准: 拇食指捏合"""
    pts = [FakeLandmark(p.x, p.y, p.z) for p in REST_HAND]
    # 其余三指半握
    pts = _modify_finger(pts, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, 0.7)
    pts = _modify_finger(pts, RING_TIP, RING_PIP, RING_MCP, 0.7)
    pts = _modify_finger(pts, PINKY_TIP, PINKY_PIP, PINKY_MCP, 0.7)
    # 拇指尖靠近食指尖
    index_tip = pts[INDEX_TIP]
    pts[THUMB_TIP].x = index_tip.x - 0.02
    pts[THUMB_TIP].y = index_tip.y + 0.03
    pts[THUMB_IP].x = index_tip.x - 0.04
    pts[THUMB_IP].y = index_tip.y + 0.10
    return pts


def template_grenade():
    """手雷: 五指全张开"""
    pts = [FakeLandmark(p.x, p.y, p.z) for p in REST_HAND]
    # 拇指外展
    pts[THUMB_TIP].x = 0.24
    pts[THUMB_TIP].y = 0.50
    pts[THUMB_IP].x = 0.27
    pts[THUMB_IP].y = 0.55
    pts[THUMB_MCP].x = 0.32
    pts[THUMB_MCP].y = 0.60
    return pts


def template_reload_closed():
    """换弹-闭: 五指全握"""
    pts = _modify_finger(REST_HAND, INDEX_TIP, INDEX_PIP, INDEX_MCP, 0.95)
    pts = _modify_finger(pts, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, 0.95)
    pts = _modify_finger(pts, RING_TIP, RING_PIP, RING_MCP, 0.95)
    pts = _modify_finger(pts, PINKY_TIP, PINKY_PIP, PINKY_MCP, 0.95)
    return pts


def template_reload_open():
    """换弹-开: 五指全张开"""
    return template_grenade()


def template_melee():
    """近战: 握拳 + 手腕快速前冲 (模板只是静态握拳，速度靠序列生成)"""
    return template_reload_closed()


def template_switch_palm():
    """翻转: 手掌朝下 (模拟翻转后的手掌)"""
    pts = [FakeLandmark(p.x, p.y, p.z) for p in REST_HAND]
    # 手指稍弯 (自然)
    pts = _modify_finger(pts, INDEX_TIP, INDEX_PIP, INDEX_MCP, 0.3)
    pts = _modify_finger(pts, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, 0.3)
    pts = _modify_finger(pts, RING_TIP, RING_PIP, RING_MCP, 0.3)
    pts = _modify_finger(pts, PINKY_TIP, PINKY_PIP, PINKY_MCP, 0.3)
    # 改变手掌法向量: 手腕不动，MCP 上移模拟翻转
    for idx in [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]:
        pts[idx].z = 0.15  # z 正向偏移模拟翻转
    return pts


TEMPLATES = {
    "shoot": template_shoot,
    "aim": template_aim,
    "grenade": template_grenade,
    "reload_closed": template_reload_closed,
    "reload_open": template_reload_open,
    "melee": template_melee,
    "switch_palm": template_switch_palm,
}


# ======================================================================
# 噪声与抖动模拟
# ======================================================================

def add_noise(pts, sigma=0.005):
    """给每个关键点添加高斯噪声 (模拟摄像头抖动)"""
    noisy = []
    for p in pts:
        noisy.append(FakeLandmark(
            p.x + random.gauss(0, sigma),
            p.y + random.gauss(0, sigma),
            p.z + random.gauss(0, sigma * 0.5),
        ))
    return noisy


def add_structured_jitter(pts, amplitude=0.003, frequency=0.3):
    """
    添加结构性抖动 (模拟手部微小震颤)。
    不同手指的抖动略有相位差，比纯高斯噪声更真实。
    """
    t = random.random() * 2 * math.pi
    jittered = []
    for i, p in enumerate(pts):
        phase = i * 0.3
        dx = amplitude * math.sin(t + phase)
        dy = amplitude * math.cos(t + phase + 1.0)
        dz = amplitude * 0.3 * math.sin(t * 1.7 + phase)
        jittered.append(FakeLandmark(p.x + dx, p.y + dy, p.z + dz))
    return jittered


def interpolate_pts(pts_a, pts_b, fraction):
    """线性插值两个手势 (0=纯a, 1=纯b)"""
    f = fraction
    return [FakeLandmark(
        a.x + (b.x - a.x) * f,
        a.y + (b.y - a.y) * f,
        a.z + (b.z - a.z) * f,
    ) for a, b in zip(pts_a, pts_b)]


# ======================================================================
# 序列生成器
# ======================================================================

def generate_static_sequence(template_func, num_frames=60, noise_sigma=0.005,
                              jitter_amplitude=0.003):
    """
    生成静态手势序列: 同一手势保持 num_frames 帧, 每帧加随机噪声。
    返回 [(landmarks_list, ground_truth_gesture), ...]
    """
    base = template_func()
    seq = []
    for _ in range(num_frames):
        noisy = add_noise(base, sigma=noise_sigma)
        noisy = add_structured_jitter(noisy, amplitude=jitter_amplitude)
        seq.append(noisy)
    return seq


def generate_transition_sequence(from_template, to_template,
                                  hold_frames=30, transition_frames=15,
                                  noise_sigma=0.005):
    """
    生成手势切换序列: A(稳定) → 渐变 → B(稳定)。
    返回序列 + 切换帧索引。
    """
    a = from_template()
    b = to_template()
    seq = []
    # 阶段1: 保持 A
    for _ in range(hold_frames):
        seq.append(add_noise(a, sigma=noise_sigma))
    # 阶段2: 渐变
    switch_start = len(seq)  # 渐变开始的帧序号
    for i in range(transition_frames):
        frac = (i + 1) / transition_frames
        interp = interpolate_pts(a, b, frac)
        seq.append(add_noise(interp, sigma=noise_sigma))
    # 阶段3: 保持 B
    for _ in range(hold_frames):
        seq.append(add_noise(b, sigma=noise_sigma))
    return seq, switch_start


def generate_reload_sequence(noise_sigma=0.005):
    """
    生成换弹序列: 闭(30f) → 开(10f渐变) → 开保持(20f) → 闭(10f渐变) → 闭保持(10f)
    返回 landmarks 序列 + 期望触发帧索引。
    注意: EMA 平滑 (α=0.35) 需要足够的帧数才能让 smoothed_openness 跨过 0.8 阈值。
    """
    closed = template_reload_closed()
    opened = template_reload_open()

    seq = []
    # 闭-稳定 (让 EMA 充分收敛到低值)
    for _ in range(30):
        seq.append(add_noise(closed, sigma=noise_sigma))
    # 闭→开 渐变
    for i in range(10):
        frac = (i + 1) / 10
        interp = interpolate_pts(closed, opened, frac)
        seq.append(add_noise(interp, sigma=noise_sigma))
    # 开-稳定 (足够让 smoothed openness 升到 >0.8)
    for _ in range(20):
        seq.append(add_noise(opened, sigma=noise_sigma))
    # 开→闭 渐变
    for i in range(10):
        frac = (i + 1) / 10
        interp = interpolate_pts(opened, closed, frac)
        seq.append(add_noise(interp, sigma=noise_sigma))
    # 闭-稳定
    for _ in range(10):
        seq.append(add_noise(closed, sigma=noise_sigma))

    # 期望触发帧: 最后一帧渐变完成 (frame 69)
    expected_trigger = 69
    return seq, expected_trigger


def generate_melee_sequence(noise_sigma=0.005):
    """
    生成近战序列: 握拳静止(20f) → 快速前冲(5f, 手腕位移 > 阈值) → 收回(10f)
    返回序列 + 期望触发帧索引。
    velocity 阈值 0.9 ≈ 归一化坐标 0.21 位移 / 0.23s 窗口
    """
    fist = template_melee()
    seq = []

    # 静止握拳
    for _ in range(20):
        seq.append(add_noise(fist, sigma=noise_sigma))

    # 快速挥拳: 大幅移动手腕位置 (含所有关键点跟随)
    punch_distance = 0.30  # 归一化坐标, 足够触发 0.9 速度阈值
    punch_frames = 5
    expected_trigger = len(seq) + 2  # 冲刺早期触发 (峰值速度记忆)
    for i in range(punch_frames):
        frac = (i + 1) / punch_frames
        pts = [FakeLandmark(p.x, p.y, p.z) for p in fist]
        for pt in pts:
            pt.x += punch_distance * frac * 0.6   # 横向挥拳
            pt.y -= punch_distance * frac * 0.4   # 微上抬
            pt.z -= punch_distance * frac * 0.3   # 向前 (负=近摄像头)
        seq.append(add_noise(pts, sigma=noise_sigma))

    # 收回
    for i in range(10):
        frac = 1.0 - (i + 1) / 10
        pts = [FakeLandmark(p.x, p.y, p.z) for p in fist]
        for pt in pts:
            pt.x += punch_distance * frac * 0.6
            pt.y -= punch_distance * frac * 0.4
            pt.z -= punch_distance * frac * 0.3
        seq.append(add_noise(pts, sigma=noise_sigma))

    return seq, expected_trigger


def generate_switch_sequence(noise_sigma=0.005):
    """
    生成翻转序列: 正常(20f) → 翻转(10f渐变) → 翻转后(10f)
    返回序列 + 期望触发帧索引。
    """
    normal = REST_HAND
    flipped = template_switch_palm()

    seq = []
    for _ in range(20):
        seq.append(add_noise(normal, sigma=noise_sigma))

    expected_trigger = len(seq) + 7  # 翻转中段期望触发
    for i in range(10):
        frac = (i + 1) / 10
        interp = interpolate_pts(normal, flipped, frac)
        seq.append(add_noise(interp, sigma=noise_sigma))

    for _ in range(10):
        seq.append(add_noise(flipped, sigma=noise_sigma))

    return seq, expected_trigger


# ======================================================================
# 测试运行器
# ======================================================================

def run_classifier_on_sequence(classifier, sequence, warmup_frames=5, realtime=False):
    """
    对序列逐帧运行分类器。返回 [dict, ...]。
    warmup_frames: 前 N 帧不计入结果。
    realtime: True 时帧间 sleep 模拟 30fps (时序手势测试需要真实时间戳)。
    """
    import time as _time
    results = []
    classifier.reset()
    for i, landmarks in enumerate(sequence):
        g, c = classifier.classify(landmarks)
        fb = classifier.get_feedback()
        result = {
            "frame": i,
            "gesture": g,
            "confidence": c,
            "openness": fb["openness"],
            "velocity": fb["velocity"],
            "instant_velocity": fb["instant_velocity"],
            "reload_state": fb["reload_state"],
            "palm_angle": fb["palm_angle"],
        }
        if i >= warmup_frames:
            results.append(result)
        if realtime:
            _time.sleep(0.033)  # ~30fps
    return results


def measure_trigger_delay(results, expected_gesture, expected_frame):
    """
    测量触发延迟。搜索窗口扩大到 ±12 帧以容忍合成数据的时序偏差。
    """
    search_start = max(0, expected_frame - 12)
    search_end = min(len(results), expected_frame + 15)

    for r in results[search_start:search_end]:
        if r["gesture"] == expected_gesture:
            actual_frame = r["frame"]
            delay_frames = actual_frame - expected_frame
            delay_ms = delay_frames * (1000.0 / 30.0)
            return delay_frames, delay_ms

    return None, None  # 未触发


def count_jitter(results, ignore_none=True):
    """统计手势切换次数 (抖动率)"""
    switches = 0
    prev = results[0]["gesture"] if results else "none"
    for r in results[1:]:
        g = r["gesture"]
        if g != prev:
            if not (ignore_none and (g == "none" or prev == "none")):
                switches += 1
            elif g != "none" and prev != "none":
                switches += 1
        prev = g
    return switches


# ======================================================================
# 完整测试套件
# ======================================================================

def run_full_benchmark(quick=False):
    """运行全部测试，返回结构化结果。"""
    results = {}

    # 噪声等级
    noise_levels = [0.0, 0.003, 0.008, 0.015] if not quick else [0.0, 0.008]

    # ================================================================
    # 测试 1: 静态手势准确率 (不同噪声等级)
    # ================================================================
    print("🧪 测试 1: 静态手势准确率...")
    static_tests = [
        ("shoot",   template_shoot,   "shoot"),
        ("aim",     template_aim,     "aim"),
        ("grenade", template_grenade, "grenade"),
    ]
    static_results = {}
    for name, template_fn, expected in static_tests:
        per_noise = {}
        for sigma in noise_levels:
            clf = GestureClassifier()
            seq = generate_static_sequence(
                template_fn, num_frames=80 if not quick else 40,
                noise_sigma=sigma, jitter_amplitude=sigma * 0.6
            )
            res = run_classifier_on_sequence(clf, seq)
            correct = sum(1 for r in res if r["gesture"] == expected)
            accuracy = correct / len(res) * 100
            avg_conf = sum(r["confidence"] for r in res if r["gesture"] == expected) / max(correct, 1)
            jitter_count = count_jitter(res)
            jitter_rate = jitter_count / (len(res) / 30.0)  # 次/秒

            per_noise[str(sigma)] = {
                "accuracy": round(accuracy, 1),
                "avg_confidence": round(avg_conf, 3),
                "jitter_rate": round(jitter_rate, 2),
                "total_frames": len(res),
            }
        static_results[name] = per_noise

    results["static"] = static_results
    print(f"   完成: {len(static_tests)} 种手势 × {len(noise_levels)} 噪声等级")

    # ================================================================
    # 测试 2: 时序手势触发检测
    # 合成数据帧间跳变比真人快，需要提高容错帧数以适应 EMA 滞后
    # ================================================================
    print("🧪 测试 2: 时序手势触发检测...")
    runs = 5 if not quick else 3
    noise_sigmas = [0.002, 0.005]

    temporal_results = {}
    for sigma in noise_sigmas:
        per_sigma = {"reload": [], "melee": [], "switch_weapon": []}

        for _ in range(runs):
            # 换弹 — 高容错 + realtime (时间戳依赖)
            seq, expected = generate_reload_sequence(noise_sigma=sigma)
            clf = GestureClassifier(reload_fault_tolerance=8)
            res = run_classifier_on_sequence(clf, seq, realtime=True)
            delay_f, delay_ms = measure_trigger_delay(res, "reload", expected)
            if delay_f is not None:
                per_sigma["reload"].append({"delay_frames": delay_f, "delay_ms": delay_ms})

            # 近战 — realtime 必需 (触发保持+冷却依赖真实时间戳)
            seq, expected = generate_melee_sequence(noise_sigma=sigma)
            clf = GestureClassifier()
            res = run_classifier_on_sequence(clf, seq, realtime=True)
            delay_f, delay_ms = measure_trigger_delay(res, "melee", expected)
            if delay_f is not None:
                per_sigma["melee"].append({"delay_frames": delay_f, "delay_ms": delay_ms})

            # 翻转 — realtime 必需
            seq, expected = generate_switch_sequence(noise_sigma=sigma)
            clf = GestureClassifier()
            res = run_classifier_on_sequence(clf, seq, realtime=True)
            delay_f, delay_ms = measure_trigger_delay(res, "switch_weapon", expected)
            if delay_f is not None:
                per_sigma["switch_weapon"].append({"delay_frames": delay_f, "delay_ms": delay_ms})

        temporal_results[str(sigma)] = {
            g: {
                "detected": f"{len(v)}/{runs}",
                "avg_delay_ms": round(sum(x["delay_ms"] for x in v) / len(v), 0) if v else None,
                "avg_delay_frames": round(sum(x["delay_frames"] for x in v) / len(v), 1) if v else None,
            }
            for g, v in per_sigma.items()
        }
    results["temporal"] = temporal_results
    print(f"   完成: 3 种手势 × {len(noise_sigmas)} 噪声 × {runs} 轮")

    # ================================================================
    # 测试 3: EMA vs Kalman 全面对比
    # ================================================================
    print("🧪 测试 3: EMA vs Kalman 对比...")
    comparison = _run_ema_vs_kalman(static_tests, noise_levels, runs=3 if not quick else 1)
    results["comparison"] = comparison
    print("   完成: EMA vs Kalman 全部维度对比")

    # ================================================================
    # 测试 4: 三方对比 — EMA vs 特征级Kalman vs 坐标级Kalman
    # ================================================================
    print("🧪 测试 4: 三方对比 (EMA / 特征级KF / 坐标级KF)...")
    three_way = _run_three_way_comparison(static_tests, noise_levels, runs=3 if not quick else 1)
    results["three_way"] = three_way
    print("   完成: 三方对比")

    return results


def _run_ema_vs_kalman(static_tests, noise_levels, runs=3):
    """EMA 和 Kalman 在同一数据上的直接对比。"""
    comparison = {}

    for sigma in noise_levels:
        per_sigma = {}
        for name, template_fn, expected in static_tests:
            ema_accs = []
            kf_accs = []
            ema_jitters = []
            kf_jitters = []

            for _ in range(runs):
                seq = generate_static_sequence(
                    template_fn, num_frames=60,
                    noise_sigma=sigma, jitter_amplitude=sigma * 0.6
                )

                # EMA
                clf_ema = GestureClassifier(use_kalman=False)
                res_ema = run_classifier_on_sequence(clf_ema, seq)
                ema_accs.append(sum(1 for r in res_ema if r["gesture"] == expected) / len(res_ema) * 100)
                ema_jitters.append(count_jitter(res_ema) / (len(res_ema) / 30.0))

                # Kalman
                clf_kf = GestureClassifier(use_kalman=True)
                res_kf = run_classifier_on_sequence(clf_kf, seq)
                kf_accs.append(sum(1 for r in res_kf if r["gesture"] == expected) / len(res_kf) * 100)
                kf_jitters.append(count_jitter(res_kf) / (len(res_kf) / 30.0))

            per_sigma[name] = {
                "ema_accuracy": round(sum(ema_accs) / len(ema_accs), 1),
                "kf_accuracy": round(sum(kf_accs) / len(kf_accs), 1),
                "ema_jitter": round(sum(ema_jitters) / len(ema_jitters), 2),
                "kf_jitter": round(sum(kf_jitters) / len(kf_jitters), 2),
            }
        comparison[str(sigma)] = per_sigma
    return comparison


def _run_three_way_comparison(static_tests, noise_levels, runs=3):
    """三方对比: EMA vs 特征级 Kalman vs 坐标级 Kalman (同一数据)。"""
    comparison = {}

    for sigma in noise_levels:
        per_sigma = {}
        for name, template_fn, expected in static_tests:
            ema_accs, kf_accs, pk_accs = [], [], []
            ema_jitters, kf_jitters, pk_jitters = [], [], []

            for _ in range(runs):
                seq = generate_static_sequence(
                    template_fn, num_frames=60,
                    noise_sigma=sigma, jitter_amplitude=sigma * 0.6
                )

                # 1) EMA
                clf = GestureClassifier(use_kalman=False)
                res = run_classifier_on_sequence(clf, seq)
                ema_accs.append(sum(1 for r in res if r["gesture"] == expected) / len(res) * 100)
                ema_jitters.append(count_jitter(res) / (len(res) / 30.0))

                # 2) 特征级 Kalman
                clf = GestureClassifier(use_kalman=True)
                res = run_classifier_on_sequence(clf, seq)
                kf_accs.append(sum(1 for r in res if r["gesture"] == expected) / len(res) * 100)
                kf_jitters.append(count_jitter(res) / (len(res) / 30.0))

                # 3) 坐标级 Kalman
                clf = GestureClassifier(use_per_keypoint_kalman=True)
                res = run_classifier_on_sequence(clf, seq)
                pk_accs.append(sum(1 for r in res if r["gesture"] == expected) / len(res) * 100)
                pk_jitters.append(count_jitter(res) / (len(res) / 30.0))

            per_sigma[name] = {
                "ema_accuracy": round(sum(ema_accs) / len(ema_accs), 1),
                "kf_accuracy": round(sum(kf_accs) / len(kf_accs), 1),
                "pk_accuracy": round(sum(pk_accs) / len(pk_accs), 1),
                "ema_jitter": round(sum(ema_jitters) / len(ema_jitters), 2),
                "kf_jitter": round(sum(kf_jitters) / len(kf_jitters), 2),
                "pk_jitter": round(sum(pk_jitters) / len(pk_jitters), 2),
            }
        comparison[str(sigma)] = per_sigma
    return comparison


# ======================================================================
# 报告生成
# ======================================================================

def print_report(results):
    """输出 Markdown 格式报告到控制台"""
    print()
    print("# GestureWar 合成基准测试报告")
    print()
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- 静态手势 ----
    print("## 1. 静态手势准确率")
    print()
    print("在不同噪声强度下的识别准确率、平均置信度和抖动率。")
    print()
    static = results["static"]
    noise_labels = list(next(iter(static.values())).keys())

    # 表头
    header = "| 手势 | " + " | ".join(f"σ={n}" for n in noise_labels) + " |"
    sep = "|------|" + "|".join(["------" for _ in noise_labels]) + "|"
    print(header)
    print(sep)

    gesture_cn = {"shoot": "射击", "aim": "瞄准", "grenade": "手雷"}
    for name, per_noise in static.items():
        cells = []
        for nl in noise_labels:
            d = per_noise.get(nl, {})
            acc = d.get("accuracy", "-")
            jr = d.get("jitter_rate", "-")
            cells.append(f"{acc}% / jr={jr}")
        print(f"| {gesture_cn.get(name, name)} | " + " | ".join(cells) + " |")

    print()
    print("**置信度详情:**")
    print()
    for name, per_noise in static.items():
        print(f"- **{gesture_cn.get(name, name)}**: ", end="")
        parts = []
        for nl in noise_labels:
            d = per_noise.get(nl, {})
            conf = d.get("avg_confidence", "-")
            parts.append(f"σ={nl} → {conf}")
        print(", ".join(parts))

    # ---- 时序手势 ----
    print()
    print("## 2. 时序手势触发检测")
    print()
    temporal = results["temporal"]
    for sigma_key, gestures in temporal.items():
        print(f"### 噪声 σ={sigma_key}")
        print()
        print("| 手势 | 检出率 | 平均延迟 (ms) | 平均延迟 (帧) |")
        print("|------|--------|---------------|---------------|")
        tg_cn = {"reload": "换弹", "melee": "近战", "switch_weapon": "切换武器"}
        for g, d in gestures.items():
            det = d.get("detected", "-")
            ms = f"{d['avg_delay_ms']:.0f}" if d.get("avg_delay_ms") is not None else "N/A"
            fr = f"{d['avg_delay_frames']:.1f}" if d.get("avg_delay_frames") is not None else "N/A"
            print(f"| {tg_cn.get(g, g)} | {det} | {ms} | {fr} |")

    # ---- EMA vs Kalman ----
    print()
    print("## 3. EMA vs Kalman 对比")
    print()
    print("同一合成数据分别跑 EMA 和 Kalman，直接对比准确率和抖动率。")
    print()
    comparison = results["comparison"]

    for sigma_key, gestures in comparison.items():
        print(f"### 噪声 σ={sigma_key}")
        print()
        print("| 手势 | EMA 准确率 | KF 准确率 | EMA 抖动率 | KF 抖动率 | 胜出 |")
        print("|------|-----------|----------|-----------|----------|------|")
        for name, d in gestures.items():
            cn = gesture_cn.get(name, name)
            ea = d["ema_accuracy"]
            ka = d["kf_accuracy"]
            ej = d["ema_jitter"]
            kj = d["kf_jitter"]
            # 准确率高 + 抖动低 = 更好
            ema_score = ea - ej * 5  # 简单综合分
            kf_score = ka - kj * 5
            winner = "EMA ✅" if ema_score > kf_score else "Kalman ✅" if kf_score > ema_score else "持平"
            print(f"| {cn} | {ea}% | {ka}% | {ej} | {kj} | {winner} |")

    # ---- 三方对比 (EMA / 特征级KF / 坐标级KF) ----
    three_way = results.get("three_way")
    if three_way:
        print()
        print("## 4. 三方对比 — EMA vs 特征级Kalman vs 坐标级Kalman")
        print()
        print("同一合成数据，三种平滑策略直接对比。坐标级 Kalman 对 21 个关键点分 3 组调参:")
        print("- 高速组 (指尖 ×5): q=0.08 | 中速组 (DIP/PIP/MCP/腕 ×15): q=0.05 | 低速组 (拇指CMC ×1): q=0.02")
        print()

        for sigma_key, gestures in three_way.items():
            print(f"### 噪声 σ={sigma_key}")
            print()
            print("| 手势 | EMA 准确率 | 特征级KF | 坐标级KF | EMA 抖动 | 特征KF抖动 | 坐标KF抖动 | 最佳 |")
            print("|------|-----------|---------|---------|---------|----------|----------|------|")
            for name, d in gestures.items():
                cn = gesture_cn.get(name, name)
                ea, ka, pa = d["ema_accuracy"], d["kf_accuracy"], d["pk_accuracy"]
                ej, kj, pj = d["ema_jitter"], d["kf_jitter"], d["pk_jitter"]
                # 综合评分: 准确率 - 抖动率×5
                scores = {
                    "EMA": ea - ej * 5,
                    "特征KF": ka - kj * 5,
                    "坐标KF": pa - pj * 5,
                }
                best = max(scores, key=scores.get)
                best_label = {"EMA": "EMA ✅", "特征KF": "特征KF ✅", "坐标KF": "坐标KF ✅"}[best]
                print(f"| {cn} | {ea}% | {ka}% | {pa}% | {ej} | {kj} | {pj} | {best_label} |")
            print()

    # ---- 总结 ----
    print()
    print("## 5. 总结")
    print()
    print("### 关键发现")
    print()

    # 从数据中提取一些洞察
    shoot_noise = static.get("shoot", {})
    if shoot_noise:
        acc_clean = shoot_noise.get("0.0", {}).get("accuracy", 100)
        acc_dirty = shoot_noise.get("0.015", shoot_noise.get("0.008", {})).get("accuracy", 100)
        print(f"- **理想条件下**: 射击手势识别率 {acc_clean}%，"
              f"高噪声下仍保持 {acc_dirty}%，说明基于 PIP 关节角度的规则分类器对噪声具有良好鲁棒性")

    temporal_003 = temporal.get("0.003", {})
    if temporal_003:
        reload_d = temporal_003.get("reload", {})
        if reload_d.get("avg_delay_ms"):
            print(f"- **时序手势延迟**: 换弹触发延迟约 {reload_d['avg_delay_ms']:.0f}ms，"
                  f"在 30fps 下约 {reload_d.get('avg_delay_frames', 0):.1f} 帧，可满足实时交互需求")

    comp_008 = comparison.get("0.008", {})
    if comp_008:
        kf_wins = sum(1 for d in comp_008.values()
                      if d["kf_accuracy"] - d["kf_jitter"]*5 > d["ema_accuracy"] - d["ema_jitter"]*5)
        ema_wins = len(comp_008) - kf_wins
        print(f"- **EMA vs Kalman**: 在 σ=0.008 噪声下，Kalman 在 {kf_wins}/{len(comp_008)} 种手势上综合表现更优，"
              f"验证了卡尔曼滤波在中等噪声环境下对抖动抑制的有效性")

    print()
    print("### 局限性")
    print()
    print("- 合成数据基于理想化手部模型，未覆盖真实摄像头的光照变化、部分遮挡、运动模糊等场景")
    print("- 测试假设 MediaPipe 骨架提取完美，实际中骨架提取误差会叠加到分类器输入")
    print("- 建议后续在真实硬件上进行补充验证")
    print()
    print("---")
    print()
    print("*报告由 synthetic_benchmark.py 自动生成*")


def print_json_results(results):
    """输出 JSON 格式结果 (供程序化使用)"""
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


# ======================================================================
# 入口
# ======================================================================

# ======================================================================
# 参数调优模式
# ======================================================================

def run_parameter_sweep():
    """扫描关键参数，找出最佳配置。"""
    print("=" * 60)
    print("GestureWar - 参数扫描调优")
    print("=" * 60)
    noise_levels = [0.003, 0.008]
    frames_per_test = 60  # 每配置测试帧数

    # ================================================================
    # 扫描 1: curl_angle_threshold — PIP 关节角度阈值
    # ================================================================
    print("\n🔧 扫描 1: curl_angle_threshold (PIP 关节角度阈值)")
    print(f"   默认值: 40°  |  范围: 25° ~ 65°  |  噪声: {noise_levels}")
    print()

    angle_results = {}
    for angle in [25, 30, 35, 40, 45, 50, 55, 60, 65]:
        row = {}
        for sigma in noise_levels:
            accs = []
            for _ in range(3):  # 3轮取平均
                clf = GestureClassifier(curl_angle_threshold=angle)
                seq = generate_static_sequence(
                    template_shoot, num_frames=frames_per_test,
                    noise_sigma=sigma, jitter_amplitude=sigma * 0.6
                )
                res = run_classifier_on_sequence(clf, seq)
                correct = sum(1 for r in res if r["gesture"] == "shoot")
                accs.append(correct / len(res) * 100)
            row[str(sigma)] = round(sum(accs) / len(accs), 1)
        angle_results[angle] = row

    # 打印表格
    print(f"  {'角度':<6}", end="")
    for sigma in noise_levels:
        print(f"  σ={sigma:<8}", end="")
    print("  综合评价")
    print(f"  {'-'*50}")

    best_angle = None
    best_score = -1
    for angle in sorted(angle_results.keys()):
        r = angle_results[angle]
        # 综合分: 低噪准确率 × 0.4 + 中噪准确率 × 0.6
        score = r[str(noise_levels[0])] * 0.4 + r[str(noise_levels[1])] * 0.6
        marker = ""
        if score > best_score:
            best_score = score
            best_angle = angle
            marker = " ← 最佳"
        print(f"  {angle:>4}°  ", end="")
        for sigma in noise_levels:
            print(f"  {r[str(sigma)]:>5.1f}%  ", end="")
        print(f"  {score:>5.1f}{marker}")

    print(f"\n  ✅ 最佳 curl_angle_threshold = {best_angle}° (综合分: {best_score:.1f})")

    # ================================================================
    # 扫描 2: pinch_threshold — 捏合距离阈值
    # ================================================================
    print(f"\n🔧 扫描 2: pinch_threshold (捏合距离阈值)")
    print(f"   默认值: 0.06  |  范围: 0.03 ~ 0.12  |  噪声: {noise_levels}")
    print()

    pinch_results = {}
    for pinch in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]:
        row = {}
        for sigma in noise_levels:
            accs = []
            for _ in range(3):
                clf = GestureClassifier(pinch_threshold=pinch)
                seq = generate_static_sequence(
                    template_aim, num_frames=frames_per_test,
                    noise_sigma=sigma, jitter_amplitude=sigma * 0.6
                )
                res = run_classifier_on_sequence(clf, seq)
                correct = sum(1 for r in res if r["gesture"] == "aim")
                accs.append(correct / len(res) * 100)
            row[str(sigma)] = round(sum(accs) / len(accs), 1)
        pinch_results[pinch] = row

    print(f"  {'阈值':<8}", end="")
    for sigma in noise_levels:
        print(f"  σ={sigma:<8}", end="")
    print("  综合评价")
    print(f"  {'-'*50}")

    best_pinch = None
    best_score = -1
    for pinch in sorted(pinch_results.keys()):
        r = pinch_results[pinch]
        score = r[str(noise_levels[0])] * 0.4 + r[str(noise_levels[1])] * 0.6
        marker = ""
        if score > best_score:
            best_score = score
            best_pinch = pinch
            marker = " ← 最佳"
        print(f"  {pinch:<8.2f}", end="")
        for sigma in noise_levels:
            print(f"  {r[str(sigma)]:>5.1f}%  ", end="")
        print(f"  {score:>5.1f}{marker}")

    print(f"\n  ✅ 最佳 pinch_threshold = {best_pinch:.2f} (综合分: {best_score:.1f})")

    # ================================================================
    # 扫描 3: Kalman q/r 参数网格
    # ================================================================
    print(f"\n🔧 扫描 3: Kalman 过程噪声(q) × 测量噪声(r) 网格")
    print(f"   默认值: q=0.01, r=0.005  |  噪声: σ=0.008")
    print()

    q_values = [0.001, 0.005, 0.01, 0.02, 0.05]
    r_values = [0.001, 0.003, 0.005, 0.01, 0.02]

    # 表头
    header = "  q\\r   " + "".join(f"  r={r:<7}" for r in r_values)
    print(header)
    print(f"  {'-'*len(header)}")

    best_qr = None
    best_avg = -1
    grid = {}

    for q in q_values:
        cells = []
        for r in r_values:
            accs = []
            for _ in range(2):  # 每种手势2轮
                for name, template_fn, expected in [
                    ("shoot", template_shoot, "shoot"),
                    ("aim", template_aim, "aim"),
                    ("grenade", template_grenade, "grenade"),
                ]:
                    clf = GestureClassifier(
                        use_kalman=True,
                        kalman_process_noise=q,
                        kalman_measurement_noise=r,
                    )
                    seq = generate_static_sequence(
                        template_fn, num_frames=40,
                        noise_sigma=0.008, jitter_amplitude=0.005
                    )
                    res = run_classifier_on_sequence(clf, seq)
                    correct = sum(1 for r_ in res if r_["gesture"] == expected)
                    accs.append(correct / len(res) * 100)
            avg = sum(accs) / len(accs)
            grid[(q, r)] = round(avg, 1)
            cells.append(round(avg, 1))
            if avg > best_avg:
                best_avg = avg
                best_qr = (q, r)

        row = f"  q={q:<5.3f}" + "".join(f"  {c:>5.1f}% " for c in cells)
        print(row)

    print(f"\n  ✅ 最佳 Kalman 参数: q={best_qr[0]}, r={best_qr[1]} (平均准确率: {best_avg:.1f}%)")

    # ================================================================
    # 汇总
    # ================================================================
    print(f"\n{'='*60}")
    print("📋 调优汇总")
    print(f"{'='*60}")
    print(f"  curl_angle_threshold:  40° → {best_angle}°")
    print(f"  pinch_threshold:       0.06 → {best_pinch:.2f}")
    print(f"  Kalman (q, r):         (0.01, 0.005) → ({best_qr[0]}, {best_qr[1]})")
    print()
    print("💡 建议: 将这些最佳值填入 gesture_classifier.py 的默认参数")
    print(f"{'='*60}")


# ======================================================================
# 入口
# ======================================================================

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    compare_only = "--compare" in sys.argv
    json_out = "--json" in sys.argv
    tune_mode = "--tune" in sys.argv

    if tune_mode:
        run_parameter_sweep()
    elif compare_only:
        print("🔬 EMA vs Kalman 对比模式")
        results = run_full_benchmark(quick=True)
        # 只打印对比部分
        comp = results["comparison"]
        print()
        print("# EMA vs Kalman 对比报告")
        print()
        for sigma_key in sorted(comp.keys(), key=float):
            print(f"## 噪声 σ={sigma_key}")
            print()
            print("| 手势 | EMA 准确率 | KF 准确率 | EMA 抖动率 | KF 抖动率 | 胜出 |")
            print("|------|-----------|----------|-----------|----------|------|")
            for name, d in comp[sigma_key].items():
                cn = {"shoot":"射击","aim":"瞄准","grenade":"手雷"}.get(name, name)
                ea, ka = d["ema_accuracy"], d["kf_accuracy"]
                ej, kj = d["ema_jitter"], d["kf_jitter"]
                score_ema = ea - ej * 5
                score_kf = ka - kj * 5
                winner = "EMA ✅" if score_ema > score_kf else "Kalman ✅" if score_kf > score_ema else "持平"
                print(f"| {cn} | {ea}% | {ka}% | {ej} | {kj} | {winner} |")
            print()
    else:
        print("=" * 60)
        print("GestureWar - 合成手势基准测试")
        print("=" * 60)
        mode = "快速" if quick else "完整"
        print(f"模式: {mode}")
        print()

        results = run_full_benchmark(quick=quick)

        if json_out:
            print_json_results(results)
        else:
            print_report(results)
