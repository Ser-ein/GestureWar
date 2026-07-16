"""
GestureWar - 手势基准测试工具
===============================

用法:
  # 录制模式 — 打开摄像头，打标签录制测试数据
  python gesture_benchmark.py

  # 分析模式 — 从日志生成统计报告
  python gesture_benchmark.py --analyze <日志文件.jsonl>

  # 对比模式 — 比较两份日志 (如 EMA vs Kalman)
  python gesture_benchmark.py --compare <日志1.jsonl> <日志2.jsonl>

录制时按键:
  [1] 静止      [2] 射击测试    [3] 瞄准测试    [4] 手雷测试
  [5] 换弹测试  [6] 近战测试    [7] 翻转测试    [0] 清除标签
  [k] 切换 Kalman/EMA          [r] 打印即时统计
  [q] 退出并保存日志
"""

import cv2
import time
import sys
import json
import os
from collections import defaultdict, Counter

from hand_tracking_0_10_35 import HandTracker

# ---------------------------------------------------------------------------
# 标签定义
# ---------------------------------------------------------------------------
LABEL_MAP = {
    ord('0'): ("rest",        "静止"),
    ord('1'): ("rest",        "静止"),
    ord('2'): ("shoot_test",  "射击测试"),
    ord('3'): ("aim_test",    "瞄准测试"),
    ord('4'): ("grenade_test","手雷测试"),
    ord('5'): ("reload_test", "换弹测试"),
    ord('6'): ("melee_test",  "近战测试"),
    ord('7'): ("switch_test", "翻转测试"),
}

GESTURE_NAMES_CN = {
    "none": "无手势", "shoot": "射击", "aim": "瞄准",
    "grenade": "手雷", "reload": "换弹", "melee": "近战",
    "switch_weapon": "切换武器",
}

# ---------------------------------------------------------------------------
# 录制模式
# ---------------------------------------------------------------------------

def record_session(output_dir="benchmark_logs"):
    """录制手势测试会话，逐帧记录分类器输出 + 用户标签。"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GestureWar - 手势基准测试 (录制模式)")
    print("=" * 60)
    print("按键说明:")
    print("  [1] 静止    [2] 射击  [3] 瞄准  [4] 手雷")
    print("  [5] 换弹    [6] 近战  [7] 翻转  [0] 清除标签")
    print("  [k] 切换 Kalman/EMA   [r] 即时统计   [q] 退出")
    print("=" * 60)

    tracker = HandTracker(max_hands=2, enable_gesture=True, use_kalman=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # --- 录制状态 ---
    frames = []              # 所有帧数据 (内存中, 最后写入文件)
    current_label = "rest"
    current_label_cn = "静止"
    session_start = time.time()
    smoother_mode = "EMA"

    print("\n摄像头已打开，开始录制...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, hand_data = tracker.process_frame(frame)

            # ---- 提取每帧的分析数据 ----
            frame_record = {
                "frame_id": hand_data["frame_id"],
                "timestamp": hand_data["timestamp"],
                "label": current_label,
                "smoother": smoother_mode,
                "num_hands": hand_data["num_hands"],
                "hands": [],
            }

            for hand in hand_data.get("hands", []):
                g = hand.get("gesture", "none")
                gc = hand.get("gesture_confidence", 0.0)

                # 从分类器获取内部状态
                fb = {}
                if tracker.gesture_classifier:
                    fb = tracker.gesture_classifier.get_feedback()

                frame_record["hands"].append({
                    "id": hand["id"],
                    "gesture": g,
                    "confidence": round(gc, 4),
                    "openness": fb.get("openness", 0.0),
                    "velocity": fb.get("velocity", 0.0),
                    "instant_velocity": fb.get("instant_velocity", 0.0),
                    "peak_velocity": fb.get("peak_velocity", 0.0),
                    "reload_state": fb.get("reload_state", "idle"),
                    "palm_angle": fb.get("palm_angle", 0.0),
                })

            frames.append(frame_record)

            # ---- HUD: 绘制标签和录制信息 ----
            h, w = processed_frame.shape[:2]
            elapsed = time.time() - session_start

            # 标签指示器
            label_colors = {
                "rest": (100, 100, 100),
                "shoot_test": (0, 165, 255),
                "aim_test": (255, 255, 0),
                "grenade_test": (0, 0, 255),
                "reload_test": (255, 0, 255),
                "melee_test": (0, 255, 255),
                "switch_test": (255, 128, 0),
            }
            label_color = label_colors.get(current_label, (200, 200, 200))

            # 顶部状态栏背景
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 48), (30, 30, 30), -1)
            processed_frame = cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0)

            cv2.putText(processed_frame, f"REC  |  {smoother_mode}  |  "
                        f"Label: {current_label_cn}  |  "
                        f"Frames: {len(frames)}  |  {elapsed:.0f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1)

            cv2.imshow('GestureWar - Benchmark Recording', processed_frame)

            # ---- 键盘控制 ----
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key in LABEL_MAP:
                current_label, current_label_cn = LABEL_MAP[key]
                print(f"[标签切换] → {current_label_cn} ({current_label})")
                # 在帧记录中插入标记
                frame_record["label_change"] = current_label
            elif key == ord('k'):
                if tracker.gesture_classifier:
                    new_mode = tracker.gesture_classifier.toggle_smoother()
                    smoother_mode = new_mode.upper()
                    print(f"[平滑器切换] → {smoother_mode}")
            elif key == ord('r'):
                _print_quick_stats(frames, tracker)

    except KeyboardInterrupt:
        print("\n录制被用户中断")
    except Exception as e:
        print(f"录制错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()

    # ---- 保存日志 ----
    if not frames:
        print("未录制任何数据，退出。")
        return None

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/benchmark_{smoother_mode}_{timestamp_str}.jsonl"
    with open(filename, 'w', encoding='utf-8') as f:
        for frame_record in frames:
            f.write(json.dumps(frame_record, ensure_ascii=False) + '\n')

    duration = frames[-1]["timestamp"] - frames[0]["timestamp"]
    print(f"\n✅ 录制完成！")
    print(f"   文件: {filename}")
    print(f"   帧数: {len(frames)}")
    print(f"   时长: {duration:.1f}s")
    print(f"   平滑器: {smoother_mode}")

    # 自动生成简要报告
    _print_quick_stats(frames, tracker)
    print(f"\n💡 运行分析: python gesture_benchmark.py --analyze {filename}")

    return filename


def _print_quick_stats(frames, tracker):
    """录制中即时打印统计摘要"""
    if len(frames) < 10:
        print("  (数据太少，至少需要 10 帧)")
        return

    # 按标签分组
    label_frames = defaultdict(list)
    for fr in frames:
        label_frames[fr["label"]].append(fr)

    # 全局手势分布
    gesture_counter = Counter()
    gesture_conf_sum = defaultdict(float)
    gesture_conf_count = defaultdict(int)
    transitions = 0
    prev_gesture = None

    for fr in frames:
        for hand in fr.get("hands", []):
            g = hand["gesture"]
            gesture_counter[g] += 1
            gesture_conf_sum[g] += hand["confidence"]
            gesture_conf_count[g] += 1
            if g != prev_gesture and prev_gesture is not None:
                transitions += 1
            prev_gesture = g

    total_frames = len(frames)
    duration = frames[-1]["timestamp"] - frames[0]["timestamp"]
    jitter_rate = transitions / max(duration, 0.01)

    si = tracker.gesture_classifier.get_smoother_info() if tracker.gesture_classifier else {}

    print(f"\n{'='*50}")
    print(f"📊 即时统计 (共 {total_frames} 帧, {duration:.1f}s)")
    print(f"   平滑器: {si.get('type', 'N/A')}")
    print(f"   手势切换次数: {transitions}")
    print(f"   抖动率: {jitter_rate:.2f} 次/秒 {'⚠ 偏高' if jitter_rate > 3 else '✅ 稳定' if jitter_rate < 1 else ''}")
    print(f"   手势分布:")
    for gesture, count in gesture_counter.most_common():
        pct = count / total_frames * 100
        avg_conf = gesture_conf_sum[gesture] / max(gesture_conf_count[gesture], 1)
        cn = GESTURE_NAMES_CN.get(gesture, gesture)
        print(f"     {cn:<6} {pct:5.1f}%  (平均置信度: {avg_conf:.2f})")

    # 每标签段统计
    print(f"\n  分段统计:")
    for label, group in sorted(label_frames.items()):
        if label == "rest":
            continue
        gc = Counter()
        for fr in group:
            for hand in fr.get("hands", []):
                gc[hand["gesture"]] += 1
        total = sum(gc.values())
        correct = gc.get(label.replace("_test", ""), 0)
        # 映射: reload_test → reload, switch_test → switch_weapon
        expected_gesture = label.replace("_test", "")
        if expected_gesture == "switch":
            expected_gesture = "switch_weapon"
            correct = gc.get("switch_weapon", 0)
        rate = correct / max(total, 1) * 100
        cn = GESTURE_NAMES_CN.get(expected_gesture, expected_gesture)
        print(f"     {cn}: {correct}/{total} ({rate:.0f}%)")


# ---------------------------------------------------------------------------
# 分析模式
# ---------------------------------------------------------------------------

def analyze_log(log_path):
    """分析录制日志，生成详细报告。"""
    if not os.path.exists(log_path):
        print(f"错误: 文件不存在 — {log_path}")
        return

    print(f"正在加载: {log_path}")
    frames = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))

    if not frames:
        print("日志为空。")
        return

    report = _compute_metrics(frames, os.path.basename(log_path))
    _print_report(report)
    return report


def compare_logs(log1_path, log2_path):
    """对比两份日志 (如 EMA vs Kalman)。"""
    logs = []
    for path in [log1_path, log2_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在 — {path}")
            return
        frames = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        logs.append(frames)

    report1 = _compute_metrics(logs[0], os.path.basename(log1_path))
    report2 = _compute_metrics(logs[1], os.path.basename(log2_path))

    _print_comparison(report1, report2)
    return report1, report2


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def _compute_metrics(frames, name):
    """从帧数据中提取所有数值指标。"""
    total = len(frames)
    duration = frames[-1]["timestamp"] - frames[0]["timestamp"]

    # 手势计数与置信度
    gesture_frames = Counter()
    gesture_confs = defaultdict(list)

    # 抖动分析
    transitions = 0
    gesture_events = []  # (frame_id, timestamp, old_gesture, new_gesture)
    prev_g = None
    prev_g_time = None
    hold_durations = defaultdict(list)  # gesture → [durations]

    # 张开度/速度序列
    openness_seq = []
    velocity_seq = []
    inst_velocity_seq = []

    # 按标签段统计
    label_segments = defaultdict(lambda: {"frames": 0, "gestures": Counter(), "confs": defaultdict(list)})
    current_seg_start = None

    last_g_time = frames[0]["timestamp"] if frames else 0

    for i, fr in enumerate(frames):
        hands = fr.get("hands", [])
        label = fr.get("label", "rest")
        ts = fr["timestamp"]

        seg = label_segments[label]
        seg["frames"] += 1

        if not hands:
            if prev_g is not None and prev_g != "none":
                hold_durations[prev_g].append(ts - last_g_time)
            prev_g = None
            prev_g_time = None
            continue

        for hand in hands:
            g = hand["gesture"]
            c = hand["confidence"]
            gesture_frames[g] += 1
            gesture_confs[g].append(c)
            seg["gestures"][g] += 1
            seg["confs"][g].append(c)

            openness_seq.append(hand.get("openness", 0))
            velocity_seq.append(hand.get("velocity", 0))
            inst_velocity_seq.append(hand.get("instant_velocity", 0))

            # 手势切换检测
            if g != prev_g and prev_g is not None:
                transitions += 1
                gesture_events.append({
                    "frame_id": fr["frame_id"],
                    "timestamp": ts,
                    "from": prev_g,
                    "to": g,
                    "confidence": c,
                })
                # 记录上一个手势保持了多久
                if prev_g_time is not None:
                    hold_durations[prev_g].append(ts - prev_g_time)
                prev_g_time = ts

            if g != prev_g:
                prev_g_time = ts

            prev_g = g
            last_g_time = ts

    # 记录最后一个手势的持续时间
    if prev_g is not None and prev_g_time is not None:
        hold_durations[prev_g].append(frames[-1]["timestamp"] - prev_g_time)

    # 抖动率
    jitter_rate = transitions / max(duration, 0.01)

    # 手势分布百分比
    total_gesture_frames = sum(gesture_frames.values())
    gesture_pct = {g: c / max(total_gesture_frames, 1) * 100
                   for g, c in gesture_frames.items()}

    # 置信度统计
    confidence_stats = {}
    for g, confs in gesture_confs.items():
        if confs:
            confidence_stats[g] = {
                "mean": sum(confs) / len(confs),
                "min": min(confs),
                "max": max(confs),
                "std": (sum((x - sum(confs)/len(confs))**2 for x in confs) / len(confs)) ** 0.5,
                "count": len(confs),
            }

    # 持续时间统计
    hold_stats = {}
    for g, durs in hold_durations.items():
        if durs:
            hold_stats[g] = {
                "mean": sum(durs) / len(durs),
                "min": min(durs),
                "max": max(durs),
                "count": len(durs),
            }

    # 张开度/速度统计
    openness_std = (sum((x - sum(openness_seq)/len(openness_seq))**2
                        for x in openness_seq) / max(len(openness_seq), 1)) ** 0.5 if openness_seq else 0

    # 每标签段的命中率
    segment_accuracy = {}
    for label, seg in label_segments.items():
        if label == "rest" or seg["frames"] < 5:
            continue
        expected = label.replace("_test", "")
        if expected == "switch":
            expected = "switch_weapon"
        correct = seg["gestures"].get(expected, 0)
        total_in_seg = sum(seg["gestures"].values())
        segment_accuracy[label] = {
            "expected": expected,
            "correct": correct,
            "total": total_in_seg,
            "rate": correct / max(total_in_seg, 1) * 100,
        }

    return {
        "name": name,
        "total_frames": total,
        "duration": duration,
        "gesture_frames": dict(gesture_frames),
        "gesture_pct": gesture_pct,
        "confidence_stats": confidence_stats,
        "hold_stats": hold_stats,
        "transitions": transitions,
        "jitter_rate": jitter_rate,
        "openness_std": openness_std,
        "segment_accuracy": segment_accuracy,
        "gesture_events": gesture_events[:20],  # 只保留前20个切换事件
    }


def _print_report(report):
    """格式化输出单份报告。"""
    name = report["name"]
    print(f"\n{'='*60}")
    print(f"📋 基准测试报告: {name}")
    print(f"{'='*60}")
    print(f"  总帧数: {report['total_frames']}")
    print(f"  时长:   {report['duration']:.1f}s")
    print(f"  抖动率: {report['jitter_rate']:.2f} 次/秒"
          f"{'  ✅ 稳定' if report['jitter_rate'] < 1 else '  ⚠ 偏高' if report['jitter_rate'] > 3 else ''}")

    print(f"\n  📊 手势分布:")
    for g, pct in sorted(report["gesture_pct"].items(), key=lambda x: x[1], reverse=True):
        cn = GESTURE_NAMES_CN.get(g, g)
        cs = report["confidence_stats"].get(g, {})
        mean_c = cs.get("mean", 0)
        std_c = cs.get("std", 0)
        print(f"     {cn:<8} {pct:5.1f}%  "
              f"置信度 μ={mean_c:.2f} σ={std_c:.3f}  (n={cs.get('count', 0)})")

    print(f"\n  ⏱ 平均持续时间:")
    for g, hs in sorted(report["hold_stats"].items(),
                         key=lambda x: x[1].get("mean", 0), reverse=True):
        cn = GESTURE_NAMES_CN.get(g, g)
        print(f"     {cn:<8} {hs['mean']*1000:.0f}ms  (min={hs['min']*1000:.0f}ms, n={hs['count']})")

    if report["segment_accuracy"]:
        print(f"\n  🏷 分段准确率:")
        for label, acc in report["segment_accuracy"].items():
            ecn = GESTURE_NAMES_CN.get(acc["expected"], acc["expected"])
            print(f"     {ecn:<8} {acc['correct']}/{acc['total']} = {acc['rate']:.0f}%")

    print(f"\n  📐 张开度标准差: {report['openness_std']:.4f}"
          f"  (越小越稳定)")

    # 抖动事件样本
    if report["gesture_events"]:
        print(f"\n  🔄 首 5 个手势切换事件:")
        for ev in report["gesture_events"][:5]:
            fcn = GESTURE_NAMES_CN.get(ev["from"], ev["from"])
            tcn = GESTURE_NAMES_CN.get(ev["to"], ev["to"])
            print(f"     #{ev['frame_id']:>5}  {fcn} → {tcn}  (置信度: {ev['confidence']:.2f})")

    print(f"\n{'='*60}\n")


def _print_comparison(r1, r2):
    """格式化输出对比报告。"""
    print(f"\n{'='*70}")
    print(f"📋 对比分析")
    print(f"{'='*70}")
    print(f"  {'指标':<24} {r1['name']:<22} {r2['name']:<22}")
    print(f"  {'-'*66}")

    def compare_row(label, v1, v2, fmt=".2f", lower_better=True):
        if isinstance(v1, str):
            print(f"  {label:<24} {v1:<22} {v2:<22}")
            return
        diff = v1 - v2
        if lower_better:
            better = "← 更好" if diff < 0 else "→ 更好" if diff > 0 else "持平"
        else:
            better = "→ 更好" if diff > 0 else "← 更好" if diff < 0 else "持平"
        arrow = "←" if diff < 0 else "→" if diff > 0 else "="
        print(f"  {label:<24} {v1:{fmt}} {arrow} {v2:{fmt}}  {better}")

    compare_row("总帧数", r1["total_frames"], r2["total_frames"], "d")
    compare_row("时长 (秒)", r1["duration"], r2["duration"], ".1f")
    compare_row("抖动率 (次/秒)", r1["jitter_rate"], r2["jitter_rate"], ".2f", True)
    compare_row("张开度标准差", r1["openness_std"], r2["openness_std"], ".4f", True)

    # 对比每类手势
    all_gestures = set(r1["confidence_stats"].keys()) | set(r2["confidence_stats"].keys())
    for g in sorted(all_gestures):
        cn = GESTURE_NAMES_CN.get(g, g)
        cs1 = r1["confidence_stats"].get(g, {})
        cs2 = r2["confidence_stats"].get(g, {})
        m1 = cs1.get("mean", 0)
        m2 = cs2.get("mean", 0)
        if cs1 and cs2:
            compare_row(f"  {cn} 平均置信度", m1, m2, ".3f", False)

        hs1 = r1["hold_stats"].get(g, {})
        hs2 = r2["hold_stats"].get(g, {})
        h1 = hs1.get("mean", 0) * 1000
        h2 = hs2.get("mean", 0) * 1000
        if hs1 and hs2:
            compare_row(f"  {cn} 平均持续 (ms)", h1, h2, ".0f")

    print(f"\n💡 提示: 抖动率越低越好 (更稳定)，平均置信度越高越好，")
    print(f"   持续时间越长说明手势保持越稳定。")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--analyze" in sys.argv:
        idx = sys.argv.index("--analyze")
        if idx + 1 < len(sys.argv):
            analyze_log(sys.argv[idx + 1])
        else:
            print("用法: python gesture_benchmark.py --analyze <日志文件.jsonl>")
    elif "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        if idx + 2 < len(sys.argv):
            compare_logs(sys.argv[idx + 1], sys.argv[idx + 2])
        else:
            print("用法: python gesture_benchmark.py --compare <日志1.jsonl> <日志2.jsonl>")
    else:
        record_session()
