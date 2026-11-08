import argparse
import csv
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

OUT_DIR = "hasil"
CAP_DIR = os.path.join(OUT_DIR, "captures")
EVAL_CSV = os.path.join(OUT_DIR, "eval_frames.csv")
BENCH_CSV = os.path.join(OUT_DIR, "benchmark_frames.csv")
REPORT_MD = os.path.join(OUT_DIR, "laporan_tabel.md")

SCALE_A = 1.5          # faktor skala ROI
BLUR_K = (15, 15)      # kernel Gaussian Blur
CANNY = (100, 200)     # threshold Canny
BENCH_FRAMES = 300     # jumlah frame benchmark
WARMUP = 30            # frame pemanasan sebelum merekam benchmark
COUNTDOWN = 3          # detik persiapan sebelum tiap gesture direkam

GESTURES = {
    1: ("Gesture 1", "Thumb + Index Pinch", "Dynamic ROI Resizing / Crop"),
    2: ("Gesture 2", "Middle Extended", "Gaussian Blur Filter"),
    3: ("Gesture 3", "Ring Extended", "Canny Edge Detection"),
    4: ("Gesture 4", "Pinky Extended", "Color Space Conversion (HSV/Gray)"),
}
LIGHTS = ["bright_300lux", "moderate_150lux", "dim_30lux"]

# ----------------------------------------------------------------------------
# Logika inti
# ----------------------------------------------------------------------------
def classify_gesture(lm):
    """Jari terulur jika y ujung (tip) < y sendi PIP (koordinat y menuju bawah)."""
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y
    state = (middle, ring, pinky)
    return {(False, False, False): 1, (True, False, False): 2,
            (False, True, False): 3, (False, False, True): 4}.get(state, 0)


def compute_roi(lm, W, H, a=SCALE_A):
    """ROI persegi dari jarak Euclidean P4-P8 dan titik tengahnya."""
    x4, y4 = lm[4].x * W, lm[4].y * H
    x8, y8 = lm[8].x * W, lm[8].y * H
    d = math.hypot(x8 - x4, y8 - y4)
    cx, cy = (x4 + x8) / 2, (y4 + y8) / 2
    half = a * d / 2
    xmin, xmax = int(max(0, cx - half)), int(min(W, cx + half))
    ymin, ymax = int(max(0, cy - half)), int(min(H, cy + half))
    return xmin, ymin, xmax, ymax, d, (x4, y4), (x8, y8)


def apply_filter(roi, gesture, use_hsv=False):
    if gesture == 2:
        return cv2.GaussianBlur(roi, BLUR_K, 0)
    if gesture == 3:
        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), *CANNY)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if gesture == 4:
        if use_hsv:
            return cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return roi


# ----------------------------------------------------------------------------
# Laporan tabel (Tabel 1, 2, 3 + pengecekan klaim di teks)
# ----------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def build_report():
    L = []
    ev = read_csv(EVAL_CSV)
    bn = read_csv(BENCH_CSV)
    overall_acc = overall_fps = None
    acc_by_g = {}

    # ---- Tabel 1
    L.append("## Tabel 1. Recognition accuracy across gesture modes\n")
    if ev:
        L.append("| Gesture State | Active Finger Combination | Target ROI Action | Total Trials | Correct Classification | Accuracy (%) |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        tt = tc = 0
        for g, (name, combo, action) in GESTURES.items():
            rows = [r for r in ev if int(r["gesture_true"]) == g]
            n = len(rows)
            c = sum(int(r["correct"]) for r in rows)
            acc = 100 * c / n if n else float("nan")
            acc_by_g[g] = acc
            tt += n
            tc += c
            L.append(f"| {name} | {combo} | {action} | {n} | {c} | {fmt(acc)} |")
        overall_acc = 100 * tc / tt if tt else float("nan")
        L.append(f"| **Overall Average** | **--** | **--** | **{tt}** | **{tc}** | **{fmt(overall_acc)}** |")

        L.append("\n### Akurasi per kondisi cahaya (untuk subbab 3.3)\n")
        L.append("| Lighting | Total Trials | Correct | Accuracy (%) |")
        L.append("| --- | --- | --- | --- |")
        for lt in LIGHTS:
            rows = [r for r in ev if r["lighting"] == lt]
            if rows:
                c = sum(int(r["correct"]) for r in rows)
                L.append(f"| {lt} | {len(rows)} | {c} | {fmt(100 * c / len(rows))} |")
        L.append("\n### Confusion (baris = gesture sebenarnya, kolom = prediksi; 0 = tidak terdeteksi/tidak valid)\n")
        L.append("| True \\ Pred | 0 | 1 | 2 | 3 | 4 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for g in GESTURES:
            rows = [r for r in ev if int(r["gesture_true"]) == g]
            cnt = [sum(1 for r in rows if int(r["gesture_pred"]) == p) for p in range(5)]
            L.append(f"| {g} | " + " | ".join(map(str, cnt)) + " |")
    else:
        L.append("_Belum ada data. Jalankan evaluasi dengan tombol E._")

    # ---- Tabel 2
    L.append("\n## Tabel 2. Frame rate performance across system execution modules\n")
    modules = {}
    if bn:
        def col(name, filt=None):
            out = []
            for r in bn:
                if filt and r["filter"] != filt:
                    continue
                v = r[name]
                if v != "":
                    out.append(float(v))
            return out
        modules = {
            "Landmark Tracking": ("MediaPipe 21 keypoint estimation", mean(col("landmark_ms"))),
            "ROI Extraction": ("Euclidean distance & cropping", mean(col("roi_ms"))),
            "Filter 1 Application": ("Gaussian Blur (15x15 Kernel)", mean(col("filter_ms", "blur"))),
            "Filter 2 Application": ("Canny Edge Gradient Filter", mean(col("filter_ms", "canny"))),
            "Filter 3 Application": ("BGR to Grayscale / HSV Conversion", mean(col("filter_ms", "gray"))),
            "Overall Pipeline": ("Full End-to-End Processing Feed", mean(col("total_ms"))),
        }
        L.append("| Execution Module | Pipeline Operations | Average Latency (ms) | Average Frame Rate (FPS) |")
        L.append("| --- | --- | --- | --- |")
        for k, (ops, ms) in modules.items():
            fps = 1000 / ms if ms and not math.isnan(ms) else float("nan")
            b = "**" if k == "Overall Pipeline" else ""
            L.append(f"| {b}{k}{b} | {ops} | {b}{fmt(ms, 3)}{b} | {b}{fmt(fps, 1)}{b} |")
            if k == "Overall Pipeline":
                overall_fps = fps
    else:
        L.append("_Belum ada data. Jalankan benchmark dengan tombol B._")

    # ---- Tabel 3 (baris Proposed)
    L.append("\n## Tabel 3. Baris 'Proposed Framework'\n")
    L.append("| Method / Framework | Hardware Requirement | Tracking Mechanism | ROI Dynamism | Recognition Accuracy (%) | Average FPS |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    L.append(f"| Proposed Framework | Intel i5-11400H CPU | Landmark Vector Geometry | Fully Dynamic | "
             f"{fmt(overall_acc, 1)}% | {fmt(overall_fps, 1)} |")

    # ---- Cek klaim
    L.append("\n## Pengecekan angka yang tertulis di naskah (sesuaikan teks jika berbeda)\n")
    L.append("| Klaim di naskah | Hasil program |")
    L.append("| --- | --- |")
    L.append(f"| Akurasi keseluruhan 96.2% | {fmt(overall_acc, 1)}% |")
    L.append(f"| Pinch (Gesture 1) tertinggi 97.6% | G1 = {fmt(acc_by_g.get(1, float('nan')), 1)}%, "
             f"tertinggi: G{max(acc_by_g, key=acc_by_g.get) if acc_by_g else '-'} |")
    L.append(f"| Canny (Gesture 3) 94.8% | {fmt(acc_by_g.get(3, float('nan')), 1)}% |")
    dim = [r for r in ev if r["lighting"] == "dim_30lux"]
    dim_acc = 100 * sum(int(r["correct"]) for r in dim) / len(dim) if dim else float("nan")
    L.append(f"| Cahaya redup (30 lux) 91.2% | {fmt(dim_acc, 1)}% |")
    L.append(f"| Rata-rata 29.4 FPS | {fmt(overall_fps, 1)} FPS |")
    if modules:
        roi_ms = modules["ROI Extraction"][1]
        fl = [modules[k][1] for k in ("Filter 1 Application", "Filter 2 Application", "Filter 3 Application")]
        over = roi_ms + max(x for x in fl if not math.isnan(x)) if any(not math.isnan(x) for x in fl) else float("nan")
        L.append(f"| Overhead ROI + filter < 3.1 ms | {fmt(over, 3)} ms (ROI + filter terberat) |")

    text = "\n".join(L)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text + f"\n\n[Laporan disimpan: {REPORT_MD}]")


# ----------------------------------------------------------------------------
# Utilitas tampilan
# ----------------------------------------------------------------------------
def put(img, text, y, color=(0, 255, 0), scale=0.6):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_composite(last):
    labels = {1: "(a) Pinch ROI", 2: "(b) Gaussian blur", 3: "(c) Canny edge", 4: "(d) Grayscale"}
    tiles = []
    for g in (1, 2, 3, 4):
        t = cv2.resize(last[g], (640, 360))
        put(t, labels[g], 28, (0, 255, 255), 0.8)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    path = os.path.join(CAP_DIR, "fig3_composite.png")
    cv2.imwrite(path, grid)
    return path


# ----------------------------------------------------------------------------
# Program utama
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fpg", type=int, default=250, help="frame per gesture per evaluasi")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    os.makedirs(CAP_DIR, exist_ok=True)
    if args.reset:
        for p in (EVAL_CSV, BENCH_CSV):
            if os.path.exists(p):
                os.remove(p)
        print("Data eval/benchmark lama dihapus.")
    if args.report:
        build_report()
        return

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 144)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Coba --cam 1.")

    light_idx, use_hsv, show_lm = 0, False, False
    ev = None      # state evaluasi
    bn = None      # state benchmark
    last_cap = {}  # screenshot terakhir per gesture untuk Gambar 3
    fps_s, t_prev = 0.0, time.perf_counter()

    while True:
        t_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # 1) Landmark tracking (termasuk konversi BGR->RGB)
        t0 = time.perf_counter()
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm_ms = (time.perf_counter() - t0) * 1000

        gesture, roi_ms, filt_ms, filt_name = 0, None, None, ""
        disp = frame.copy()
        lm = res.multi_hand_landmarks[0].landmark if res.multi_hand_landmarks else None

        if lm is not None:
            gesture = classify_gesture(lm)

            # 2) Ekstraksi ROI
            t0 = time.perf_counter()
            xmin, ymin, xmax, ymax, d, p4, p8 = compute_roi(lm, W, H)
            roi = frame[ymin:ymax, xmin:xmax]
            roi_ms = (time.perf_counter() - t0) * 1000

            # Saat benchmark, filter digilir 2->3->4 agar ketiganya terukur
            g_apply = gesture
            if bn is not None and bn["i"] >= 0:
                g_apply = (2, 3, 4)[bn["i"] % 3]

            # 3) Filter dalam ROI
            if roi.size > 100 and g_apply in (2, 3, 4):
                t0 = time.perf_counter()
                out = apply_filter(roi, g_apply, use_hsv)
                filt_ms = (time.perf_counter() - t0) * 1000
                filt_name = {2: "blur", 3: "canny", 4: "gray"}[g_apply]
                disp[ymin:ymax, xmin:xmax] = out
            color = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 128, 0), 3: (0, 200, 255), 4: (255, 0, 255)}[gesture]
            cv2.rectangle(disp, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.line(disp, (int(p4[0]), int(p4[1])), (int(p8[0]), int(p8[1])), (255, 255, 255), 2)
            for p in (p4, p8):
                cv2.circle(disp, (int(p[0]), int(p[1])), 8, (0, 255, 255), -1)
            if show_lm:
                mp_draw.draw_landmarks(disp, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                cv2.putText(disp, "P4", (int(p4[0]) + 10, int(p4[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(disp, "P8", (int(p8[0]) + 10, int(p8[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(disp, f"d={d:.0f}px", (int((p4[0] + p8[0]) / 2), int((p4[1] + p8[1]) / 2) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ---------------- HUD & state machine ----------------
        gname = GESTURES[gesture][2] if gesture else "tidak ada / tidak valid"
        put(disp, f"FPS {fps_s:4.1f} | cahaya: {LIGHTS[light_idx]} | G{gesture}: {gname}", 24)

        if ev is not None:
            tg = ev["queue"][0]
            if ev["phase"] == "countdown":
                left = COUNTDOWN - (time.time() - ev["t0"])
                put(disp, f"EVAL: siapkan Gesture {tg} - {GESTURES[tg][1]} ({max(0, left):.1f}s)", 55, (0, 255, 255), 0.8)
                if left <= 0:
                    ev["phase"], ev["n"] = "record", 0
            else:
                ev["rows"].append([datetime.now().isoformat(timespec="seconds"), LIGHTS[light_idx],
                                   tg, gesture, int(gesture == tg)])
                ev["n"] += 1
                put(disp, f"EVAL REKAM Gesture {tg}: {ev['n']}/{args.fpg}", 55, (0, 0, 255), 0.8)
                if ev["n"] >= args.fpg:
                    ev["queue"].pop(0)
                    if ev["queue"]:
                        ev["phase"], ev["t0"] = "countdown", time.time()
                    else:
                        new = not os.path.exists(EVAL_CSV)
                        with open(EVAL_CSV, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            if new:
                                w.writerow(["timestamp", "lighting", "gesture_true", "gesture_pred", "correct"])
                            w.writerows(ev["rows"])
                        ev = None
                        print(f"Evaluasi selesai, data ditambahkan ke {EVAL_CSV}")
                        build_report()

        if bn is not None:
            bn["t"] += 1
            if bn["t"] <= WARMUP:
                put(disp, f"BENCH pemanasan {bn['t']}/{WARMUP} - pinch tangan di depan kamera", 55, (0, 255, 255), 0.8)
            else:
                bn["i"] = bn["t"] - WARMUP - 1
                put(disp, f"BENCH REKAM {bn['i'] + 1}/{BENCH_FRAMES}", 55, (0, 0, 255), 0.8)

        cv2.imshow("Gesture ROI (S=capture K=landmark L=cahaya E=eval B=bench R=laporan Q=keluar)", disp)
        key = cv2.waitKey(1) & 0xFF

        # total latency end-to-end (capture -> proses -> tampil)
        total_ms = (time.perf_counter() - t_start) * 1000
        if bn is not None and bn["t"] > WARMUP:
            bn["rows"].append([fmt(lm_ms, 4), "" if roi_ms is None else fmt(roi_ms, 4), filt_name,
                               "" if filt_ms is None else fmt(filt_ms, 4), fmt(total_ms, 4), int(lm is not None)])
            if len(bn["rows"]) >= BENCH_FRAMES:
                with open(BENCH_CSV, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["landmark_ms", "roi_ms", "filter", "filter_ms", "total_ms", "hand_detected"])
                    w.writerows(bn["rows"])
                det = sum(r[5] for r in bn["rows"])
                bn = None
                print(f"Benchmark selesai ({det}/{BENCH_FRAMES} frame ada tangan). Data: {BENCH_CSV}")
                build_report()

        now = time.perf_counter()
        inst = 1 / max(now - t_prev, 1e-6)
        fps_s = inst if fps_s == 0 else 0.9 * fps_s + 0.1 * inst
        t_prev = now

        # ---------------- tombol ----------------
        if key in (ord("q"), 27):
            break
        elif key == ord("k"):
            show_lm = not show_lm
        elif key == ord("h"):
            use_hsv = not use_hsv
            print("Gesture 4 ->", "HSV" if use_hsv else "Grayscale")
        elif key == ord("l"):
            light_idx = (light_idx + 1) % len(LIGHTS)
        elif key == ord("e") and ev is None and bn is None:
            ev = {"queue": [1, 2, 3, 4], "phase": "countdown", "t0": time.time(), "n": 0, "rows": []}
        elif key == ord("b") and ev is None and bn is None:
            bn = {"t": 0, "i": -1, "rows": []}
        elif key == ord("x"):
            ev, bn = None, None
            print("Dibatalkan.")
        elif key == ord("r"):
            build_report()
        elif key == ord("s"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "landmark" if show_lm else f"g{gesture}"
            path = os.path.join(CAP_DIR, f"{tag}_{stamp}.png")
            cv2.imwrite(path, disp)
            print("Tersimpan:", path)
            if gesture in GESTURES and not show_lm:
                last_cap[gesture] = disp.copy()
                if len(last_cap) == 4:
                    print("Komposit Gambar 3:", make_composite(last_cap))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
