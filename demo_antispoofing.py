"""
╔══════════════════════════════════════════════════════════╗
║       DEMO FACE ANTI-SPOOFING — CDCN                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Cara Menjalankan:
  1. Install dependencies:
     pip install torch torchvision opencv-python numpy

  2. Letakkan file model di folder yang sama:
     - cdcn_final.pth

  3. Jalankan:
     python demo_antispoofing.py

Kontrol Keyboard:
  [1] Skenario 1 — Wajah Asli
  [2] Skenario 2 — Replay Attack (HP)
  [S] Screenshot
  [R] Reset statistik
  [Q] Keluar
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from datetime import datetime
from collections import deque

# ══════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THETA         = 0.7
OPTIMAL_THR   = 0.6397
MEAN          = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD           = np.array([0.229, 0.224, 0.225], dtype=np.float32)
WINDOW_NAME   = "Face Anti-Spoofing Demo — UGM Magister KA"
SMOOTHING_WIN = 10

# Warna BGR
C_GREEN  = (0, 210, 0)
C_RED    = (0, 0, 220)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_YELLOW = (0, 215, 255)
C_GRAY   = (180, 180, 180)
C_DARK   = (30, 30, 30)
C_TEAL   = (180, 200, 0)

# ══════════════════════════════════════════════════════════
# ARSITEKTUR CDCN
# ══════════════════════════════════════════════════════════
class Conv2d_cd(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1,
                 dilation=1, groups=1, bias=False, theta=0.7):
        super().__init__()
        self.conv  = nn.Conv2d(in_ch, out_ch, ks, stride, pad,
                               dilation, groups, bias)
        self.theta = theta

    def forward(self, x):
        out = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out
        kd  = self.conv.weight.sum(2).sum(2)[:, :, None, None]
        return out - self.theta * F.conv2d(
            x, kd, self.conv.bias,
            self.conv.stride, 0, groups=self.conv.groups)


class CDCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, theta=0.7):
        super().__init__()
        self.c1   = Conv2d_cd(in_ch, out_ch, theta=theta)
        self.b1   = nn.BatchNorm2d(out_ch)
        self.c2   = Conv2d_cd(out_ch, out_ch, theta=theta)
        self.b2   = nn.BatchNorm2d(out_ch)
        self.skip = (nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch))
                     if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.skip(x)
        x = self.relu(self.b1(self.c1(x)))
        return self.relu(self.b2(self.c2(x)) + r)


class CDCN(nn.Module):
    def __init__(self, theta=0.7):
        super().__init__()
        self.stem = nn.Sequential(
            Conv2d_cd(3, 64, theta=theta), nn.BatchNorm2d(64), nn.ReLU(True),
            Conv2d_cd(64, 64, theta=theta), nn.BatchNorm2d(64), nn.ReLU(True))
        self.s1 = nn.Sequential(nn.MaxPool2d(2), CDCBlock(64, 128, theta))
        self.s2 = nn.Sequential(nn.MaxPool2d(2), CDCBlock(128, 256, theta))
        self.s3 = nn.Sequential(nn.MaxPool2d(2), CDCBlock(256, 512, theta))
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 1, 1), nn.Sigmoid())
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x):
        f = self.s3(self.s2(self.s1(self.stem(x))))
        return self.cls_head(f), self.depth_head(f)


# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
def load_model(device):
    """Cari dan load file model — support cdcn_final.pth dan cdcn_best.pth."""
    candidates = ["cdcn_final.pth", "cdcn_best.pth"]
    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break

    if found is None:
        return None, OPTIMAL_THR

    try:
        model = CDCN(theta=THETA)
        ckpt  = torch.load(found, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        acer = ckpt.get("acer", 0.0)
        thr  = ckpt.get("optimal_thr", OPTIMAL_THR)
        print(f"  ✅ {found} loaded")
        print(f"     ACER      : {acer:.2f}%")
        print(f"     Threshold : {thr:.4f}")
        return model.to(device), thr
    except Exception as e:
        print(f"  ❌ Gagal load {found}: {e}")
        return None, OPTIMAL_THR


# ══════════════════════════════════════════════════════════
# PREPROCESSING & PREDIKSI
# ══════════════════════════════════════════════════════════
def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    t   = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return t


@torch.no_grad()
def predict(model, frame_bgr, device, threshold):
    t             = preprocess(frame_bgr).to(device)
    logit, depth  = model(t)
    prob          = torch.sigmoid(logit.squeeze()).item()
    label         = "REAL" if prob >= threshold else "FAKE"
    depth_map     = depth.squeeze().cpu().numpy()
    return label, prob, depth_map


# ══════════════════════════════════════════════════════════
# DRAWING UTILITIES
# ══════════════════════════════════════════════════════════
def filled_rect(img, pt1, pt2, color, alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_text_bg(img, text, pos, scale=0.55, color=C_WHITE,
                 bg=C_DARK, thick=1, pad=6, alpha=0.75):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    filled_rect(img,
                (x - pad, y - th - pad),
                (x + tw + pad, y + bl + pad),
                bg, alpha)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_prob_bar(img, prob, x, y, width, height, thr, label=""):
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)

    # Fill
    fill_w     = int(prob * width)
    bar_color  = C_GREEN if prob >= thr else C_RED
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), bar_color, -1)

    # Threshold line
    thr_x = int(thr * width) + x
    cv2.line(img, (thr_x, y - 3), (thr_x, y + height + 3), C_YELLOW, 2)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"{prob:.1%}",
                (x + 6, y + height - 5),
                font, 0.45, C_WHITE, 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label,
                    (x + width - 80, y + height - 5),
                    font, 0.4, C_GRAY, 1, cv2.LINE_AA)


def draw_depth_map(img, depth_map, x, y, w=130, h=95):
    dm       = cv2.resize(depth_map, (w, h))
    dm_color = cv2.applyColorMap((dm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), C_WHITE, 1)
    img[y:y + h, x:x + w] = dm_color
    draw_text_bg(img, "Depth Map", (x, y - 8),
                 scale=0.38, pad=3, alpha=0.7)

    # Interpretasi
    avg = float(depth_map.mean())
    interp = "3D Wajah Asli" if avg > 0.5 else "Permukaan Datar"
    color  = C_GREEN if avg > 0.5 else C_RED
    draw_text_bg(img, interp, (x, y + h + 14),
                 scale=0.38, color=color, pad=3, alpha=0.7)


def draw_keyboard_guide(img, h, w):
    filled_rect(img, (0, h - 38), (w, h), C_BLACK, alpha=0.85)
    keys = [
        ("[1] Wajah Asli", C_GREEN),
        ("[2] Replay Attack", C_RED),
        ("[S] Screenshot", C_YELLOW),
        ("[R] Reset", C_GRAY),
        ("[Q] Keluar", C_WHITE),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    gap  = w // len(keys)
    for i, (text, color) in enumerate(keys):
        cv2.putText(img, text, (i * gap + 6, h - 12),
                    font, 0.4, color, 1, cv2.LINE_AA)


def draw_stats(img, stats, x, y):
    n  = max(stats["total"], 1)
    nl = stats["live"]
    ns = stats["spoof"]
    filled_rect(img, (x, y), (x + 210, y + 115), (15, 15, 15), alpha=0.8)
    cv2.rectangle(img, (x, y), (x + 210, y + 115), (60, 60, 60), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "STATISTIK SESI",
                (x + 8, y + 18), font, 0.45, C_YELLOW, 1, cv2.LINE_AA)
    cv2.line(img, (x + 5, y + 24), (x + 205, y + 24), (60, 60, 60), 1)
    rows = [
        (f"Frame   : {stats['total']}",          C_WHITE),
        (f"LIVE    : {nl} ({nl/n*100:.0f}%)",     C_GREEN),
        (f"SPOOF   : {ns} ({ns/n*100:.0f}%)",     C_RED),
        (f"Avg Prob: {stats['avg_prob']:.3f}",    C_WHITE),
        (f"FPS     : {stats['fps']:.1f}",         C_TEAL),
    ]
    for i, (text, color) in enumerate(rows):
        cv2.putText(img, text, (x + 8, y + 42 + i * 17),
                    font, 0.42, color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
# RENDER FRAME
# ══════════════════════════════════════════════════════════
def render_frame(frame, model, thr, stats, smooth_buf, scenario):
    h, w  = frame.shape[:2]
    out   = frame.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # ── Prediksi ─────────────────────────────────────────
    label, prob, depth_map = predict(model, frame, DEVICE, thr)
    smooth_buf.append(prob)
    prob_s  = float(np.mean(smooth_buf))
    label_s = "REAL" if prob_s >= thr else "FAKE"
    is_live = label_s == "REAL"

    # ── Update statistik ─────────────────────────────────
    stats["total"] += 1
    stats["avg_prob"] = (stats["avg_prob"] * (stats["total"] - 1) + prob) / stats["total"]
    if is_live:
        stats["live"] += 1
    else:
        stats["spoof"] += 1

    # ── Warna utama ───────────────────────────────────────
    main_color = C_GREEN if is_live else C_RED

    # ── Border frame ─────────────────────────────────────
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), main_color, 6)

    # ── Banner atas ───────────────────────────────────────
    filled_rect(out, (0, 0), (w, 68), C_BLACK, alpha=0.72)
    cv2.rectangle(out, (0, 0), (7, 68), main_color, -1)

    icon  = "LIVE" if is_live else "SPOOF"
    check = "v" if is_live else "x"
    cv2.putText(out, f"{check}  {icon}",
                (20, 46), font, 1.4, main_color, 3, cv2.LINE_AA)
    cv2.putText(out, f"{prob_s:.1%}",
                (190, 46), font, 1.2, C_WHITE, 2, cv2.LINE_AA)

    scen_text = "Skenario 1: Wajah Asli" if scenario == 1 else "Skenario 2: Replay Attack"
    cv2.putText(out, scen_text,
                (w - 260, 22), font, 0.45, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(out, f"CDCN  thr={thr:.4f}",
                (w - 210, 52), font, 0.4, C_TEAL, 1, cv2.LINE_AA)
    cv2.line(out, (0, 68), (w, 68), main_color, 2)

    # ── Warning SPOOF (Skenario 2) ────────────────────────
    if scenario == 2 and not is_live:
        overlay = out.copy()
        bx1, bx2 = w // 4, 3 * w // 4
        by1, by2 = h // 2 - 55, h // 2 + 55
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 130), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), C_RED, 2)
        cv2.putText(out, "! REPLAY ATTACK",
                    (bx1 + 18, h // 2 - 8),
                    font, 0.85, C_WHITE, 2, cv2.LINE_AA)
        cv2.putText(out, "AKSES DITOLAK",
                    (bx1 + 35, h // 2 + 30),
                    font, 0.75, C_YELLOW, 2, cv2.LINE_AA)

    # ── Probability bar ───────────────────────────────────
    bar_y = h - 95
    draw_text_bg(out, f"threshold = {thr:.4f}",
                 (w - 195, bar_y - 6),
                 scale=0.38, color=C_YELLOW, pad=3, alpha=0.7)
    draw_prob_bar(out, prob_s,
                  x=10, y=bar_y,
                  width=w - 20, height=28,
                  thr=thr, label="Probabilitas REAL")

    # ── Depth map ─────────────────────────────────────────
    draw_depth_map(out, depth_map, x=w - 148, y=h - 220)

    # ── Statistik ─────────────────────────────────────────
    draw_stats(out, stats, x=10, y=80)

    # ── Keyboard guide ────────────────────────────────────
    draw_keyboard_guide(out, h, w)

    return out


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  DEMO FACE ANTI-SPOOFING - CDCN")
    print("=" * 55)
    print(f"  Device  : {DEVICE}")
    print(f"  Python  : {sys.version.split()[0]}")
    print()

    # ── Load model ─────────────────────────────────────────
    print("📦 Loading model CDCN...")
    model, thr = load_model(DEVICE)

    if model is None:
        print()
        print("❌ File model tidak ditemukan!")
        print("   Letakkan salah satu file berikut di folder yang sama:")
        print("   - cdcn_final.pth")
        print("   - cdcn_best.pth")
        print()
        print("   Cara mendapatkan:")
        print("   1. Jalankan notebook di Google Colab")
        print("   2. Jalankan Bagian 11 (Simpan & Download)")
        print("   3. Ekstrak lcc_results.zip")
        print("   4. Copy cdcn_final.pth ke folder ini")
        input("\n   Tekan Enter untuk keluar...")
        return

    # ── Buka webcam ────────────────────────────────────────
    print()
    print("Membuka webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam tidak ditemukan!")
        print("   Pastikan webcam tersambung dan tidak dipakai aplikasi lain.")
        input("   Tekan Enter untuk keluar...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Webcam terbuka: {aw}×{ah}")
    print()
    print("  Kontrol keyboard:")
    print("  [1] Skenario 1 — Wajah Asli")
    print("  [2] Skenario 2 — Replay Attack (arahkan layar HP)")
    print("  [S] Screenshot")
    print("  [R] Reset statistik")
    print("  [Q] Keluar")
    print()

    # ── State ──────────────────────────────────────────────
    scenario  = 1
    stats     = {"total": 0, "live": 0, "spoof": 0,
                 "avg_prob": 0.0, "fps": 0.0}
    smooth    = deque(maxlen=SMOOTHING_WIN)
    fps_buf   = deque(maxlen=30)
    ss_dir    = "screenshots"
    os.makedirs(ss_dir, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, aw, ah)

    print(f"✅ Demo berjalan — Skenario {scenario} aktif\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Gagal baca frame"); break

        frame = cv2.flip(frame, 1)

        # FPS
        fps_buf.append(time.time())
        if len(fps_buf) >= 2:
            stats["fps"] = len(fps_buf) / (fps_buf[-1] - fps_buf[0] + 1e-6)

        # Render
        try:
            display = render_frame(frame, model, thr, stats, smooth, scenario)
        except Exception as e:
            display = frame.copy()
            cv2.putText(display, f"Error: {str(e)[:60]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

        cv2.imshow(WINDOW_NAME, display)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), ord('Q'), 27]:
            print("\n Demo dihentikan.")
            break

        elif key == ord('1'):
            scenario = 1
            smooth.clear()
            stats = {"total": 0, "live": 0, "spoof": 0,
                     "avg_prob": 0.0, "fps": stats["fps"]}
            print("🟢 Skenario 1: Wajah Asli — arahkan wajah ke kamera")

        elif key == ord('2'):
            scenario = 2
            smooth.clear()
            stats = {"total": 0, "live": 0, "spoof": 0,
                     "avg_prob": 0.0, "fps": stats["fps"]}
            print("🔴 Skenario 2: Replay Attack — arahkan layar HP ke kamera")

        elif key in [ord('s'), ord('S')]:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{ss_dir}/demo_s{scenario}_{ts}.jpg"
            cv2.imwrite(path, display)
            print(f"📸 Screenshot: {path}")

        elif key in [ord('r'), ord('R')]:
            smooth.clear()
            stats = {"total": 0, "live": 0, "spoof": 0,
                     "avg_prob": 0.0, "fps": stats["fps"]}
            print("Statistik direset")

    # ── Cleanup ────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 45)
    print("  RINGKASAN SESI")
    print("=" * 45)
    n = max(stats["total"], 1)
    print(f"  Total frame : {stats['total']}")
    print(f"  LIVE        : {stats['live']} ({stats['live']/n*100:.0f}%)")
    print(f"  SPOOF       : {stats['spoof']} ({stats['spoof']/n*100:.0f}%)")
    print(f"  Avg prob    : {stats['avg_prob']:.3f}")
    print("=" * 45)


if __name__ == "__main__":
    main()
