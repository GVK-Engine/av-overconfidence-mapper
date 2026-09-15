# AV Overconfidence Death Zone Mapper
# Vamshikrishna Gadde | MS Robotics ASU
# Real footage demo, KITTI + YOLOv8, final

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from ultralytics import YOLO
from config import *

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\image_02\data"
)

FPS = 24
DURATION_S = 20
TOTAL_FRAMES = FPS * DURATION_S
HOLD = 6
SMOOTH_WINDOW = 21
FREEZE_SECONDS = 1

plt.rcParams.update({
    'figure.facecolor': CHART_BG,
    'axes.facecolor': CHART_BG,
    'text.color': CHART_TEXT,
    'axes.labelcolor': CHART_TEXT,
    'xtick.color': CHART_TEXT,
    'ytick.color': CHART_TEXT,
    'font.family': 'monospace'
})


def apply_exposure_glare(img, level):
    if level <= 0:
        return img.copy()
    out = img.astype(np.float32)
    out = out * (1.0 - 0.6 * level)
    h, w = img.shape[:2]
    cx, cy = int(w * 0.72), int(h * 0.30)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2
                   + (yy - cy) ** 2)
    radius = min(h, w) * (0.15 + 0.55 * level)
    glare = np.clip(
        1.0 - dist / (radius + 1e-6), 0, 1
    ) * 255.0 * level
    for c in range(3):
        out[:, :, c] = np.clip(
            out[:, :, c] + glare, 0, 255
        )
    k = int(1 + level * 8)
    if k % 2 == 0:
        k += 1
    out = cv2.GaussianBlur(
        out.astype(np.uint8), (k, k), 0
    )
    noise = np.random.normal(
        0, 6.0 * level, out.shape
    )
    return np.clip(
        out.astype(np.float32) + noise, 0, 255
    ).astype(np.uint8)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1)
          + (bx2 - bx1) * (by2 - by1)
          - inter + 1e-9)
    return inter / ua


def detect(model, img):
    results = model(img, verbose=False,
                    conf=0.25)
    out = []
    for b in results[0].boxes:
        out.append((
            b.xyxy[0].cpu().numpy(),
            float(b.conf[0].cpu().numpy())
        ))
    return out


def top3_conf(dets):
    if not dets:
        return 0.0
    confs = sorted(
        [c for _, c in dets], reverse=True
    )
    return float(np.mean(confs[:3]))


def annotate(img, current, baseline):
    vis = img.copy()
    matched = 0
    for base_box, _ in baseline:
        hit = None
        for cur_box, cur_conf in current:
            if iou(base_box, cur_box) > 0.4:
                hit = (cur_box, cur_conf)
                break
        x1, y1, x2, y2 = base_box.astype(int)
        if hit is not None:
            matched += 1
            cv2.rectangle(
                vis, (x1, y1), (x2, y2),
                (0, 255, 0), 3
            )
            label = f"DETECTED {hit[1]:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 2
            )
            cv2.rectangle(
                vis, (x1, y1 - th - 10),
                (x1 + tw + 6, y1),
                (0, 100, 0), -1
            )
            cv2.putText(
                vis, label,
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
        else:
            cv2.rectangle(
                vis, (x1, y1), (x2, y2),
                (0, 0, 255), 3
            )
            label = "MISSED"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 2
            )
            cv2.rectangle(
                vis, (x1, y1 - th - 10),
                (x1 + tw + 6, y1),
                (100, 0, 0), -1
            )
            cv2.putText(
                vis, label,
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )
    retention = matched / max(1, len(baseline))
    return vis, retention


def smooth(values, window):
    if len(values) < 2:
        return list(values)
    kernel = min(window, len(values))
    padded = np.concatenate([
        np.full(kernel - 1, values[0]),
        values
    ])
    return list(np.convolve(
        padded, np.ones(kernel) / kernel,
        mode='valid'
    ))


def draw_progress_bar(fig, progress):
    """Thin progress bar at bottom of frame."""
    bar_ax = fig.add_axes([0.05, 0.02, 0.9, 0.008])
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')
    bar_ax.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='#222222'
        )
    )
    bar_ax.add_patch(
        plt.Rectangle(
            (0, 0), progress, 1,
            facecolor='#00BFFF'
        )
    )


def main():
    frames = sorted(glob.glob(
        os.path.join(KITTI_DIR, "*.png")
    ))
    if not frames:
        print(f"No KITTI frames at {KITTI_DIR}")
        return

    print(f"KITTI frames: {len(frames)}")
    model = YOLO("yolov8n.pt")

    out_dir = os.path.join(RESULTS_DIR, "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, "real_overconfidence_demo.mp4"
    )
    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264",
        quality=8, macro_block_size=None
    )

    conf_hist, ret_hist, deg_hist = [], [], []
    vis_rgb = None
    retention = 1.0
    confidence = 1.0

    total_with_freeze = (
        TOTAL_FRAMES + FPS * FREEZE_SECONDS
    )
    print(f"Rendering {total_with_freeze} frames")

    for f in range(total_with_freeze):
        f_clamped = min(f, TOTAL_FRAMES - 1)
        level = f_clamped / (TOTAL_FRAMES - 1)

        if f_clamped % HOLD == 0 or vis_rgb is None:
            kitti_idx = (
                (f_clamped // HOLD) % len(frames)
            )
            clean = cv2.imread(
                frames[kitti_idx]
            )
            degraded = apply_exposure_glare(
                clean, level
            )
            baseline = detect(model, clean)
            current = detect(model, degraded)
            vis, retention = annotate(
                degraded, current, baseline
            )
            confidence = top3_conf(current)
            vis_rgb = cv2.cvtColor(
                vis, cv2.COLOR_BGR2RGB
            )

        conf_hist.append(confidence * 100)
        ret_hist.append(retention * 100)
        deg_hist.append(level * 100)

        conf_s = smooth(conf_hist, SMOOTH_WINDOW)
        ret_s = smooth(ret_hist, SMOOTH_WINDOW)

        gap = conf_s[-1] - ret_s[-1]
        in_death_zone = (
            conf_s[-1] > 70 and ret_s[-1] < 50
        )

        fig = plt.figure(
            figsize=(16, 9), dpi=100,
            facecolor=CHART_BG
        )
        gs = fig.add_gridspec(
            2, 3, height_ratios=[1.4, 1],
            hspace=0.28, wspace=0.25,
            left=0.05, right=0.97,
            top=0.90, bottom=0.10
        )
        ax_img = fig.add_subplot(gs[0, :2])
        ax_panel = fig.add_subplot(gs[0, 2])
        ax_curve = fig.add_subplot(gs[1, :])

        ax_img.imshow(vis_rgb)
        ax_img.axis("off")
        ax_img.set_title(
            f"Real KITTI footage | YOLOv8 live | "
            f"Exposure Loss + Glare {round(level*100)}% | "
            f"Green detected, Red missed",
            color=CHART_TEXT, fontsize=10
        )

        ax_panel.axis("off")
        ax_panel.set_xlim(0, 10)
        ax_panel.set_ylim(0, 10)
        ax_panel.text(
            5, 9.3, "LIVE MEASUREMENTS",
            ha="center", color="white",
            fontsize=13, fontweight="bold"
        )
        ax_panel.text(
            0.5, 8.0,
            f"Top detection confidence: "
            f"{round(conf_s[-1])}%",
            color="#00BFFF", fontsize=11,
            fontweight="bold"
        )
        ax_panel.text(
            0.5, 7.0,
            f"Objects still detected: "
            f"{round(ret_s[-1])}%",
            color=("#FF4444"
                   if ret_s[-1] < 50
                   else "#00FF00"),
            fontsize=11, fontweight="bold"
        )
        ax_panel.text(
            0.5, 6.0,
            f"Overconfidence gap: {round(gap)}%",
            color=("#FF4444" if gap > 25
                   else "#FFD700" if gap > 0
                   else "#00FF00"),
            fontsize=11, fontweight="bold"
        )
        if in_death_zone:
            ax_panel.text(
                5, 4.0, "DEATH ZONE",
                ha="center", color="#FF2222",
                fontsize=18, fontweight="bold"
            )
            ax_panel.text(
                5, 2.9,
                "Confident in survivors\n"
                "while losing the scene",
                ha="center", color="white",
                fontsize=9
            )

        ax_curve.plot(
            deg_hist, conf_s,
            color="#00BFFF", linewidth=3,
            linestyle="--",
            label="Top-3 detection confidence"
        )
        ax_curve.plot(
            deg_hist, ret_s,
            color="#FF4444", linewidth=3,
            label="Objects still detected"
        )
        ax_curve.axhline(
            y=70, color="#888888",
            linestyle=":", linewidth=1
        )
        ax_curve.axhline(
            y=50, color="#888888",
            linestyle=":", linewidth=1
        )
        ax_curve.set_xlim(0, 100)
        ax_curve.set_ylim(0, 105)
        ax_curve.set_xlabel(
            "Exposure Loss + Glare Severity (%)",
            fontsize=10
        )
        ax_curve.set_ylabel(
            "Score (%)", fontsize=10
        )
        ax_curve.legend(
            facecolor="#1a1a1a",
            labelcolor="white",
            fontsize=9, loc="lower left"
        )
        ax_curve.grid(
            True, alpha=0.25,
            linestyle="--", axis='y'
        )
        fig.suptitle(
            f"Real Overconfidence Measurement | "
            f"{TITLE_NAME}",
            color=CHART_TEXT, fontsize=13
        )

        draw_progress_bar(
            fig, f / total_with_freeze
        )

        fig.canvas.draw()
        buf = np.frombuffer(
            fig.canvas.buffer_rgba(),
            dtype=np.uint8
        )
        w, h = fig.canvas.get_width_height()
        writer.append_data(
            buf.reshape(h, w, 4)[:, :, :3]
        )
        plt.close(fig)

        if f % FPS == 0:
            print(f"  {f // FPS}s")

    writer.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()