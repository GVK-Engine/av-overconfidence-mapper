# AV Overconfidence Death Zone Mapper
# Vamshikrishna Gadde | MS Robotics ASU
# Synthetic explainer demo — clear captions,
# slow pacing, one idea per scene

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os
from config import *

plt.rcParams.update({
    'figure.facecolor': CHART_BG,
    'axes.facecolor': CHART_BG,
    'text.color': CHART_TEXT,
    'axes.labelcolor': CHART_TEXT,
    'xtick.color': CHART_TEXT,
    'ytick.color': CHART_TEXT,
    'font.family': 'monospace'
})

FPS = 24
SCENE_SECONDS = 8
NUM_SCENES = 3
TOTAL_FRAMES = FPS * SCENE_SECONDS * NUM_SCENES


def confidence_curve(levels):
    return np.where(
        levels < 45, 96,
        np.maximum(72, 96 - (levels - 45) * 0.8)
    )


def accuracy_curve(levels):
    return np.maximum(
        12, 96 * np.exp(
            -np.maximum(0, levels - 18) * 0.038
        )
    )


def draw_caption(fig, text):
    fig.text(
        0.5, 0.965, text,
        ha='center', va='top',
        color='#FFD700', fontsize=13,
        fontweight='bold'
    )


def draw_chart(ax, current_deg, in_death_zone):
    ax.set_facecolor('#0a0a0a')
    levels = np.linspace(0, 100, 200)
    conf = confidence_curve(levels)
    acc = accuracy_curve(levels)

    ax.plot(levels, conf, color='#00BFFF',
           linewidth=3, linestyle='--',
           label='Reported Confidence')
    ax.plot(levels, acc, color='#FF4444',
           linewidth=3, label='True Accuracy')

    death_mask = (conf > 80) & (acc < 50)
    ax.fill_between(
        levels, acc, conf, where=death_mask,
        alpha=0.25, color='red',
        label='Death Zone'
    )

    ax.axvline(x=current_deg, color='white',
              linewidth=2.5, linestyle=':')
    cur_conf = float(np.interp(current_deg, levels, conf))
    cur_acc = float(np.interp(current_deg, levels, acc))
    ax.plot(current_deg, cur_conf, 'o',
           color='#00BFFF', markersize=12, zorder=5)
    ax.plot(current_deg, cur_acc, 'o',
           color='#FF4444', markersize=12, zorder=5)

    ax.axhline(y=80, color='#888888',
              linestyle=':', linewidth=1)
    ax.axhline(y=50, color='#888888',
              linestyle=':', linewidth=1)
    ax.text(1, 82, 'Confidence threshold: 80%',
           color='#888888', fontsize=9)
    ax.text(1, 44, 'Accuracy threshold: 50%',
           color='#888888', fontsize=9)

    if in_death_zone:
        ax.text(current_deg + 2, 62, 'YOU ARE HERE',
               color='#FF4444', fontsize=10,
               fontweight='bold')

    ax.set_xlabel('Sensor Degradation (%)', fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(facecolor='#1a1a1a', labelcolor='white',
             fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.25, linestyle='--', axis='y')


def draw_readout(ax, degradation, confidence,
                 accuracy, in_death_zone):
    ax.set_facecolor('#0a0a0a')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.3, 'WHAT THE SYSTEM REPORTS',
           ha='center', color='white',
           fontsize=13, fontweight='bold')

    ax.text(0.5, 7.8,
           f'Sensor degradation: '
           f'{round(degradation)}%',
           color='white', fontsize=11)
    ax.text(0.5, 6.6,
           f'System says confidence: '
           f'{round(confidence)}%',
           color='#00BFFF', fontsize=12,
           fontweight='bold')
    ax.text(0.5, 5.4,
           f'What is actually true: '
           f'{round(accuracy)}% accurate',
           color='#FF4444', fontsize=12,
           fontweight='bold')

    gap = confidence - accuracy
    ax.text(0.5, 4.0,
           f'Gap between claim and reality: '
           f'{round(gap)} points',
           color='#FFD700', fontsize=11,
           fontweight='bold')

    if in_death_zone:
        box = patches.FancyBboxPatch(
            (0.3, 0.4), 9.4, 2.6,
            boxstyle="round,pad=0.2",
            facecolor='#330000',
            edgecolor='#FF0000', linewidth=2
        )
        ax.add_patch(box)
        ax.text(5, 2.2, 'DANGEROUS: confidently wrong',
               ha='center', color='#FF4444',
               fontsize=13, fontweight='bold')
        ax.text(5, 1.1,
               'The system does not know it has failed',
               ha='center', color='white', fontsize=9)
    else:
        box = patches.FancyBboxPatch(
            (0.3, 0.4), 9.4, 2.6,
            boxstyle="round,pad=0.2",
            facecolor='#002200',
            edgecolor='#00FF00', linewidth=2
        )
        ax.add_patch(box)
        ax.text(5, 1.7, 'System is calibrated',
               ha='center', color='#00FF00',
               fontsize=13, fontweight='bold')


def generate_frame(frame_num):
    fig = plt.figure(figsize=(16, 9), dpi=100,
                     facecolor=CHART_BG)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1],
        hspace=0.35, wspace=0.25,
        left=0.06, right=0.96,
        top=0.85, bottom=0.08
    )
    ax_chart = fig.add_subplot(gs[:, 0])
    ax_read = fig.add_subplot(gs[0, 1])

    frames_per_scene = FPS * SCENE_SECONDS
    scene = frame_num // frames_per_scene
    sp = (frame_num % frames_per_scene) / frames_per_scene

    if scene == 0:
        degradation = sp * 40
        caption = (
            "STEP 1: Sensors clean, confidence "
            "and accuracy agree"
        )
    elif scene == 1:
        degradation = 40 + sp * 15
        caption = (
            "STEP 2: Degradation rising, accuracy "
            "drops but confidence barely moves"
        )
    else:
        degradation = 55 + sp * 30
        caption = (
            "STEP 3: DEATH ZONE — system stays "
            "confident while it is actually failing"
        )

    confidence = float(np.interp(
        degradation,
        np.linspace(0, 100, 200),
        confidence_curve(np.linspace(0, 100, 200))
    ))
    accuracy = float(np.interp(
        degradation,
        np.linspace(0, 100, 200),
        accuracy_curve(np.linspace(0, 100, 200))
    ))
    in_death_zone = confidence > 80 and accuracy < 50

    draw_chart(ax_chart, degradation, in_death_zone)
    draw_readout(
        ax_read, degradation, confidence,
        accuracy, in_death_zone
    )

    draw_caption(fig, caption)
    fig.suptitle(
        f'AV Overconfidence Death Zone — How It '
        f'Happens | {TITLE_NAME}',
        color=CHART_TEXT, fontsize=13, y=0.995
    )

    fig.canvas.draw()
    buf = np.frombuffer(
        fig.canvas.buffer_rgba(), dtype=np.uint8
    )
    w, h = fig.canvas.get_width_height()
    frame_arr = buf.reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return frame_arr


def render_video(output_path, fps=FPS):
    os.makedirs(os.path.dirname(output_path),
               exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec='libx264',
        quality=8, macro_block_size=None
    )
    print(f"Rendering {TOTAL_FRAMES} frames")
    for f in range(TOTAL_FRAMES):
        writer.append_data(generate_frame(f))
        if f % fps == 0:
            print(f"  {f // fps}s")
    writer.close()
    print(f"Saved: {output_path}")