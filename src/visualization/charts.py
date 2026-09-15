# AV Overconfidence Death Zone Mapper
# Vamshikrishna Gadde | MS Robotics ASU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os
from config import *

plt.rcParams.update({
    'figure.facecolor': CHART_BG,
    'axes.facecolor': CHART_BG,
    'text.color': CHART_TEXT,
    'axes.labelcolor': CHART_TEXT,
    'xtick.color': CHART_TEXT,
    'ytick.color': CHART_TEXT,
    'axes.edgecolor': '#333333',
    'grid.color': '#222222',
    'font.family': 'monospace'
})

COLORS = {
    'school_zone_occlusion':       '#FF4444',
    'construction_zone_confusion': '#FF8800',
    'exposure_loss_glare':         '#FFD700',
    'rain_plus_crowd':             '#00BFFF',
    'blind_curve_occlusion':       '#FF69B4'
}


def plot_confidence_accuracy_divergence(
        results, save=True):
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8),
        facecolor=CHART_BG
    )

    ax = axes[0]
    ax.set_facecolor(CHART_BG)

    levels_pct = [l * 100
                  for l in DEGRADATION_LEVELS]

    for condition, data in results.items():
        color = COLORS[condition]
        label = CONDITION_LABELS[condition]
        accs = [a * 100
                for a in data['accuracies']]
        confs = [c * 100
                 for c in data['confidences']]

        ax.plot(levels_pct, confs,
               color=color,
               linewidth=3,
               linestyle='--',
               alpha=0.9)
        ax.plot(levels_pct, accs,
               color=color,
               linewidth=3,
               label=label,
               alpha=0.9)

    ax.axhspan(0, ACCURACY_THRESHOLD * 100,
              alpha=0.08, color='red',
              zorder=0)
    ax.axhline(y=CONFIDENCE_THRESHOLD * 100,
              color='red', linestyle=':',
              linewidth=1.5, alpha=0.7)
    ax.axhline(y=ACCURACY_THRESHOLD * 100,
              color='red', linestyle=':',
              linewidth=1.5, alpha=0.7)

    ax.text(95, CONFIDENCE_THRESHOLD * 100 + 1,
           'Confidence Threshold 80%',
           color='white', fontsize=8,
           ha='right')
    ax.text(95, ACCURACY_THRESHOLD * 100 + 1,
           'Accuracy Threshold 50%',
           color='white', fontsize=8,
           ha='right')

    ax.text(75, 25,
           'OVERCONFIDENCE\nDEATH ZONE',
           color='red', fontsize=12,
           fontweight='bold',
           ha='center', va='center',
           alpha=0.6)

    ax.set_xlabel(
        'Degradation Severity (%)',
        fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        'Score (%)',
        fontsize=12, labelpad=10
    )
    ax.set_title(
        'Confidence vs Accuracy Divergence\n'
        'Solid = Accuracy | Dashed = Confidence',
        fontsize=11, color=CHART_TEXT, pad=15
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25,
           linestyle='--', axis='y')
    ax.legend(
        facecolor='#1a1a1a',
        labelcolor=CHART_TEXT,
        fontsize=8,
        loc='lower left'
    )
    ax.text(0.98, 0.02, 'N=200 trials per level',
           transform=ax.transAxes,
           color='#666666', fontsize=8,
           ha='right', style='italic')

    ax2 = axes[1]
    ax2.set_facecolor(CHART_BG)

    conditions = list(results.keys())
    boundaries = [
        results[c]['death_zone_boundary'] * 100
        if results[c]['death_zone_boundary']
        else 100
        for c in conditions
    ]
    labels = [CONDITION_LABELS[c]
              for c in conditions]
    colors = [COLORS[c] for c in conditions]

    sorted_data = sorted(
        zip(boundaries, labels, colors),
        key=lambda x: x[0]
    )
    boundaries_s = [d[0] for d in sorted_data]
    labels_s = [d[1] for d in sorted_data]
    colors_s = [d[2] for d in sorted_data]

    bars = ax2.barh(
        labels_s, boundaries_s,
        color=colors_s,
        alpha=0.85,
        edgecolor='#333333',
        height=0.5
    )

    for bar, val in zip(bars, boundaries_s):
        ax2.text(
            val + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{round(val)}%',
            va='center',
            color=CHART_TEXT,
            fontsize=11,
            fontweight='bold'
        )

    ax2.axvline(x=50, color='red',
               linestyle='--',
               linewidth=1.5, alpha=0.7,
               label='Early warning zone')

    ax2.set_xlabel(
        'Degradation Level at Death Zone Entry (%)',
        fontsize=11, labelpad=10
    )
    ax2.set_title(
        'Death Zone Entry Boundary Per Condition\n'
        'Lower = More Dangerous',
        fontsize=11, color=CHART_TEXT, pad=15
    )
    ax2.grid(True, alpha=0.25, axis='x')
    ax2.set_xlim(0, 110)

    fig.suptitle(
        f'AV Overconfidence Death Zone | {TITLE_NAME}',
        fontsize=14, color=CHART_TEXT, y=1.02
    )

    plt.tight_layout()
    if save:
        path = os.path.join(
            RESULTS_DIR, 'charts',
            'hero_confidence_accuracy.png'
        )
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )
        plt.savefig(
            path, dpi=CHART_DPI,
            bbox_inches='tight',
            facecolor=CHART_BG
        )
        print(f"Saved: {path}")
    plt.close()
    return fig


def plot_death_zone_heatmap(results, save=True):
    fig, ax = plt.subplots(
        figsize=(14, 6),
        facecolor=CHART_BG
    )
    ax.set_facecolor(CHART_BG)

    conditions = list(results.keys())
    levels_pct = [l * 100
                  for l in DEGRADATION_LEVELS]

    matrix = np.array([
        results[c]['death_zone_rates']
        for c in conditions
    ]) * 100

    im = ax.imshow(
        matrix,
        aspect='auto',
        cmap='RdYlGn_r',
        vmin=0, vmax=100,
        interpolation='bilinear'
    )

    ax.set_xticks(range(len(levels_pct)))
    ax.set_xticklabels(
        [f'{round(l)}%' for l in levels_pct],
        color=CHART_TEXT, fontsize=9
    )
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(
        [CONDITION_LABELS[c]
         for c in conditions],
        color=CHART_TEXT, fontsize=10
    )

    for i in range(len(conditions)):
        for j in range(len(levels_pct)):
            val = matrix[i, j]
            text_color = ('black'
                         if val > 50
                         else 'white')
            ax.text(
                j, i, f'{round(val)}%',
                ha='center', va='center',
                color=text_color,
                fontsize=8,
                fontweight='bold'
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        'Death Zone Rate (%)',
        color=CHART_TEXT
    )
    cbar.ax.yaxis.set_tick_params(
        color=CHART_TEXT
    )
    plt.setp(
        cbar.ax.yaxis.get_ticklabels(),
        color=CHART_TEXT
    )

    ax.set_xlabel(
        'Degradation Severity (%)',
        color=CHART_TEXT, fontsize=12
    )
    ax.set_title(
        f'Overconfidence Death Zone Rate | '
        f'{TITLE_NAME}',
        fontsize=13, color=CHART_TEXT, pad=15
    )

    plt.tight_layout()
    if save:
        path = os.path.join(
            RESULTS_DIR, 'charts',
            'death_zone_heatmap.png'
        )
        plt.savefig(
            path, dpi=CHART_DPI,
            bbox_inches='tight',
            facecolor=CHART_BG
        )
        print(f"Saved: {path}")
    plt.close()
    return fig


def plot_incident_correlation(
        results, save=True):
    fig, ax = plt.subplots(
        figsize=(14, 7),
        facecolor=CHART_BG
    )
    ax.set_facecolor(CHART_BG)

    conditions = list(results.keys())
    levels_pct = [l * 100
                  for l in DEGRADATION_LEVELS]

    for condition in conditions:
        data = results[condition]
        color = COLORS[condition]
        label = CONDITION_LABELS[condition]
        gaps = [g * 100 for g in data['gaps']]

        ax.plot(
            levels_pct, gaps,
            color=color,
            linewidth=3,
            label=label,
            alpha=0.9
        )

        boundary = data['death_zone_boundary']
        if boundary:
            ax.axvline(
                x=boundary * 100,
                color=color,
                linestyle='--',
                alpha=0.4,
                linewidth=1
            )

    incident_y_positions = [15, 25, 35, 45]
    for idx, (incident, (condition, deg)) in \
            enumerate(
                INCIDENT_DEGRADATION_ESTIMATES
                .items()
            ):
        color = COLORS[condition]
        y_pos = incident_y_positions[
            idx % len(incident_y_positions)
        ]
        ax.annotate(
            f'★ {incident}',
            xy=(deg * 100, y_pos),
            xytext=(deg * 100 + 3, y_pos + 5),
            color=color,
            fontsize=9,
            fontweight='bold',
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=1.5
            )
        )
        ax.scatter(
            deg * 100, y_pos,
            color=color,
            s=200, zorder=5,
            marker='*'
        )

    ax.axhline(
        y=20, color='red',
        linestyle=':', alpha=0.7,
        linewidth=1.5,
        label='Dangerous gap threshold (20%)'
    )
    ax.fill_between(
        levels_pct,
        20, 100,
        alpha=0.05, color='red'
    )

    ax.text(
        75, 55,
        'DANGER ZONE\nConfidence >> Accuracy',
        color='red', fontsize=11,
        fontweight='bold',
        ha='center', alpha=0.7
    )

    ax.set_xlabel(
        'Degradation Severity (%)',
        fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        'Overconfidence Gap (%)\n'
        '(Confidence minus Accuracy)',
        fontsize=12, labelpad=10
    )
    ax.set_title(
        f'Real 2026 Incidents vs Measured '
        f'Overconfidence Boundaries | {TITLE_NAME}',
        fontsize=13, color=CHART_TEXT, pad=15
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.25,
           linestyle='--', axis='y')
    ax.legend(
        facecolor='#1a1a1a',
        labelcolor=CHART_TEXT,
        fontsize=9,
        loc='upper left'
    )

    plt.tight_layout()
    if save:
        path = os.path.join(
            RESULTS_DIR, 'charts',
            'incident_correlation.png'
        )
        plt.savefig(
            path, dpi=CHART_DPI,
            bbox_inches='tight',
            facecolor=CHART_BG
        )
        print(f"Saved: {path}")
    plt.close()
    return fig


def plot_safety_envelope(results, save=True):
    fig, ax = plt.subplots(
        figsize=(13, 7),
        facecolor=CHART_BG
    )
    ax.set_facecolor(CHART_BG)

    conditions = list(results.keys())
    safe_limits = []
    labels = []
    colors_list = []

    for condition in conditions:
        data = results[condition]
        boundary = data['death_zone_boundary']
        safe_limit = (boundary * 100
                     if boundary else 100)
        safe_limits.append(safe_limit)
        labels.append(
            CONDITION_LABELS[condition]
        )
        colors_list.append(COLORS[condition])

    sorted_data = sorted(
        zip(safe_limits, labels,
            colors_list, conditions),
        key=lambda x: x[0]
    )
    safe_limits_s = [d[0] for d in sorted_data]
    labels_s = [d[1] for d in sorted_data]
    colors_s = [d[2] for d in sorted_data]
    conditions_s = [d[3] for d in sorted_data]

    bars = ax.bar(
        labels_s, safe_limits_s,
        color=colors_s,
        alpha=0.85,
        edgecolor='#333333',
        linewidth=1.5,
        width=0.6
    )

    for bar, val, condition in zip(
        bars, safe_limits_s, conditions_s
    ):
        incident = CONDITION_INCIDENTS.get(
            condition, ''
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{round(val)}%',
            ha='center', va='bottom',
            color=CHART_TEXT,
            fontsize=13,
            fontweight='bold'
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            2,
            incident,
            ha='center', va='bottom',
            color='white',
            fontsize=8,
            fontweight='bold',
            style='italic'
        )

    ax.axhline(
        y=np.mean(safe_limits_s),
        color='white',
        linestyle='--',
        alpha=0.4,
        linewidth=1.5,
        label=f'Mean safe limit: '
              f'{round(np.mean(safe_limits_s))}%'
    )

    most_dangerous = labels_s[0]
    ax.annotate(
        f'Most Dangerous:\n{most_dangerous}',
        xy=(0, safe_limits_s[0]),
        xytext=(1, safe_limits_s[0] + 8),
        color=colors_s[0],
        fontsize=10,
        fontweight='bold',
        arrowprops=dict(
            arrowstyle='->',
            color=colors_s[0],
            lw=2
        )
    )

    ax.set_ylabel(
        'Maximum Safe Degradation Level (%)',
        fontsize=12, labelpad=10
    )
    ax.set_ylim(0, 110)
    ax.set_title(
        f'AV Safety Envelope | Max Safe '
        f'Operating Degradation | {TITLE_NAME}',
        fontsize=13, color=CHART_TEXT, pad=15
    )
    ax.tick_params(
        colors=CHART_TEXT, labelsize=10
    )
    ax.grid(
        True, alpha=0.2,
        axis='y', linestyle='--'
    )
    ax.legend(
        facecolor='#1a1a1a',
        labelcolor=CHART_TEXT,
        fontsize=10
    )

    insight = (
        f'{most_dangerous} is the most '
        f'dangerous condition\n'
        f'entering the death zone at only '
        f'{round(safe_limits_s[0])}% degradation'
    )
    ax.text(
        0.98, 0.05, insight,
        transform=ax.transAxes,
        color='#FF8800',
        fontsize=10,
        ha='right', va='bottom',
        bbox=dict(
            boxstyle='round',
            facecolor='#1a1a1a',
            alpha=0.8
        )
    )

    plt.tight_layout()
    if save:
        path = os.path.join(
            RESULTS_DIR, 'charts',
            'safety_envelope.png'
        )
        plt.savefig(
            path, dpi=CHART_DPI,
            bbox_inches='tight',
            facecolor=CHART_BG
        )
        print(f"Saved: {path}")
    plt.close()
    return fig