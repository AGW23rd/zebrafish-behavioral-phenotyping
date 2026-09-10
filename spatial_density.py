"""
Spatial Density Maps — Publication-Ready Figures
================================================
Figure A: Spatial density map of ALL larval trajectories (log density scale)
Figure B: Spatial density of STOPPING coordinates (near-zero velocity),
          showing central-vs-peripheral distribution for Thigmotaxis Index
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. PUBLICATION FORMATTING
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
print("Loading tracking data (~60 MB)...")
df = pd.read_excel('runs/locomotion_raw_data.xlsx')
print(f"  Loaded {len(df):,} tracking points")

# Arena dimensions (from benchmark_tracker / analyze_wells)
WIDTH  = 2048
HEIGHT = 1536
COLS   = 3
ROWS   = 2
w_step = WIDTH / COLS    # ~682.67 per well
h_step = HEIGHT / ROWS   # 768 per well

# ──────────────────────────────────────────────────────────────────────────────
# 3. HELPER — 2D histogram with Gaussian smoothing
# ──────────────────────────────────────────────────────────────────────────────
def make_density(cx, cy, bins=300, sigma=3.0):
    """Return smoothed 2D histogram (density grid) + extent."""
    H, xedges, yedges = np.histogram2d(
        cx, cy, bins=bins,
        range=[[0, WIDTH], [0, HEIGHT]]
    )
    # Gaussian smooth for visual clarity
    H = gaussian_filter(H.T, sigma=sigma)  # transpose so y-axis is rows
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]  # origin='upper'
    return H, extent

def draw_well_grid(ax, draw_inner_zones=False):
    """Overlay the 2×3 well boundaries."""
    for r in range(ROWS):
        for c in range(COLS):
            x0 = c * w_step
            y0 = r * h_step
            # Well boundary
            rect = Rectangle((x0, y0), w_step, h_step,
                              linewidth=1.8, edgecolor='white',
                              facecolor='none', linestyle='-', alpha=0.85)
            ax.add_patch(rect)
            # Well label
            well_id = r * COLS + c + 1
            ax.text(x0 + w_step/2, y0 + 22, f'Well {well_id}',
                    ha='center', va='top', color='white',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

            if draw_inner_zones:
                # Inner zone boundary (sqrt(0.5) ≈ 70.7% scaling)
                s = np.sqrt(0.5)
                margin_x = (1 - s) / 2 * w_step
                margin_y = (1 - s) / 2 * h_step
                inner = Rectangle((x0 + margin_x, y0 + margin_y),
                                  w_step * s, h_step * s,
                                  linewidth=1.5, edgecolor='cyan',
                                  facecolor='none', linestyle='--', alpha=0.9)
                ax.add_patch(inner)

# ──────────────────────────────────────────────────────────────────────────────
# 4. FIGURE 1 — All-Trajectory Spatial Density (Log Scale)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Figure 1: All-trajectory density map...")

fig1, ax1 = plt.subplots(figsize=(9, 6.5))

H_all, extent_all = make_density(df['cx'].values, df['cy'].values,
                                  bins=350, sigma=4.0)

# Replace zeros for log scale
H_all[H_all == 0] = np.nan

im1 = ax1.imshow(H_all, extent=extent_all, origin='upper',
                  norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(H_all)),
                  cmap='inferno', aspect='equal', interpolation='bilinear')

draw_well_grid(ax1, draw_inner_zones=False)

cbar1 = fig1.colorbar(im1, ax=ax1, shrink=0.82, pad=0.02, label='Point Density (log scale)')
cbar1.ax.tick_params(labelsize=8)

ax1.set_title('Spatial Density of Larval Trajectories Across Wells\n(Log Density Scale)',
              fontweight='bold', pad=12)
ax1.set_xlabel('X Position (pixels)', fontweight='bold')
ax1.set_ylabel('Y Position (pixels)', fontweight='bold')
ax1.set_xlim(0, WIDTH)
ax1.set_ylim(HEIGHT, 0)

# Stats annotation
n_total = len(df)
n_objects = df['object_id'].nunique()
ax1.text(0.01, 0.01,
         f'N = {n_total:,} points  |  {n_objects} tracked objects',
         transform=ax1.transAxes, fontsize=8, color='white',
         va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

fig1.tight_layout()
fig1.savefig('Figure_SpatialDensity_AllTrajectories.png')
print("  Saved: Figure_SpatialDensity_AllTrajectories.png")
plt.close(fig1)

# ──────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 2 — Stopping-Point Density with Thigmotaxis Zones
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Figure 2: Stopping-point density (thigmotaxis)...")

# Filter to near-zero-velocity points (is_stopped flag or speed < 0.5 px/frame)
stopped = df[df['is_stopped'] == True].copy()
print(f"  Stopped points: {len(stopped):,}")

fig2, ax2 = plt.subplots(figsize=(9, 6.5))

H_stop, extent_stop = make_density(stopped['cx'].values, stopped['cy'].values,
                                    bins=350, sigma=4.5)
H_stop[H_stop == 0] = np.nan

im2 = ax2.imshow(H_stop, extent=extent_stop, origin='upper',
                  norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(H_stop)),
                  cmap='YlOrRd', aspect='equal', interpolation='bilinear')

draw_well_grid(ax2, draw_inner_zones=True)

cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.82, pad=0.02, label='Stop Density (log scale)')
cbar2.ax.tick_params(labelsize=8)

ax2.set_title('Spatial Density of Stopping Coordinates (Near-Zero Velocity)\n'
              'Central vs. Peripheral Distribution  —  Thigmotaxis Index Basis',
              fontweight='bold', pad=12)
ax2.set_xlabel('X Position (pixels)', fontweight='bold')
ax2.set_ylabel('Y Position (pixels)', fontweight='bold')
ax2.set_xlim(0, WIDTH)
ax2.set_ylim(HEIGHT, 0)

# Legend for zone boundaries
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='white', linewidth=1.8, linestyle='-',  label='Well boundary'),
    Line2D([0], [0], color='cyan',  linewidth=1.5, linestyle='--', label='Inner zone (50% area)'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=8,
           framealpha=0.7, facecolor='black', labelcolor='white',
           edgecolor='gray')

# Stats annotation
n_stopped = len(stopped)
pct_outer_total = 0
count = 0
s = np.sqrt(0.5)
for r in range(ROWS):
    for c in range(COLS):
        x0 = c * w_step
        y0 = r * h_step
        margin_x = (1 - s) / 2 * w_step
        margin_y = (1 - s) / 2 * h_step
        # Find stops in this well
        in_well = stopped[(stopped['cx'] >= x0) & (stopped['cx'] < x0 + w_step) &
                          (stopped['cy'] >= y0) & (stopped['cy'] < y0 + h_step)]
        if len(in_well) > 0:
            in_outer = (
                (in_well['cx'] < x0 + margin_x) |
                (in_well['cx'] > x0 + w_step - margin_x) |
                (in_well['cy'] < y0 + margin_y) |
                (in_well['cy'] > y0 + h_step - margin_y)
            )
            pct_outer_total += in_outer.sum()
            count += len(in_well)

pct_outer = pct_outer_total / count * 100 if count > 0 else 0

ax2.text(0.01, 0.01,
         f'N = {n_stopped:,} stop events  |  {pct_outer:.1f}% in peripheral zone (outer 50% area)',
         transform=ax2.transAxes, fontsize=8, color='black',
         va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig2.tight_layout()
fig2.savefig('Figure_SpatialDensity_StoppingThigmotaxis.png')
print("  Saved: Figure_SpatialDensity_StoppingThigmotaxis.png")
plt.close(fig2)

print("\nDone! Both spatial density figures saved.")
