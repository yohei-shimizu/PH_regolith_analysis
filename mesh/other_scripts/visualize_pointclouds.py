"""
Point cloud comparison visualisation for mesh01/02/03.txt.

Outputs:
  pointcloud_overview.png   — 3-D overlay + three 2-D projection grids
  pointcloud_density.png    — per-axis density (histogram) comparison
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import os

RNG_SEED = 42
VIS_N    = 8000          # points per file for scatter plots
DENSITY_N = 50_000       # points per file for density histograms
_MESH = Path(__file__).resolve().parents[1]
DATA_DIR   = str(_MESH / 'data')
OUTPUT_DIR = str(_MESH / 'other_png')
COLORS  = ['#e05c5c', '#4e8fdf', '#3cbd7a']      # red / blue / green
LABELS  = ['mesh01', 'mesh02', 'mesh03']
FILES   = ['mesh01.txt', 'mesh02.txt', 'mesh03.txt']

rng = np.random.default_rng(RNG_SEED)

# ── Load & subsample ──────────────────────────────────────────────────────────

print("Loading data...")
full  = []
small = []   # for scatter
dense = []   # for density
for fname in FILES:
    pts = np.loadtxt(os.path.join(DATA_DIR, fname))
    full.append(pts)
    idx_s = rng.choice(len(pts), min(VIS_N,   len(pts)), replace=False)
    idx_d = rng.choice(len(pts), min(DENSITY_N, len(pts)), replace=False)
    small.append(pts[idx_s])
    dense.append(pts[idx_d])
    print(f"  {fname}: {len(pts):,} pts")

# ── Figure 1: overview  (3-D + 2-D projections) ───────────────────────────────
# Layout: 4 rows × 3 cols
#   row 0: 3-D scatter (all three files overlaid + one per file)
#   rows 1-3: xy / xz / yz projections, one column per file

print("\nRendering overview figure...")
fig = plt.figure(figsize=(18, 22))
fig.suptitle("Point Cloud Comparison — mesh01 / mesh02 / mesh03", fontsize=16, y=0.98)

# --- row 0: 3-D views --------------------------------------------------------
# col 0: all three overlaid
ax3d_all = fig.add_subplot(4, 3, 1, projection='3d')
for pts, col, lbl in zip(small, COLORS, LABELS):
    ax3d_all.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                     c=col, s=0.4, alpha=0.4, linewidths=0, label=lbl)
ax3d_all.set_title("Overlay (all 3)", fontsize=10)
ax3d_all.set_xlabel('X'); ax3d_all.set_ylabel('Y'); ax3d_all.set_zlabel('Z')
ax3d_all.legend(markerscale=6, fontsize=8)

# cols 1-2: individual 3-D views for mesh01 and mesh02 (mesh03 shown via projections)
for col_idx, (pts, col, lbl) in enumerate(zip(small[:2], COLORS[:2], LABELS[:2]), start=2):
    ax = fig.add_subplot(4, 3, col_idx, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=col, s=0.4, alpha=0.5, linewidths=0)
    ax.set_title(lbl, fontsize=10, color=col)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# --- rows 1-3: 2-D projections -----------------------------------------------
proj_cfg = [
    ('XY', 0, 1, 'X', 'Y'),
    ('XZ', 0, 2, 'X', 'Z'),
    ('YZ', 1, 2, 'Y', 'Z'),
]
for row_idx, (plane, xi, yi, xl, yl) in enumerate(proj_cfg, start=1):
    for col_idx, (pts, col, lbl) in enumerate(zip(small, COLORS, LABELS), start=1):
        ax = fig.add_subplot(4, 3, row_idx * 3 + col_idx)
        ax.scatter(pts[:, xi], pts[:, yi],
                   c=col, s=0.3, alpha=0.35, linewidths=0, rasterized=True)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_title(f"{lbl}  [{plane}]", fontsize=9, color=col)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = os.path.join(OUTPUT_DIR, 'pointcloud_overview.png')
plt.savefig(out1, dpi=150)
plt.close(fig)
print(f"  Saved: {os.path.basename(out1)}")

# ── Figure 2: density / distribution comparison ──────────────────────────────
print("Rendering density figure...")
BINS = 120

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Point Cloud Density Comparison — mesh01 / mesh02 / mesh03",
             fontsize=14, y=0.995)

axes_labels = ['X', 'Y', 'Z']
for row, (axis_lbl, ai) in enumerate(zip(axes_labels, [0, 1, 2])):
    for col, (pts, col_c, lbl) in enumerate(zip(dense, COLORS, LABELS)):
        ax = axes[row, col]
        ax.hist(pts[:, ai], bins=BINS, color=col_c, alpha=0.75,
                edgecolor='none', density=True)
        ax.set_title(f"{lbl} — {axis_lbl} distribution", fontsize=9, color=col_c)
        ax.set_xlabel(axis_lbl, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

# Bottom row: 2-D hexbin density maps (XY projection) for each file
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("XY Density Map — mesh01 / mesh02 / mesh03", fontsize=13)
for ax, pts, col_c, lbl in zip(axes2, dense, COLORS, LABELS):
    hb = ax.hexbin(pts[:, 0], pts[:, 1], gridsize=80, cmap='hot_r',
                   mincnt=1, linewidths=0)
    plt.colorbar(hb, ax=ax, label='count')
    ax.set_title(lbl, fontsize=11, color=col_c)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, 'pointcloud_density_xy.png')
plt.savefig(out3, dpi=150)
plt.close(fig2)

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'pointcloud_density.png')
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"  Saved: {os.path.basename(out2)}")
print(f"  Saved: {os.path.basename(out3)}")

# ── Figure 3: Z-slice comparison (horizontal cross-sections) ─────────────────
print("Rendering Z-slice figure...")
z_vals = [0.25, 0.5, 0.75]    # fractional Z positions

# normalise Z for each file individually to pick representative slices
fig, axes = plt.subplots(len(z_vals), 3, figsize=(16, 14))
fig.suptitle("Horizontal Cross-Sections (Z-slices) — mesh01 / mesh02 / mesh03",
             fontsize=13, y=0.995)

for row, frac in enumerate(z_vals):
    for col, (pts, col_c, lbl) in enumerate(zip(full, COLORS, LABELS)):
        zlo, zhi = pts[:, 2].min(), pts[:, 2].max()
        z_mid  = zlo + frac * (zhi - zlo)
        z_band = (zhi - zlo) * 0.02          # ±1 % of z-range
        mask   = np.abs(pts[:, 2] - z_mid) < z_band
        sl     = pts[mask]
        # subsample if too many
        if len(sl) > 5000:
            sl = sl[rng.choice(len(sl), 5000, replace=False)]
        ax = axes[row, col]
        ax.scatter(sl[:, 0], sl[:, 1], c=col_c, s=0.8, alpha=0.5,
                   linewidths=0, rasterized=True)
        ax.set_title(f"{lbl}  z≈{z_mid:.1f}  (n={len(sl):,})", fontsize=8, color=col_c)
        ax.set_xlabel('X', fontsize=7); ax.set_ylabel('Y', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout(rect=[0, 0, 1, 0.985])
out4 = os.path.join(OUTPUT_DIR, 'pointcloud_zslices.png')
plt.savefig(out4, dpi=150)
plt.close(fig)
print(f"  Saved: {os.path.basename(out4)}")

print("\nAll done.")
