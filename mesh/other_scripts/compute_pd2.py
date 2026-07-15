"""
Compute H2 (2nd order) persistent diagrams for mesh01/02/03.txt
using three downsampling methods: random, top-N, and spatial matching.

Raw (non-normalised) coordinates are used so that birth/death values
fall naturally in the (0, 1) range — matching the reference approach
from pathlib import Path
from analysis.ipynb cell 7.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import homcloud.interface as hc
import gc
import os

N         = 10000
RNG_SEED  = 42
HIST_BINS = 128
HIST_RANGE = (0.0, 1.0)
_MESH = Path(__file__).resolve().parents[1]
DATA_DIR   = str(_MESH / 'data')
OUTPUT_DIR = str(_MESH / 'other_png')

# ── Data loading ──────────────────────────────────────────────────────────────

print("Loading data...")
meshes = {}
for tag, fname in [('mesh01', 'mesh01.txt'),
                   ('mesh02', 'mesh02.txt'),
                   ('mesh03', 'mesh03.txt')]:
    pts = np.loadtxt(os.path.join(DATA_DIR, fname))
    meshes[tag] = pts
    print(f"  {tag}: {pts.shape[0]:,} points")

# ── Common region (intersection of bboxes, for spatial-match sampling) ────────

bboxes = {}
for tag, pts in meshes.items():
    bboxes[tag] = {ax: (pts[:, i].min(), pts[:, i].max())
                   for i, ax in enumerate(['x', 'y', 'z'])}

common = {ax: (max(bboxes[t][ax][0] for t in bboxes),
               min(bboxes[t][ax][1] for t in bboxes))
          for ax in ['x', 'y', 'z']}

print("\nCommon bounding box (intersection):")
for ax in ['x', 'y', 'z']:
    lo, hi = common[ax]
    print(f"  {ax}=[{lo:.3f}, {hi:.3f}]")

# ── Downsampling functions ────────────────────────────────────────────────────

rng = np.random.default_rng(RNG_SEED)

def sample_random(pts):
    idx = rng.choice(len(pts), N, replace=False)
    return pts[idx]

def sample_top(pts):
    return pts[:N].copy()

def sample_spatial(pts):
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = common['x'], common['y'], common['z']
    mask = (
        (pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
        (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi) &
        (pts[:, 2] >= zlo) & (pts[:, 2] <= zhi)
    )
    filtered = pts[mask]
    print(f"    -> {len(filtered):,} pts in common region", end='')
    if len(filtered) >= N:
        idx = rng.choice(len(filtered), N, replace=False)
        result = filtered[idx]
    else:
        print(f" (fewer than {N}, using all)", end='')
        result = filtered
    print()
    return result

# ── PD computation and plotting ───────────────────────────────────────────────

def compute_and_plot(pts, title, png_path):
    """Compute H2 persistent diagram on raw (non-normalised) coordinates."""
    pdgm_path = png_path.replace('.png', '.pdgm')
    print(f"    Computing alpha filtration ({len(pts)} pts)...")
    pdlist = hc.PDList.from_alpha_filtration(
        pts,
        save_to=pdgm_path,
        save_boundary_map=True,
    )
    pd2 = pdlist.dth_diagram(2)

    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        # Same call as analysis.ipynb cell 7
        pd2.histogram(HIST_RANGE, HIST_BINS).plot(ax=ax, colorbar={"type": "log"})
    except Exception as e:
        print(f"    histogram() failed ({e}), scatter fallback")
        pairs = list(pd2.pairs())
        if pairs:
            b = np.array([p.birth for p in pairs])
            d = np.array([p.death for p in pairs])
            ax.scatter(b, d, s=5, alpha=0.5, c='steelblue', edgecolors='none')
            ax.plot(HIST_RANGE, HIST_RANGE, 'k--', lw=0.8)
            ax.set_xlim(*HIST_RANGE); ax.set_ylim(*HIST_RANGE)
            ax.set_xlabel('Birth'); ax.set_ylabel('Death')
        else:
            ax.text(0.5, 0.5, 'No H2 features', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13)

    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)

    del pdlist, pd2
    gc.collect()
    print(f"    Saved: {os.path.basename(png_path)}")

# ── Run all combinations ──────────────────────────────────────────────────────

methods = [
    ('random',  'Random',        sample_random),
    ('top',     'Top rows',      sample_top),
    ('spatial', 'Spatial match', sample_spatial),
]

for method_key, method_label, sampler in methods:
    print(f"\n=== Method: {method_label} ===")
    for tag, pts in meshes.items():
        print(f"  {tag}:")
        sampled = sampler(pts)
        title = f"H2 PD — {tag}  ({method_label}, n={len(sampled)})"
        png_path = os.path.join(OUTPUT_DIR, f"{tag}_pd2_{method_key}.png")
        compute_and_plot(sampled, title, png_path)

print("\nAll done.")
