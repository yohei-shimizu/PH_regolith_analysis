"""
Convert images to grayscale and compute H1 (1st order) persistent diagrams
using homcloud's bitmap levelset filtration.
"""

from pathlib import Path
import os
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import homcloud.interface as hc

_HERE = Path(__file__).resolve().parent
INPUT_DIR  = str(_HERE / 'raw')
OUTPUT_DIR = str(_HERE)
IMAGES = ['img260512-13.jpg', 'img260512-15.jpg']


TARGET_SIZE = (512, 288)  # width x height (16:9, keeps aspect ratio)


def load_grayscale(path):
    img = Image.open(path).convert('L')
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)
    arr = arr / 255.0
    return arr


def compute_pd1(gray_array, pdgm_path, mode='sublevel'):
    pdlist = hc.PDList.from_bitmap_levelset(
        gray_array,
        mode=mode,
        save_to=pdgm_path,
        save_boundary_map=False,
    )
    return pdlist


def plot_pd1(pd1, title, png_path):
    pairs = pd1.pairs()
    if pairs:
        births = np.array([p.birth for p in pairs])
        deaths = np.array([p.death for p in pairs])
        # Remove infinite deaths
        finite = np.isfinite(deaths) & (deaths > births)
        births, deaths = births[finite], deaths[finite]
    else:
        births, deaths = np.array([]), np.array([])

    fig, ax = plt.subplots(figsize=(7, 7))

    if len(births) > 0:
        vmax = max(deaths.max(), births.max())
        hist_range = [0.0, 1.0]
        in_range = (births >= 0) & (deaths <= 1.0) & (deaths > births)
        b, d = births[in_range], deaths[in_range]

        if len(b) > 0:
            h, xe, ye = np.histogram2d(b, d, bins=128,
                                       range=[hist_range, hist_range])
            im = ax.pcolormesh(xe, ye,
                               np.ma.masked_where(h == 0, np.log1p(h)).T,
                               cmap='hot_r', shading='flat')
            plt.colorbar(im, ax=ax, label='log(1 + count)')
            ax.text(0.02, 0.97,
                    f"{len(b):,} H1 features",
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No H1 features in range [0,1]',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No H1 features found',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Birth (intensity)', fontsize=12)
    ax.set_ylabel('Death (intensity)', fontsize=12)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  -> saved: {png_path}")


def process_image(filename, mode='sublevel'):
    base = os.path.splitext(filename)[0]
    img_path  = os.path.join(INPUT_DIR, filename)
    pdgm_path = os.path.join(OUTPUT_DIR, f"{base}_pd1.pdgm")
    png_path  = os.path.join(OUTPUT_DIR, f"{base}_pd1.png")

    print(f"\nProcessing {filename} ...")
    gray = load_grayscale(img_path)
    print(f"  Image shape: {gray.shape},  mode: {mode}")

    # Save grayscale image for reference
    gray_png = os.path.join(OUTPUT_DIR, f"{base}_gray.png")
    Image.fromarray((gray * 255).astype(np.uint8)).save(gray_png)
    print(f"  Grayscale saved: {gray_png}")

    pdlist = compute_pd1(gray, pdgm_path, mode=mode)
    pd1 = pdlist.dth_diagram(1)
    pairs = pd1.pairs()
    print(f"  H1 features: {len(pairs)}")

    title = f"H1 Persistent Diagram — {filename}\n(grayscale {mode}level filtration)"
    plot_pd1(pd1, title, png_path)


for fname in IMAGES:
    process_image(fname, mode='sublevel')

print("\nDone.")
