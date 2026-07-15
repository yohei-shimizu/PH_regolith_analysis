"""
Full-resolution H1 (1st order) persistent diagram via spatial chunking.
Same strategy as compute_pd2_full.py but using dth_diagram(1).
"""

from pathlib import Path
import gc, os, tempfile, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import homcloud.interface as hc

CHUNK_SIZE = 10_000
_MESH = Path(__file__).resolve().parents[1]
DATA_DIR   = str(_MESH / 'data')
CACHE_DIR  = str(_MESH / 'cache')
OUTPUT_DIR = str(_MESH / 'output')
HIST_BINS  = 128

def spatial_chunks(indices, pts, chunk_size):
    if len(indices) <= chunk_size:
        return [indices]
    p    = pts[indices]
    axis = int(p.var(axis=0).argmax())
    med  = float(np.median(p[:, axis]))
    left  = indices[p[:, axis] <= med]
    right = indices[p[:, axis] >  med]
    return (spatial_chunks(left,  pts, chunk_size) +
            spatial_chunks(right, pts, chunk_size))

def collect_h1_pairs(pts, tag, chunk_size=CHUNK_SIZE):
    npz_path = os.path.join(CACHE_DIR, f"{tag}_pd1_full_pairs.npz")
    if os.path.exists(npz_path):
        print(f"    Loading cached pairs from {os.path.basename(npz_path)}")
        data = np.load(npz_path)
        return data['births'], data['deaths']

    chunks   = spatial_chunks(np.arange(len(pts)), pts, chunk_size)
    n_chunks = len(chunks)
    print(f"    {n_chunks} spatial chunks  ({chunk_size} pts target)")

    all_b, all_d = [], []
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdgm_path = os.path.join(tmpdir, 'chunk.pdgm')
        for i, idx in enumerate(chunks):
            try:
                pdlist = hc.PDList.from_alpha_filtration(
                    pts[idx], save_to=pdgm_path, save_boundary_map=False)
                pd1 = pdlist.dth_diagram(1)
                for pair in pd1.pairs():
                    all_b.append(pair.birth)
                    all_d.append(pair.death)
                del pdlist, pd1
            except Exception as exc:
                print(f"    [WARN] chunk {i} failed: {exc}")
            finally:
                if os.path.exists(pdgm_path):
                    os.remove(pdgm_path)
                gc.collect()

            if (i + 1) % 10 == 0 or (i + 1) == n_chunks:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_chunks - i - 1)
                print(f"    chunk {i+1:4d}/{n_chunks}  "
                      f"features: {len(all_b):,}  "
                      f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s")

    births = np.asarray(all_b, dtype=np.float64)
    deaths = np.asarray(all_d, dtype=np.float64)
    np.savez_compressed(npz_path, births=births, deaths=deaths)
    print(f"    Pairs saved to {os.path.basename(npz_path)}")
    return births, deaths

def plot_pd(births, deaths, title, png_path, hist_range, bins=HIST_BINS):
    fig, ax = plt.subplots(figsize=(7, 7))
    valid = (deaths > births) & (births >= hist_range[0]) & (deaths <= hist_range[1])
    b, d  = births[valid], deaths[valid]
    if len(b) > 0:
        h, xe, ye = np.histogram2d(b, d, bins=bins, range=[hist_range, hist_range])
        im = ax.pcolormesh(xe, ye,
                           np.ma.masked_where(h==0, np.log1p(h)).T,
                           cmap='hot_r', shading='flat')
        plt.colorbar(im, ax=ax, label='log(1 + count)')
        pct = 100 * len(b) / len(births) if len(births) else 0
        ax.text(0.02, 0.97,
                f"{len(b):,} H1 features ({pct:.1f}%)\nrange: [0, {hist_range[1]:.4f}]",
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'No H1 features in range',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.plot(hist_range, hist_range, 'k--', lw=0.8)
    ax.set_xlim(*hist_range); ax.set_ylim(*hist_range)
    ax.set_xlabel('Birth', fontsize=12); ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"    -> saved: {os.path.basename(png_path)}")

FILES = [('mesh01.txt','mesh01'), ('mesh02.txt','mesh02'), ('mesh03.txt','mesh03')]

# Phase 1: collect pairs
all_pairs = {}
for fname, tag in FILES:
    print(f"\n{'='*60}\nProcessing {tag} ...")
    t0 = time.time()
    pts = np.loadtxt(os.path.join(DATA_DIR, fname))
    n_total = len(pts)
    print(f"  {n_total:,} points (raw coordinates)")
    births, deaths = collect_h1_pairs(pts, tag)
    all_pairs[tag] = (births, deaths, n_total)
    print(f"  {len(births):,} H1 pairs  ({time.time()-t0:.0f} s)")
    del pts; gc.collect()

# Phase 2: shared range from 99.9th percentile within (0,1) candidates
print("\nDetermining shared histogram range ...")
upper_candidates = []
for tag, (b, d, _) in all_pairs.items():
    valid = (d > b) & (d <= 1.0)
    if valid.sum():
        p999 = float(np.percentile(d[valid], 99.9))
        upper_candidates.append(p999)
        print(f"  {tag}: 99.9th pct (in ≤1) = {p999:.5f}  max death = {d[d>b].max():.5f}")

shared_upper = max(upper_candidates) * 1.05
hist_range   = (0.0, round(shared_upper, 5))
print(f"  -> shared hist range: {hist_range}")

# Phase 3: plot
print("\nPlotting ...")
for tag, (births, deaths, n_total) in all_pairs.items():
    title = (f"H1 PD — {tag}  (full {n_total:,} pts, "
             f"spatial chunks of {CHUNK_SIZE})")
    png_path = os.path.join(OUTPUT_DIR, f"{tag}_pd1_full.png")
    plot_pd(births, deaths, title, png_path, hist_range)

print("\nAll done.")
