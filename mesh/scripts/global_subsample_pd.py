"""
(b) 大スケール再計算: 全点を同数に一様サブサンプルし、分割なしで
alpha filtration を実行。チャンク境界アーティファクトを排除し、
試料全体スケール (ダマ・大空隙) の H1/H2 を 3 試料で直接比較する。

全試料を同じ点数にするため正規化不要で比較可能。
"""
from pathlib import Path
import os, gc, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import homcloud.interface as hc

_MESH = Path(__file__).resolve().parents[1]
DATA_DIR   = str(_MESH / 'data')
OUTPUT_DIR = str(_MESH / 'output')
CACHE_DIR  = str(_MESH / 'cache')
TAGS  = ['mesh01', 'mesh02', 'mesh03']
LAB   = {'mesh01': 'mesh01 GOOD / classified / 3wt%',
         'mesh02': 'mesh02 NG / unclassified / 3wt%',
         'mesh03': 'mesh03 NG / unclassified / 2wt%'}
COL   = {'mesh01': '#1f77b4', 'mesh02': '#ff7f0e', 'mesh03': '#2ca02c'}
N_SUB = 50000          # 全試料共通のサブサンプル点数
SEED  = 7

def compute_subsample(tag):
    npz = os.path.join(CACHE_DIR, f'{tag}_sub{N_SUB}_pd.npz')
    if os.path.exists(npz):
        print(f"  [{tag}] load cache")
        z = np.load(npz)
        return {1: (z['b1'], z['d1']), 2: (z['b2'], z['d2'])}
    pts = np.loadtxt(os.path.join(DATA_DIR, f'{tag}.txt'))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pts), size=N_SUB, replace=False)
    sub = pts[idx]
    print(f"  [{tag}] {len(pts):,} -> {N_SUB:,} pts, running alpha filtration ...")
    t0 = time.time()
    pdlist = hc.PDList.from_alpha_filtration(sub, save_boundary_map=False)
    out = {}
    for deg in [1, 2]:
        pd = pdlist.dth_diagram(deg)
        out[deg] = (np.asarray(pd.births), np.asarray(pd.deaths))
    print(f"  [{tag}] done ({time.time()-t0:.0f}s)  "
          f"H1={len(out[1][0]):,}  H2={len(out[2][0]):,}")
    np.savez_compressed(npz, b1=out[1][0], d1=out[1][1],
                        b2=out[2][0], d2=out[2][1])
    del pdlist; gc.collect()
    return out

# ── 計算 ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"(b) Global subsample alpha filtration  (N={N_SUB}, no chunking)")
print("=" * 60)
res = {tag: compute_subsample(tag) for tag in TAGS}

def clean(b, d):
    v = (d > b) & np.isfinite(d)
    return b[v], d[v]

def betti(b, d, r):
    return (np.searchsorted(np.sort(b), r, 'right')
            - np.searchsorted(np.sort(d), r, 'right')).astype(float)

# ── 図1: PD (H1, H2) を試料ごと・次数ごとに 2x3 散布 ──────────────────────────
print("\nPlotting PD scatter ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, deg in enumerate([1, 2]):
    # 共通レンジ
    alld = []
    for tag in TAGS:
        b, d = clean(*res[tag][deg]); alld.append(d)
    hi = float(np.percentile(np.concatenate(alld), 99)) * 1.1
    for col, tag in enumerate(TAGS):
        ax = axes[row][col]
        b, d = clean(*res[tag][deg])
        ax.scatter(b, d, s=3, alpha=0.25, color=COL[tag], edgecolors='none')
        ax.plot([0, hi], [0, hi], 'k--', lw=0.7)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(f'{LAB[tag]}\nH{deg}: {len(b):,} features', fontsize=9)
        ax.set_xlabel('Birth (alpha r^2)', fontsize=9)
        ax.set_ylabel('Death (alpha r^2)', fontsize=9)
        ax.grid(True, alpha=0.3)
fig.suptitle(f'(b) Global PD (subsample N={N_SUB}, no chunking)', fontsize=12)
plt.tight_layout()
o1 = os.path.join(OUTPUT_DIR, 'global_pd_scatter.png')
plt.savefig(o1, dpi=150); plt.close(fig); print(f"  -> {o1}")

# ── 図2: Betti 曲線 (同点数なので直接比較) ───────────────────────────────────
print("Plotting global Betti curves ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k, deg in enumerate([1, 2]):
    alld = [clean(*res[t][deg])[1] for t in TAGS]
    rmax = float(np.percentile(np.concatenate(alld), 99.5))
    r = np.linspace(0, rmax, 800)
    for tag in TAGS:
        b, d = clean(*res[tag][deg])
        axes[k].plot(r, betti(b, d, r), color=COL[tag], lw=1.6, label=LAB[tag])
    axes[k].set_title(f'Global H{deg} Betti  (same N={N_SUB})', fontsize=11)
    axes[k].set_xlabel('r (alpha r^2)', fontsize=10)
    axes[k].set_ylabel(f'beta_{deg}(r)', fontsize=10)
    axes[k].legend(fontsize=8); axes[k].grid(True, alpha=0.3)
    axes[k].set_xlim(0, rmax); axes[k].set_ylim(bottom=0)
fig.suptitle(f'(b) Global Persistent Betti (subsample N={N_SUB})', fontsize=12)
plt.tight_layout()
o2 = os.path.join(OUTPUT_DIR, 'global_betti.png')
plt.savefig(o2, dpi=150); plt.close(fig); print(f"  -> {o2}")

# ── 図3: 大スケール H2 (=ダマ) の √birth 分布 ───────────────────────────────
print("Plotting global clump-size (H2 sqrt-birth) ...")
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 2.0, 50)
for tag in TAGS:
    b, d = clean(*res[tag][2])
    sb = np.sqrt(np.clip(b, 0, None))
    ax.hist(sb, bins=bins, histtype='step', lw=1.8, color=COL[tag],
            label=f"{LAB[tag]}  (n={len(b):,})")
ax.set_title(f'(b) Global clump size: H2 sqrt(birth)  (N={N_SUB})', fontsize=11)
ax.set_xlabel('sqrt(birth) ~ clump size (global scale)', fontsize=10)
ax.set_ylabel('count', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
o3 = os.path.join(OUTPUT_DIR, 'global_clump_size.png')
plt.savefig(o3, dpi=150); plt.close(fig); print(f"  -> {o3}")

# ── サマリー ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'mesh':8s} {'H1 n':>8s} {'H2 n':>7s} {'H2 √birth p50':>13s} {'H2 √birth p90':>13s}")
for tag in TAGS:
    b1, d1 = clean(*res[tag][1]); b2, d2 = clean(*res[tag][2])
    sb = np.sqrt(np.clip(b2, 0, None))
    print(f"{tag:8s} {len(b1):>8,} {len(b2):>7,} "
          f"{np.percentile(sb,50):>13.4f} {np.percentile(sb,90):>13.4f}")
print("All done.")
