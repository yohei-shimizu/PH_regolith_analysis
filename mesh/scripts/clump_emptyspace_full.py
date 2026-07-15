"""
点群=空隙表面 と確定したことを踏まえ、ダマ(=空隙表面が無い充填領域)を
直接測る。フル点群(密度情報を保持)で、領域内ランダム点から最近接の
空隙表面点までの距離 D を計算。D が大きい = 空隙の無い固体塊(ダマ)が大きい。

これは H2(=空隙)とは相補的に、固体相(ダマ)を直接定量する独立指標。
"""
from pathlib import Path
import os
import numpy as np
from scipy.spatial import cKDTree, Delaunay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_MESH = Path(__file__).resolve().parents[1]
DATA = str(_MESH / 'data')
OUT  = str(_MESH / 'output')
TAGS = ['mesh01', 'mesh02', 'mesh03']
LAB  = {'mesh01': 'mesh01 GOOD/classified',
        'mesh02': 'mesh02 NG/unclass 3wt%',
        'mesh03': 'mesh03 NG/unclass 2wt%'}
COL  = {'mesh01': '#1f77b4', 'mesh02': '#ff7f0e', 'mesh03': '#2ca02c'}
M_TEST = 50000

def emptyspace_full(pts, m=M_TEST):
    mn, mx = pts.min(0), pts.max(0)
    rng = np.random.default_rng(3)
    hull = Delaunay(pts[rng.choice(len(pts), 20000, replace=False)])
    tree = cKDTree(pts)
    got = []
    tot = 0
    while tot < m:
        cand = rng.uniform(mn, mx, size=(m, 3))
        inside = cand[hull.find_simplex(cand) >= 0]
        got.append(inside); tot += len(inside)
    test = np.vstack(got)[:m]
    d, _ = tree.query(test, k=1)
    return d

print("Clump (solid-region) size via empty-space on FULL point clouds")
print("  points = pore SURFACE samples -> empty space = solid (clump)")
print("=" * 66)
print(f"{'mesh':28s} {'Npts':>9s} {'median':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s}")
emp = {}
for t in TAGS:
    pts = np.loadtxt(os.path.join(DATA, f'{t}.txt'))
    d = emptyspace_full(pts); emp[t] = d
    print(f"{LAB[t]:28s} {len(pts):>9,} {np.median(d):>8.4f} "
          f"{np.percentile(d,90):>8.4f} {np.percentile(d,99):>8.4f} {d.max():>8.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for t in TAGS:
    d = np.sort(emp[t]); y = np.arange(1, len(d)+1)/len(d)
    axes[0].plot(d, y, color=COL[t], lw=1.8, label=LAB[t])
    axes[1].plot(d, 1-y, color=COL[t], lw=1.8, label=LAB[t])
axes[0].set_title('Clump size CDF  F(r)\n(dist to nearest pore surface = solid depth)',
                  fontsize=10)
axes[0].set_xlabel('r ~ solid/clump size'); axes[0].set_ylabel('cumulative prob.')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[1].set_yscale('log')
axes[1].set_title('Clump size tail 1-F(r)\n(large r = big resin clumps / ダマ)', fontsize=10)
axes[1].set_xlabel('r ~ solid/clump size'); axes[1].set_ylabel('P(D>r)')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
fig.suptitle('Direct clump (solid-phase) measurement — full point clouds '
             '(points = pore surfaces)', fontsize=12)
plt.tight_layout()
o = os.path.join(OUT, 'clump_emptyspace_full.png')
plt.savefig(o, dpi=150); plt.close(fig)
print(f"\n-> {o}")
