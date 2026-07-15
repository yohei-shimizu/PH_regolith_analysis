"""
V3改: PH非依存の古典統計でダマ(H2)構造を独立に検証する。
中心指標 = 空間接触分布 (spherical contact / empty-space function):
  領域内のランダム点から最近接の空隙点までの距離 D の分布。
  D が大きい = 空隙が無い領域(=充填体/ダマ)が大きい。
  → H2(空隙に囲まれた充填体)の独立対応量。
公平化: 同点数 N、テスト点は凸包内に限定して外部空間を除外。
補助: 複数スケールの計数変動 CV(L)。
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
N_SUB, SEED = 80000, 7
M_TEST = 40000          # テスト点数

def subsample(tag):
    pts = np.loadtxt(os.path.join(DATA, f'{tag}.txt'))
    idx = np.random.default_rng(SEED).choice(len(pts), N_SUB, replace=False)
    return pts[idx]

def empty_space(pts, m=M_TEST):
    """凸包内ランダム点から最近接空隙点までの距離分布。"""
    mn, mx = pts.min(0), pts.max(0)
    rng = np.random.default_rng(3)
    # 凸包 (サブセットで構築して高速化)
    hull_pts = pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)]
    dela = Delaunay(hull_pts)
    tree = cKDTree(pts)
    got = []
    while len(got) < m:
        cand = rng.uniform(mn, mx, size=(m, 3))
        inside = dela.find_simplex(cand) >= 0
        got.append(cand[inside])
        if sum(len(g) for g in got) >= m:
            break
    test = np.vstack(got)[:m]
    d, _ = tree.query(test, k=1)
    return d

def cv_multiscale(pts, Ls):
    """各 voxel 辺長 L での計数変動 CV(L)。"""
    mn, mx = pts.min(0), pts.max(0); span = mx - mn
    out = []
    for L in Ls:
        nb = np.maximum((span / L).astype(int), 1)
        edges = [np.linspace(mn[i], mx[i], nb[i] + 1) for i in range(3)]
        H, _ = np.histogramdd(pts, bins=edges)
        c = H.ravel()
        out.append(c.std() / c.mean())
    return np.array(out)

print("V3': empty-space function (PH-independent H2 analog)")
print("=" * 64)
data = {t: subsample(t) for t in TAGS}

emp = {}
print(f"\nEmpty-space distance D (interior random pts -> nearest void pt)")
print(f"{'mesh':28s} {'median':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s}")
for t in TAGS:
    d = empty_space(data[t]); emp[t] = d
    print(f"{LAB[t]:28s} {np.median(d):>8.4f} {np.percentile(d,90):>8.4f} "
          f"{np.percentile(d,99):>8.4f} {d.max():>8.4f}")

Ls = np.array([0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0])
cvs = {t: cv_multiscale(data[t], Ls) for t in TAGS}

# ── 図 ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# (1) empty-space CDF
for t in TAGS:
    d = np.sort(emp[t]); y = np.arange(1, len(d)+1)/len(d)
    axes[0].plot(d, y, color=COL[t], lw=1.8, label=LAB[t])
axes[0].set_title('Empty-space function F(r)\n(dist to nearest void pt)', fontsize=10)
axes[0].set_xlabel('r (empty-space distance ~ solid/clump size)', fontsize=9)
axes[0].set_ylabel('cumulative prob.', fontsize=9)
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3); axes[0].set_xlim(0, None)
# (2) empty-space tail (survival, log)
for t in TAGS:
    d = np.sort(emp[t]); surv = 1 - np.arange(1, len(d)+1)/len(d)
    axes[1].plot(d, surv, color=COL[t], lw=1.8, label=LAB[t])
axes[1].set_yscale('log')
axes[1].set_title('Empty-space tail 1-F(r)\n(large r = big clumps)', fontsize=10)
axes[1].set_xlabel('r', fontsize=9); axes[1].set_ylabel('P(D > r)', fontsize=9)
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
# (3) CV(L)
for t in TAGS:
    axes[2].plot(Ls, cvs[t], 'o-', color=COL[t], lw=1.6, label=LAB[t])
axes[2].set_xscale('log')
axes[2].set_title('(c) Count-variation CV vs voxel size L', fontsize=10)
axes[2].set_xlabel('voxel edge L', fontsize=9); axes[2].set_ylabel('CV', fontsize=9)
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

fig.suptitle('V3 Empty-space function & multiscale clustering (PH-independent)',
             fontsize=12)
plt.tight_layout()
o = os.path.join(OUT, 'validate_emptyspace.png')
plt.savefig(o, dpi=150); plt.close(fig)
print(f"\n-> {o}")
np.savez(os.path.join(OUT, 'validate_emptyspace.npz'),
         **{f'{t}_emp': emp[t] for t in TAGS}, Ls=Ls,
         **{f'{t}_cv': cvs[t] for t in TAGS})
print("Done.")
