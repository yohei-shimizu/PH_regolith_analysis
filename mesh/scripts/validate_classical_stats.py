"""
V3: PH に依存しない古典的空間統計による収束的妥当性検証。
空隙ポイントクラウドの「均一性 vs クラスタリング」を独立手法で測り、
PH の良否ランキング(良品=均一/不良=クラスタ化)を再現するか確認する。

指標:
  (a) 局所密度の変動係数 CV と Morisita 指数 Iδ  … クラスタリング強度
  (b) 動径分布関数 g(r)                          … 秩序スケール
  (c) 最近接距離分布                              … 充填の規則性
公平性: 全試料を同点数 N_SUB に統一。
"""
from pathlib import Path
import os
import numpy as np
from scipy.spatial import cKDTree
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

def subsample(tag):
    pts = np.loadtxt(os.path.join(DATA, f'{tag}.txt'))
    idx = np.random.default_rng(SEED).choice(len(pts), N_SUB, replace=False)
    return pts[idx]

def local_density_stats(pts, nvox=40):
    """voxel グリッドの点数分布から CV と Morisita Iδ。"""
    mn, mx = pts.min(0), pts.max(0)
    # 各軸を nvox 分割
    edges = [np.linspace(mn[i], mx[i], nvox + 1) for i in range(3)]
    H, _ = np.histogramdd(pts, bins=edges)
    counts = H.ravel()
    occ = counts[counts > 0]               # 占有ボクセルのみ(空隙存在域)
    cv = counts.std() / counts.mean()      # 全ボクセル基準の変動係数
    n = counts.sum(); Q = len(counts)
    morisita = Q * (counts*(counts-1)).sum() / (n*(n-1))  # Iδ: 1=ランダム,>1=集中
    return cv, morisita, occ

def pair_correlation(pts, rmax, nbins=60, n_ref=8000):
    """g(r): 参照点サブセットから動径方向の隣接点数を理想気体で正規化。"""
    box = pts.max(0) - pts.min(0)
    V = box.prod()
    rho = len(pts) / V
    tree = cKDTree(pts)
    rng = np.random.default_rng(0)
    ref = pts[rng.choice(len(pts), min(n_ref, len(pts)), replace=False)]
    edges = np.linspace(0, rmax, nbins + 1)
    counts = np.zeros(nbins)
    for p in ref:
        dd = tree.query_ball_point(p, rmax, return_length=False)
        r = np.linalg.norm(pts[dd] - p, axis=1)
        r = r[r > 1e-9]
        counts += np.histogram(r, bins=edges)[0]
    shell_vol = 4/3*np.pi*(edges[1:]**3 - edges[:-1]**3)
    g = counts / (len(ref) * rho * shell_vol)
    rc = 0.5*(edges[1:] + edges[:-1])
    return rc, g

print("V3: classical spatial statistics (PH-independent)")
print("=" * 64)
data = {t: subsample(t) for t in TAGS}

# (a) 局所密度クラスタリング
print(f"\n(a) Local-density clustering  (N={N_SUB}, 40^3 voxels)")
print(f"{'mesh':28s} {'CV':>8s} {'Morisita Iδ':>12s} {'occ.vox':>8s}")
stats = {}
for t in TAGS:
    cv, mor, occ = local_density_stats(data[t])
    stats[t] = (cv, mor)
    print(f"{LAB[t]:28s} {cv:>8.3f} {mor:>12.3f} {len(occ):>8,}")

# (b) g(r), (c) 最近接距離
print(f"\n(b)(c) computing g(r) and nearest-neighbor ...")
gr = {}; nn = {}
for t in TAGS:
    rc, g = pair_correlation(data[t], rmax=3.0)
    gr[t] = (rc, g)
    tree = cKDTree(data[t]); dd, _ = tree.query(data[t], k=2)
    nn[t] = dd[:, 1]
    print(f"  {t}: g(r) peak={g.max():.2f} @ r={rc[g.argmax()]:.3f}  "
          f"nn median={np.median(nn[t]):.4f}  nn CV={nn[t].std()/nn[t].mean():.3f}")

# ── 図 ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# (a) bar: CV & Morisita
x = np.arange(len(TAGS)); w = 0.35
cvs = [stats[t][0] for t in TAGS]; mors = [stats[t][1] for t in TAGS]
axes[0].bar(x - w/2, cvs, w, color='#888', label='density CV')
ax0b = axes[0].twinx()
ax0b.bar(x + w/2, mors, w, color='#c0392b', label='Morisita Iδ')
axes[0].set_xticks(x); axes[0].set_xticklabels([t for t in TAGS], fontsize=9)
axes[0].set_ylabel('density CV'); ax0b.set_ylabel('Morisita Iδ')
axes[0].set_title('(a) Clustering: CV & Morisita\n(higher = more clustered)', fontsize=10)
axes[0].legend(loc='upper left', fontsize=8); ax0b.legend(loc='upper right', fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')
# (b) g(r)
for t in TAGS:
    rc, g = gr[t]
    axes[1].plot(rc, g, color=COL[t], lw=1.6, label=LAB[t])
axes[1].axhline(1.0, color='k', ls='--', lw=0.7)
axes[1].set_title('(b) Pair correlation g(r)', fontsize=11)
axes[1].set_xlabel('r'); axes[1].set_ylabel('g(r)')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
# (c) nn distance
bins = np.linspace(0, 0.6, 60)
for t in TAGS:
    axes[2].hist(nn[t], bins=bins, histtype='step', lw=1.8, color=COL[t],
                 density=True, label=LAB[t])
axes[2].set_title('(c) Nearest-neighbor distance', fontsize=11)
axes[2].set_xlabel('nn distance'); axes[2].set_ylabel('density')
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

fig.suptitle('V3 Classical spatial statistics (PH-independent) — convergent validity',
             fontsize=12)
plt.tight_layout()
o = os.path.join(OUT, 'validate_classical_stats.png')
plt.savefig(o, dpi=150); plt.close(fig)
print(f"\n-> {o}")

np.savez(os.path.join(OUT, 'validate_classical_stats.npz'),
         **{f'{t}_cv': stats[t][0] for t in TAGS},
         **{f'{t}_morisita': stats[t][1] for t in TAGS})
print("Done.")
