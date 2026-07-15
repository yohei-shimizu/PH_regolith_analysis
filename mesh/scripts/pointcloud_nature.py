"""
点群の幾何学的性質を判定する: VGSTUDIO が出力した空隙点群が
「空隙表面のサンプル(2D多様体)」か「体積/重心(3D)」か「線状(1D)」かを
局所主成分分析(PCA)で判別する。これにより H2 の物理的意味を確定する。

各点の k 近傍の共分散固有値 λ1≥λ2≥λ3 から:
  linearity  = (λ1-λ2)/λ1   (大 → 線状 1D)
  planarity  = (λ2-λ3)/λ1   (大 → 面状 2D = 表面サンプル)
  sphericity =  λ3/λ1        (大 → 等方 3D = 体積/重心)
"""
from pathlib import Path
import os
import numpy as np
from scipy.spatial import cKDTree

_MESH = Path(__file__).resolve().parents[1]
DATA = str(_MESH / 'data')
TAGS = ['mesh01', 'mesh02', 'mesh03']
LAB  = {'mesh01': 'mesh01 GOOD', 'mesh02': 'mesh02 NG-3wt%', 'mesh03': 'mesh03 NG-2wt%'}
K = 20            # 近傍数
NSAMP = 8000      # 評価点数

print("Point-cloud geometric nature via local PCA")
print(f"  k-NN = {K},  sampled points = {NSAMP}")
print("=" * 70)
print(f"{'mesh':16s} {'nn-dist':>8s} {'linearity':>10s} {'planarity':>10s} "
      f"{'sphericity':>11s}  verdict")
for t in TAGS:
    pts = np.loadtxt(os.path.join(DATA, f'{t}.txt'))
    tree = cKDTree(pts)
    rng = np.random.default_rng(0)
    samp = pts[rng.choice(len(pts), NSAMP, replace=False)]
    dd, idx = tree.query(samp, k=K + 1)        # 自身含む
    nn_med = np.median(dd[:, 1])
    lin = pla = sph = 0.0
    for i in range(NSAMP):
        nb = pts[idx[i, 1:]]                    # 自身除く近傍
        nb = nb - nb.mean(0)
        cov = nb.T @ nb / (K - 1)
        ev = np.linalg.eigvalsh(cov)[::-1]     # λ1≥λ2≥λ3
        ev = ev / (ev[0] + 1e-30)
        lin += (ev[0] - ev[1])
        pla += (ev[1] - ev[2])
        sph += ev[2]
    lin /= NSAMP; pla /= NSAMP; sph /= NSAMP
    verdict = ('SURFACE (2D)' if pla > lin and pla > sph
               else 'LINEAR (1D)' if lin > pla and lin > sph
               else 'VOLUME (3D)')
    print(f"{LAB[t]:16s} {nn_med:>8.4f} {lin:>10.3f} {pla:>10.3f} "
          f"{sph:>11.3f}  {verdict}")
print("\n(planarity 最大 → 空隙表面サンプル / sphericity 最大 → 体積・重心)")
