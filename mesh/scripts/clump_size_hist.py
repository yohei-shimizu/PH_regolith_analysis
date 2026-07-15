"""
(a) ダマ(固体塊)サイズ分布: H2 の √birth ヒストグラムを3試料で比較。
空隙点群の H2 = 空隙に囲まれた充填領域 = 粒子/ダマ。
birth (α半径²) の √ がその塊の形成スケール ≒ ダマ径の指標。
"""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_MESH = Path(__file__).resolve().parents[1]
CACHE_DIR  = str(_MESH / 'cache')
OUTPUT_DIR = str(_MESH / 'output')
TAGS  = ['mesh01', 'mesh02', 'mesh03']
LAB   = {'mesh01': 'mesh01 GOOD / classified / 3wt%',
         'mesh02': 'mesh02 NG / unclassified / 3wt%',
         'mesh03': 'mesh03 NG / unclassified / 2wt%'}
COL   = {'mesh01': '#1f77b4', 'mesh02': '#ff7f0e', 'mesh03': '#2ca02c'}
NPTS  = {'mesh01': 749984, 'mesh02': 509246, 'mesh03': 380795}
PERS_THR = 0.002          # ノイズ除去 (persistence ≥ thr)

def load(tag, deg):
    d = np.load(os.path.join(CACHE_DIR, f'{tag}_pd{deg}_full_pairs.npz'))
    b, dd = d['births'], d['deaths']
    v = (dd > b) & np.isfinite(dd)
    return b[v], dd[v]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bins = np.linspace(0.0, 0.5, 60)

for tag in TAGS:
    b, d = load(tag, 2)
    keep = (d - b) >= PERS_THR
    sb = np.sqrt(np.clip(b[keep], 0, None))     # √birth = ダマ形成スケール
    # 左: 正規化頻度 (密度比較・点数差を吸収)
    axes[0].hist(sb, bins=bins, histtype='step', lw=1.8,
                 color=COL[tag], density=True, label=LAB[tag])
    # 右: 単位点あたり個数 (per 1000 pts) で絶対量も比較
    w = np.full(len(sb), 1000.0 / NPTS[tag])
    axes[1].hist(sb, bins=bins, histtype='step', lw=1.8,
                 color=COL[tag], weights=w, label=LAB[tag])
    print(f"{LAB[tag]}: n(H2,pers≥{PERS_THR})={keep.sum():,}  "
          f"√birth median={np.median(sb):.4f}  p90={np.percentile(sb,90):.4f}")

axes[0].set_title('Clump-size distribution (normalized: prob. density)', fontsize=11)
axes[0].set_ylabel('probability density', fontsize=10)
axes[1].set_title('Clump-size distribution (count per 1000 void pts)', fontsize=11)
axes[1].set_ylabel('count per 1000 void points', fontsize=10)
for ax in axes:
    ax.set_xlabel('sqrt(birth)  (solid-lump formation scale ~ clump size)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)

fig.suptitle('(a) Clump-size distribution from H2 birth scale  (persistence ≥ %.3f)'
             % PERS_THR, fontsize=12)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'clump_size_hist.png')
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"-> {out}")
