"""
顕微鏡写真の Persistent Homology テクスチャ解析 (分級前13 / 分級後15)。
2D画像 sublevel filtration なので H0(連結成分) と H1(ループ) のみ。
(PH2 は 3D の X線CT でのみ存在)

指標 (二値化閾値に依らない):
  - persistence 分布 (= 特徴のコントラスト/明瞭さ)
  - total persistence  : Σ(d-b)  テクスチャ複雑さ
  - persistence entropy: 分布の均一さ
  - 高persistence特徴数 : 明瞭な粒の数
微細分が多い(分級前)ほど 低persistence の特徴が多い と予想。
"""
from pathlib import Path
import os
import numpy as np
import homcloud.interface as hc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = str(Path(__file__).resolve().parent)
IMG = {'before (13)': 'img260512-13_pd1.pdgm',
       'after (15)':  'img260512-15_pd1.pdgm'}
COL = {'before (13)': '#d62728', 'after (15)': '#2ca02c'}

def get(pdgm, deg):
    pd = hc.PDList(pdgm).dth_diagram(deg)
    b, d = np.asarray(pd.births), np.asarray(pd.deaths)
    v = np.isfinite(d) & (d > b)
    return b[v], d[v]

def pers_entropy(p):
    p = p[p > 0]
    w = p / p.sum()
    return float(-(w * np.log(w)).sum())

rows = []
data = {}
for name, f in IMG.items():
    for deg in [0, 1]:
        b, d = get(os.path.join(DIR, f), deg)
        pers = d - b
        data[(name, deg)] = pers
        rows.append((name, deg, len(pers), pers.sum(), np.median(pers),
                     (pers > 0.05).sum(), (pers > 0.10).sum(),
                     pers_entropy(pers)))

print(f"{'image':12s} {'deg':>3s} {'n':>7s} {'totalP':>9s} {'medP':>7s} "
      f"{'n(P>.05)':>9s} {'n(P>.10)':>9s} {'entropy':>8s}")
for r in rows:
    print(f"{r[0]:12s} H{r[1]:<2d} {r[2]:>7,} {r[3]:>9.1f} {r[4]:>7.4f} "
          f"{r[5]:>9,} {r[6]:>9,} {r[7]:>8.3f}")

# ── 図: persistence 分布 (H0, H1) ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bins = np.linspace(0, 0.4, 60)
for k, deg in enumerate([0, 1]):
    for name in IMG:
        p = data[(name, deg)]
        axes[k].hist(p, bins=bins, histtype='step', lw=2, color=COL[name],
                     density=True, label=name)
    axes[k].set_title(f'H{deg} persistence distribution', fontsize=11)
    axes[k].set_xlabel('persistence (death - birth) = contrast', fontsize=10)
    axes[k].set_ylabel('probability density', fontsize=10)
    axes[k].legend(fontsize=9); axes[k].grid(True, alpha=0.3)
    axes[k].set_yscale('log')
fig.suptitle('Microscope-image PH texture: persistence distribution '
             '(before vs after classification)', fontsize=12)
plt.tight_layout()
out = os.path.join(DIR, 'microscope_ph_persistence.png')
plt.savefig(out, dpi=150); plt.close(fig)
print(f"\n-> {out}")

# 指標を保存
summary = {}
for name in IMG:
    for deg in [0, 1]:
        p = data[(name, deg)]
        key = name.split()[0]
        summary[f'{key}_H{deg}_n']       = len(p)
        summary[f'{key}_H{deg}_totalP']  = float(p.sum())
        summary[f'{key}_H{deg}_medP']    = float(np.median(p))
        summary[f'{key}_H{deg}_nP05']    = int((p > 0.05).sum())
        summary[f'{key}_H{deg}_entropy'] = pers_entropy(p)
np.savez(os.path.join(DIR, 'microscope_ph_metrics.npz'), **summary)
print("Saved PH metrics npz. Done.")
