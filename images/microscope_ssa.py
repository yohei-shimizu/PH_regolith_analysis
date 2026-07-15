"""
顕微鏡写真 (分級前 *13 / 分級後 *15) から比表面積・粒径を定量化。
月面ロボットのカメラによる原料判別を想定し、樹脂コート前レゴリスの
状態指標を抽出する。

指標:
  - エッジ密度 (Sobel勾配, Cannyエッジ長/面積) = 界面量 ~ 比表面積(2D)
  - 二値化(Otsu)した明粒子の 周囲長/面積 = 比周囲長 (specific perimeter)
  - 粒径分布 (連結成分の等価円直径), 微細分割合
全画像同一倍率(スケールバー2.0mm)なので相対比較は単位に依らない。
"""
from pathlib import Path
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, sobel
from skimage.feature import canny
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import remove_small_objects

_HERE = Path(__file__).resolve().parent
IMG_DIR = str(_HERE / 'raw')
OUT_DIR = str(_HERE)
IMAGES = {'before (13, 分級前)': 'img260512-13.jpg',
          'after  (15, 分級後)': 'img260512-15.jpg'}
COL = {'before (13, 分級前)': '#d62728', 'after  (15, 分級後)': '#2ca02c'}
CROP_BOTTOM = 300   # 右下オーバーレイ除去

def load(fname):
    g = np.array(Image.open(os.path.join(IMG_DIR, fname)).convert('L'),
                 dtype=np.float64)
    g = g[:-CROP_BOTTOM, :]      # 下端のスケールバー/日時を除外
    return g

results = {}
for name, fname in IMAGES.items():
    g = load(fname)
    gn = g / 255.0
    H, W = g.shape
    area = H * W

    # --- エッジ密度 (界面量 ~ 比表面積) ---
    sob = sobel(gn)
    edge_mean = float(sob.mean())
    cny = canny(gn, sigma=2.0)
    edge_len_density = float(cny.sum()) / area     # エッジ画素/面積

    # --- 明粒子の二値化と比周囲長 ---
    t = threshold_otsu(g)
    binmask = g > t                                # 明部=粒子表面
    lab = label(binmask)
    lab = remove_small_objects(lab, min_size=20)
    props = regionprops(lab)
    diams = np.array([p.equivalent_diameter for p in props])  # px
    grain_area_frac = binmask.mean()
    peri_total = perimeter(binmask)                # 総界面長
    specific_perimeter = peri_total / max(binmask.sum(), 1)   # 周囲長/粒子面積

    fines_frac = float((diams < 15).sum()) / max(len(diams), 1)  # <15px を微細分

    results[name] = dict(
        edge_mean=edge_mean, edge_len_density=edge_len_density,
        grain_area_frac=grain_area_frac,
        specific_perimeter=specific_perimeter,
        n_grains=len(diams), diams=diams,
        d_median=float(np.median(diams)), d_mean=float(diams.mean()),
        fines_frac=fines_frac, otsu=t)
    print(f"\n[{name}]  ({H}x{W})")
    print(f"  Sobel edge mean          : {edge_mean:.5f}")
    print(f"  Canny edge length / area : {edge_len_density:.5f}  (~specific surface)")
    print(f"  grain area fraction      : {grain_area_frac:.3f}")
    print(f"  specific perimeter       : {specific_perimeter:.5f}")
    print(f"  n grains                 : {len(diams):,}")
    print(f"  equiv-diam median / mean : {np.median(diams):.2f} / {diams.mean():.2f} px")
    print(f"  fines fraction (<15px)   : {fines_frac:.3f}")

# ── 図: 粒径分布 + バー指標 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (1) 粒径分布
bins = np.logspace(np.log10(3), np.log10(300), 40)
for name in IMAGES:
    axes[0].hist(results[name]['diams'], bins=bins, histtype='step', lw=2,
                 color=COL[name], density=True, label=name.split('(')[0])
axes[0].set_xscale('log')
axes[0].set_title('Particle size distribution', fontsize=11)
axes[0].set_xlabel('equivalent diameter [px]', fontsize=10)
axes[0].set_ylabel('probability density', fontsize=10)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# (2) 比表面積系指標
metrics = ['edge_mean', 'edge_len_density', 'specific_perimeter']
mlab = ['Sobel edge\nmean', 'Canny edge\nlen/area', 'specific\nperimeter']
x = np.arange(len(metrics)); w = 0.35
for i, name in enumerate(IMAGES):
    vals = [results[name][m] for m in metrics]
    # 各指標を before で正規化して相対比較
    axes[1].bar(x + (i-0.5)*w, vals, w, color=COL[name], label=name.split('(')[0])
axes[1].set_xticks(x); axes[1].set_xticklabels(mlab, fontsize=9)
axes[1].set_title('Specific-surface-area proxies', fontsize=11)
axes[1].set_ylabel('value', fontsize=10)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis='y')

# (3) 粒数・微細分割合
names = list(IMAGES.keys())
fines = [results[n]['fines_frac'] for n in names]
dmed  = [results[n]['d_median'] for n in names]
axb = axes[2]
xb = np.arange(len(names))
b1 = axb.bar(xb-0.2, fines, 0.4, color='#888', label='fines frac (<15px)')
axb2 = axb.twinx()
b2 = axb2.bar(xb+0.2, dmed, 0.4, color='#1f77b4', label='median diam [px]')
axb.set_xticks(xb); axb.set_xticklabels([n.split('(')[0] for n in names], fontsize=9)
axb.set_ylabel('fines fraction', fontsize=10)
axb2.set_ylabel('median diameter [px]', fontsize=10)
axb.set_title('Fines fraction & median grain size', fontsize=11)
axb.legend(loc='upper left', fontsize=8); axb2.legend(loc='upper right', fontsize=8)

fig.suptitle('Microscope-image specific surface area & grain size '
             '(before vs after classification)', fontsize=12)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'microscope_ssa.png')
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"\n-> {out}")

# 結果を npz 保存 (PPT/後段比較用)
np.savez(os.path.join(OUT_DIR, 'microscope_ssa.npz'),
         **{f"{k}_{m}": results[k][m]
            for k in results for m in
            ['edge_mean','edge_len_density','grain_area_frac',
             'specific_perimeter','d_median','d_mean','fines_frac','n_grains']})
print("Saved metrics npz. Done.")
