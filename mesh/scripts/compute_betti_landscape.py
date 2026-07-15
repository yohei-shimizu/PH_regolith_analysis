"""
(1) persistence 閾値でノイズ除去した正規化パーシステントベッチ数
(2) パーシステンス・ランドスケープによる H1 比較

cache/*.npz (birth, death) から計算。
- 正規化: β_k(r) / N   (N = 各メッシュの点数)
- ノイズ除去: persistence = death - birth が閾値以上のペアのみ採用
- ランドスケープ: λ_j(t) = j 番目に大きいテント関数値 (厳密、chunk 処理)
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
TAGS       = ['mesh01', 'mesh02', 'mesh03']
NPTS       = {'mesh01': 749984, 'mesh02': 509246, 'mesh03': 380795}
MESH_COL   = {'mesh01': '#1f77b4', 'mesh02': '#ff7f0e', 'mesh03': '#2ca02c'}

# 次数ごとの persistence 閾値（分布の percentile 調査に基づく）
PERS_THR = {
    1: [0.0, 0.001, 0.003, 0.010],
    2: [0.0, 0.0005, 0.002, 0.005],
}


def load_pairs(tag, degree):
    path = os.path.join(CACHE_DIR, f'{tag}_pd{degree}_full_pairs.npz')
    data = np.load(path)
    b, d = data['births'], data['deaths']
    valid = (d > b) & np.isfinite(d)
    return b[valid], d[valid]


def betti_curve(births, deaths, r_values):
    bs = np.sort(births)
    ds = np.sort(deaths)
    return (np.searchsorted(bs, r_values, side='right')
            - np.searchsorted(ds, r_values, side='right')).astype(np.float64)


# ════════════════════════════════════════════════════════════════════════════
# Part 1: persistence 閾値つき 正規化ベッチ数
# ════════════════════════════════════════════════════════════════════════════

print("=" * 64)
print("Part 1: noise-thresholded normalized Betti numbers")
print("=" * 64)

# r の評価範囲は H1/H2 とも signal が収まる [0, 0.1] を共通に使う
R_MAX  = 0.10
N_EVAL = 800
r_values = np.linspace(0.0, R_MAX, N_EVAL)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

betti_summary = {}   # (tag, deg, thr) -> peak normalized betti

for row, deg in enumerate([1, 2]):
    thrs = PERS_THR[deg]
    for col, tag in enumerate(TAGS):
        ax = axes[row][col]
        b, d = load_pairs(tag, deg)
        pers = d - b
        N = NPTS[tag]

        for thr in thrs:
            keep = pers >= thr
            bk, dk = b[keep], d[keep]
            curve = betti_curve(bk, dk, r_values) / N * 1000.0  # per 1000 pts
            peak = curve.max()
            betti_summary[(tag, deg, thr)] = (peak, len(bk))
            ax.plot(r_values, curve, lw=1.4,
                    label=f'thr={thr:g}  (n={len(bk):,})')

        ax.set_title(f'{tag} — H{deg}  (normalized β_{deg}/N ×1000)', fontsize=10)
        ax.set_xlabel('r  (alpha radius²)', fontsize=9)
        ax.set_ylabel(f'β_{deg}(r) per 1000 pts', fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, R_MAX)
        ax.set_ylim(bottom=0)

fig.suptitle('Noise-thresholded Normalized Persistent Betti Numbers '
             '(persistence ≥ thr; per 1000 points)', fontsize=12)
plt.tight_layout()
out_betti = os.path.join(OUTPUT_DIR, 'betti_normalized_thresholded.png')
plt.savefig(out_betti, dpi=150)
plt.close(fig)
print(f"  -> {out_betti}")

# サマリー表
for deg in [1, 2]:
    print(f"\n  -- H{deg}: peak normalized β (per 1000 pts) --")
    thrs = PERS_THR[deg]
    head = "  thr \\ mesh   " + "".join(f"{t:>14s}" for t in TAGS)
    print(head)
    for thr in thrs:
        row = f"  {thr:<10g}  "
        for tag in TAGS:
            peak, n = betti_summary[(tag, deg, thr)]
            row += f"{peak:>14.4f}"
        print(row)


# ════════════════════════════════════════════════════════════════════════════
# Part 2: パーシステンス・ランドスケープ (H1)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("Part 2: persistence landscape (H1)")
print("=" * 64)

K_LAYERS = 5          # λ_1 .. λ_5
T_MAX    = 0.10
T_EVAL   = 600
CHUNK    = 4000
t_values = np.linspace(0.0, T_MAX, T_EVAL)
# ランドスケープは signal を見たいので軽くノイズ除去 (persistence ≥ 1e-4)
LS_PERS_THR = 1e-4


def landscape(births, deaths, t_values, K, chunk=CHUNK):
    """λ_1..λ_K を厳密に計算 (chunk ごとに列方向 top-K を更新)。"""
    T = len(t_values)
    topk = np.zeros((K, T), dtype=np.float64)   # 昇順保持: topk[-1]=λ_1
    t = t_values[None, :]
    for s in range(0, len(births), chunk):
        b = births[s:s+chunk][:, None]
        d = deaths[s:s+chunk][:, None]
        tents = np.maximum(0.0, np.minimum(t - b, d - t))   # (m, T)
        cat = np.vstack([topk, tents])
        cat.sort(axis=0)
        topk = cat[-K:]
    # λ_1 = 最大 → 降順に並べ替えて返す
    return topk[::-1]   # shape (K, T): [λ_1, λ_2, ..., λ_K]


landscapes = {}
for tag in TAGS:
    b, d = load_pairs(tag, 1)
    keep = (d - b) >= LS_PERS_THR
    b, d = b[keep], d[keep]
    lam = landscape(b, d, t_values, K_LAYERS)
    landscapes[tag] = lam
    l1_norm = np.trapz(lam.sum(axis=0), t_values)
    l2_norm = np.sqrt(np.trapz((lam**2).sum(axis=0), t_values))
    print(f"  {tag}: n(pers≥{LS_PERS_THR})={len(b):,}  "
          f"L1={l1_norm:.5f}  L2={l2_norm:.5f}  "
          f"L1/N={l1_norm/NPTS[tag]*1e6:.4f}e-6")

# --- 図A: 層ごとに 3 メッシュを重ねる (λ_1, λ_2, λ_3) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for j, ax in enumerate(axes):
    for tag in TAGS:
        ax.plot(t_values, landscapes[tag][j], color=MESH_COL[tag],
                lw=1.5, label=tag)
    ax.set_title(f'H1 Landscape  λ_{j+1}(t)', fontsize=11)
    ax.set_xlabel('t  (alpha radius²)', fontsize=9)
    ax.set_ylabel(f'λ_{j+1}', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(bottom=0)
fig.suptitle('Persistence Landscape comparison (H1) — layers λ_1, λ_2, λ_3',
             fontsize=12)
plt.tight_layout()
out_ls1 = os.path.join(OUTPUT_DIR, 'landscape_h1_by_layer.png')
plt.savefig(out_ls1, dpi=150)
plt.close(fig)
print(f"  -> {out_ls1}")

# --- 図B: メッシュごとに λ_1..λ_5 を重ねる ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, tag in zip(axes, TAGS):
    lam = landscapes[tag]
    for j in range(K_LAYERS):
        ax.plot(t_values, lam[j], lw=1.3, label=f'λ_{j+1}')
    ax.set_title(f'{tag} — H1 Landscape', fontsize=11)
    ax.set_xlabel('t  (alpha radius²)', fontsize=9)
    ax.set_ylabel('λ_j', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(bottom=0)
fig.suptitle('Persistence Landscape (H1) — layers per mesh', fontsize=12)
plt.tight_layout()
out_ls2 = os.path.join(OUTPUT_DIR, 'landscape_h1_by_mesh.png')
plt.savefig(out_ls2, dpi=150)
plt.close(fig)
print(f"  -> {out_ls2}")

print("\nAll done.")
