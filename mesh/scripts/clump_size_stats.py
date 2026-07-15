"""
ダマ径分布 (H2 の √birth) の3試料間の統計検定。

主データ : (b) サブサンプル全体計算 (cache/{tag}_sub50000_pd.npz)
           - チャンク境界アーティファクト無し / 同点数 N=50000
補助データ: (a) チャンク full 計算 (cache/{tag}_pd2_full_pairs.npz)

検定:
  - 全体差: Kruskal-Wallis, k-sample Anderson-Darling
  - 対比較: 2標本 KS, Mann-Whitney U(片側), Cliff's delta(効果量),
            中央値差の bootstrap 95%CI
注意: 特徴数が多いため p 値は極小になりやすい → 効果量を併記。
"""
from pathlib import Path
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_MESH = Path(__file__).resolve().parents[1]
CACHE_DIR  = str(_MESH / 'cache')
OUTPUT_DIR = str(_MESH / 'output')
TAGS = ['mesh01', 'mesh02', 'mesh03']
LAB  = {'mesh01': 'mesh01 GOOD/classified/3wt%',
        'mesh02': 'mesh02 NG/unclassified/3wt%',
        'mesh03': 'mesh03 NG/unclassified/2wt%'}
COL  = {'mesh01': '#1f77b4', 'mesh02': '#ff7f0e', 'mesh03': '#2ca02c'}
PERS_THR = 0.002
RNG = np.random.default_rng(0)


def sqrt_birth_global(tag):
    z = np.load(os.path.join(CACHE_DIR, f'{tag}_sub50000_pd.npz'))
    b, d = z['b2'], z['d2']
    v = (d > b) & np.isfinite(d) & ((d - b) >= PERS_THR)
    return np.sqrt(np.clip(b[v], 0, None))


def sqrt_birth_chunk(tag):
    z = np.load(os.path.join(CACHE_DIR, f'{tag}_pd2_full_pairs.npz'))
    b, d = z['births'], z['deaths']
    v = (d > b) & np.isfinite(d) & ((d - b) >= PERS_THR)
    return np.sqrt(np.clip(b[v], 0, None))


def cliffs_delta(x, y):
    """δ = (#(x>y) - #(x<y)) / (nx*ny).  正なら x の方が大きい傾向。"""
    ys = np.sort(y)
    gt = np.searchsorted(ys, x, 'left').sum()          # #(y < x)
    lt = (len(ys) - np.searchsorted(ys, x, 'right')).sum()  # #(y > x)
    return (gt - lt) / (len(x) * len(ys))


def interp_delta(d):
    a = abs(d)
    return ('negligible' if a < 0.147 else 'small' if a < 0.33
            else 'medium' if a < 0.474 else 'large')


def boot_median_diff(x, y, n=5000):
    dif = np.empty(n)
    for i in range(n):
        dif[i] = (np.median(RNG.choice(x, len(x))) -
                  np.median(RNG.choice(y, len(y))))
    return np.percentile(dif, [2.5, 97.5])


def run_block(name, getter):
    print("\n" + "=" * 70)
    print(f"[{name}]  H2 sqrt(birth) = clump size  (persistence >= {PERS_THR})")
    print("=" * 70)
    data = {t: getter(t) for t in TAGS}
    for t in TAGS:
        x = data[t]
        print(f"  {LAB[t]:32s} n={len(x):>6,}  "
              f"median={np.median(x):.4f}  mean={x.mean():.4f}  "
              f"p90={np.percentile(x,90):.4f}")

    # --- 全体差 ---
    kw = stats.kruskal(*[data[t] for t in TAGS])
    print(f"\n  Kruskal-Wallis (3群同分布H0): H={kw.statistic:.2f}  p={kw.pvalue:.3e}")
    try:
        ad = stats.anderson_ksamp([data[t] for t in TAGS])
        print(f"  Anderson-Darling k-sample : A2={ad.statistic:.2f}  "
              f"p={ad.significance_level:.4f} (capped)")
    except Exception as e:
        print(f"  Anderson-Darling: {e}")

    # --- 対比較 ---
    pairs = [('mesh01', 'mesh02'), ('mesh01', 'mesh03'), ('mesh02', 'mesh03')]
    print(f"\n  {'pair':17s} {'KS D':>7s} {'KS p':>11s} {'MWU p(>)':>11s} "
          f"{'Cliff d':>8s} {'effect':>10s} {'median diff 95%CI':>22s}")
    for a, b in pairs:
        xa, xb = data[a], data[b]
        ks = stats.ks_2samp(xa, xb)
        # 片側: a の方が大きい (NG ほど大?) → ここでは a>b を検定
        mwu = stats.mannwhitneyu(xa, xb, alternative='greater')
        dlt = cliffs_delta(xa, xb)
        ci = boot_median_diff(xa, xb)
        print(f"  {a[-2:]}vs{b[-2:]:11s} {ks.statistic:>7.4f} {ks.pvalue:>11.2e} "
              f"{mwu.pvalue:>11.2e} {dlt:>8.3f} {interp_delta(dlt):>10s} "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    return data


# ── 実行 ──────────────────────────────────────────────────────────────────────
print("Clump-size (H2 sqrt-birth) statistical tests")
data_global = run_block("(b) GLOBAL subsample N=50000, no chunking", sqrt_birth_global)
data_chunk  = run_block("(a) CHUNKED full",                          sqrt_birth_chunk)

# ── ECDF 図 (主データ) ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, ttl in [(axes[0], data_global, '(b) global subsample N=50000'),
                      (axes[1], data_chunk,  '(a) chunked full')]:
    for t in TAGS:
        x = np.sort(data[t])
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=COL[t], lw=1.8, label=LAB[t])
    ax.set_title(f'ECDF of clump size (H2 sqrt-birth)\n{ttl}', fontsize=10)
    ax.set_xlabel('sqrt(birth) ~ clump size', fontsize=10)
    ax.set_ylabel('cumulative probability', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
fig.suptitle('Clump-size ECDF comparison (KS test basis)', fontsize=12)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'clump_size_ecdf.png')
plt.savefig(out, dpi=150); plt.close(fig)
print(f"\n-> {out}")
print("\nNote: large feature counts make p-values extremely small;")
print("      interpret with Cliff's delta (effect size) and median CI.")
print("      Features are spatially dependent -> p-values are optimistic.")
print("All done.")
