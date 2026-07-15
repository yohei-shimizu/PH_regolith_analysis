"""
キャッシュ済みペアデータ (cache/*.npz) から
パーシステントベッチ数 β_1(r) および β_2(r) を計算して可視化する。

β_k(r) = #{(b,d) : b ≤ r < d}  (filtration parameter r における k 次 Betti 数)

効率化: np.searchsorted により O(N log N + M log N) で計算。
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
N_EVAL     = 1000   # r サンプル点数


# ── データ読み込み ────────────────────────────────────────────────────────────

def load_pairs(tag, degree):
    path = os.path.join(CACHE_DIR, f'{tag}_pd{degree}_full_pairs.npz')
    data = np.load(path)
    b, d = data['births'], data['deaths']
    valid = (d > b) & np.isfinite(d)
    return b[valid], d[valid]


# ── ベッチ数計算 ──────────────────────────────────────────────────────────────

def persistent_betti_curve(births, deaths, r_values):
    """β_k(r) を r_values 全点について一括計算。"""
    bs = np.sort(births)
    ds = np.sort(deaths)
    n_born = np.searchsorted(bs, r_values, side='right')
    n_dead = np.searchsorted(ds, r_values, side='right')
    return (n_born - n_dead).astype(np.int64)


# ── ペアの読み込みとベッチ曲線の計算 ─────────────────────────────────────────

print("Loading pairs and computing Betti curves ...")
results = {}   # (tag, deg) -> (r_values, betti_curve)

for tag in TAGS:
    for deg in [1, 2]:
        b, d = load_pairs(tag, deg)
        r_max = float(np.percentile(d, 99.9))
        r_values = np.linspace(0.0, r_max, N_EVAL)
        betti = persistent_betti_curve(b, d, r_values)
        results[(tag, deg)] = (r_values, betti, b, d)
        peak = int(betti.max())
        r_peak = float(r_values[betti.argmax()])
        print(f"  {tag} H{deg}: pairs={len(b):,}  peak β={peak:,} @ r={r_peak:.5f}  r_max={r_max:.5f}")


# ── 図1: メッシュごとに H1・H2 を重ねたサブプロット (1×3) ───────────────────

print("\nPlotting Figure 1: per-mesh H1+H2 ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {'H1': '#1f77b4', 'H2': '#d62728'}

for ax, tag in zip(axes, TAGS):
    for deg in [1, 2]:
        r, betti, _, _ = results[(tag, deg)]
        ax.plot(r, betti, color=colors[f'H{deg}'],
                label=f'β_{deg}(r)', lw=1.5)
    ax.set_xlabel('r  (alpha radius²)', fontsize=10)
    ax.set_ylabel('Persistent Betti number', fontsize=10)
    ax.set_title(tag, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

fig.suptitle('Persistent Betti Numbers β_1(r) and β_2(r) — mesh01/02/03  (spatial chunks)',
             fontsize=12)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'betti_per_mesh.png')
plt.savefig(out1, dpi=150)
plt.close(fig)
print(f"  -> {out1}")


# ── 図2: 次数ごとに 3 メッシュを重ねたサブプロット (1×2) ──────────────────

print("Plotting Figure 2: per-degree all meshes ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
mesh_colors = {'mesh01': '#1f77b4', 'mesh02': '#ff7f0e', 'mesh03': '#2ca02c'}

for ax, deg in zip(axes, [1, 2]):
    for tag in TAGS:
        r, betti, _, _ = results[(tag, deg)]
        ax.plot(r, betti, color=mesh_colors[tag], label=tag, lw=1.5)
    ax.set_xlabel('r  (alpha radius²)', fontsize=10)
    ax.set_ylabel(f'β_{deg}(r)', fontsize=10)
    ax.set_title(f'H{deg} Persistent Betti Number', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

fig.suptitle('Persistent Betti Numbers — mesh01/02/03  (spatial chunks)', fontsize=12)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'betti_per_degree.png')
plt.savefig(out2, dpi=150)
plt.close(fig)
print(f"  -> {out2}")


# ── 図3: 2×3 グリッド (degree × mesh)、対数スケール付き ──────────────────

print("Plotting Figure 3: 2x3 grid with log scale ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for row, deg in enumerate([1, 2]):
    for col, tag in enumerate(TAGS):
        ax = axes[row][col]
        r, betti, b, d = results[(tag, deg)]

        ax.plot(r, betti, color=colors[f'H{deg}'], lw=1.5)

        peak = int(betti.max())
        r_peak = float(r[betti.argmax()])
        ax.axvline(r_peak, color='gray', lw=0.8, ls='--')
        ax.text(0.97, 0.95,
                f"peak: {peak:,}\n@ r={r_peak:.4f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))

        ax.set_xlabel('r  (alpha radius²)', fontsize=9)
        ax.set_ylabel(f'β_{deg}(r)', fontsize=9)
        ax.set_title(f'{tag} — H{deg}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

fig.suptitle('Persistent Betti Numbers β_1, β_2  (mesh01–03, spatial chunks)',
             fontsize=12)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, 'betti_grid.png')
plt.savefig(out3, dpi=150)
plt.close(fig)
print(f"  -> {out3}")


# ── サマリー表 ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"{'':8s}  {'H1 peak β':>12s}  {'H1 r_peak':>10s}  {'H2 peak β':>12s}  {'H2 r_peak':>10s}")
print("-" * 60)
for tag in TAGS:
    r1, b1, _, _ = results[(tag, 1)]
    r2, b2, _, _ = results[(tag, 2)]
    print(f"{tag}  {int(b1.max()):>12,}  {r1[b1.argmax()]:>10.5f}"
          f"  {int(b2.max()):>12,}  {r2[b2.argmax()]:>10.5f}")
print("=" * 60)
print("\nAll done.")
