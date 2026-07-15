"""
V1: PH記述子のブートストラップ安定性
    各メッシュ B 複製の記述子の 試料内ばらつき vs 試料間分離 を比較。
V2: パーシステント図間距離 (bottleneck / Wasserstein)
    同一試料の複製間距離 ≪ 異試料間距離 を示す (安定性定理に基づく判別妥当性)。
"""
from pathlib import Path
import os, itertools
import numpy as np
import homcloud.interface as hc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_MESH = Path(__file__).resolve().parents[1]
REP  = str(_MESH / 'cache' / 'replicates')
OUT  = str(_MESH / 'output')
TAGS = ['mesh01', 'mesh02', 'mesh03']
LAB  = {'mesh01': 'GOOD', 'mesh02': 'NG-3wt%', 'mesh03': 'NG-2wt%'}
COL  = {'mesh01': '#1f77b4', 'mesh02': '#ff7f0e', 'mesh03': '#2ca02c'}
B    = 8
PTHR = {1: 0.003, 2: 0.002}

def load(tag, r, deg):
    z = np.load(os.path.join(REP, f'{tag}_r{r}.npz'))
    b, d = z[f'b{deg}'], z[f'd{deg}']
    v = (d > b) & np.isfinite(d)
    return b[v], d[v]

# ════════════════════════════════════════════════════════════════════════════
# V1: 記述子のブートストラップ安定性
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("V1: bootstrap stability of PH descriptors  (B=%d replicates/mesh)" % B)
print("=" * 70)

def descriptors(tag, r):
    b1, d1 = load(tag, r, 1); b2, d2 = load(tag, r, 2)
    p1, p2 = d1 - b1, d2 - b2
    k1, k2 = p1 >= PTHR[1], p2 >= PTHR[2]
    # H2 count (per 1000 pts; N=30000)
    h2_count = k2.sum() / 30000 * 1000
    h2_totalP = p2[k2].sum()
    h2_sqrtbirth_med = np.median(np.sqrt(np.clip(b2[k2], 0, None)))
    h1_totalP = p1[k1].sum()
    return dict(h2_count=h2_count, h2_totalP=h2_totalP,
                h2_sb=h2_sqrtbirth_med, h1_totalP=h1_totalP)

vals = {t: [descriptors(t, r) for r in range(B)] for t in TAGS}
keys = ['h2_count', 'h2_totalP', 'h2_sb', 'h1_totalP']
klab = {'h2_count': 'H2 count /1000pts', 'h2_totalP': 'H2 total persist.',
        'h2_sb': 'H2 sqrt-birth med', 'h1_totalP': 'H1 total persist.'}
stat = {t: {k: np.array([v[k] for v in vals[t]]) for k in keys} for t in TAGS}

for k in keys:
    print(f"\n  {klab[k]}:")
    for t in TAGS:
        a = stat[t][k]
        print(f"    {LAB[t]:9s} mean={a.mean():>10.4f}  std={a.std():>9.4f}  "
              f"CV={a.std()/abs(a.mean())*100:>5.1f}%")
    # 良品 vs 不良 の分離 (z = |Δmean| / pooled std)
    g = stat['mesh01'][k]
    for t in ['mesh02', 'mesh03']:
        o = stat[t][k]
        pooled = np.sqrt((g.var() + o.var()) / 2)
        z = abs(g.mean() - o.mean()) / (pooled + 1e-12)
        print(f"      GOOD vs {LAB[t]}: separation z = {z:.1f} sigma")

# ── V1 図: 記述子の箱ひげ ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
for ax, k in zip(axes, keys):
    bp = ax.boxplot([stat[t][k] for t in TAGS], labels=[LAB[t] for t in TAGS],
                    patch_artist=True, widths=0.6)
    for patch, t in zip(bp['boxes'], TAGS):
        patch.set_facecolor(COL[t]); patch.set_alpha(0.6)
    ax.set_title(klab[k], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('V1 Bootstrap stability of PH descriptors (8 independent subsamples/mesh, N=30k)',
             fontsize=12)
plt.tight_layout()
o1 = os.path.join(OUT, 'validate_stability.png')
plt.savefig(o1, dpi=150); plt.close(fig); print(f"\n  -> {o1}")

# ════════════════════════════════════════════════════════════════════════════
# V2: PD間距離 (bottleneck) — 試料内 vs 試料間
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("V2: PD distances (bottleneck) within vs between samples")
print("=" * 70)

# 全複製の H2 PD (persistence閾値で軽量化) を用意
items = [(t, r) for t in TAGS for r in range(B)]
def make_pd(tag, r, deg):
    b, d = load(tag, r, deg)
    p = d - b; k = p >= PTHR[deg]
    return hc.PD.from_birth_death(deg, b[k], d[k])

DEG = 2   # ダマ(H2)で評価
pds = {(t, r): make_pd(t, r, DEG) for (t, r) in items}

n = len(items)
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        dist = hc.distance.bottleneck(pds[items[i]], pds[items[j]])
        D[i, j] = D[j, i] = dist

# 試料内/試料間 平均
labels = [t for (t, r) in items]
within, between = [], []
for i in range(n):
    for j in range(i + 1, n):
        (within if labels[i] == labels[j] else between).append(D[i, j])
within, between = np.array(within), np.array(between)
print(f"\n  H{DEG} bottleneck distance:")
print(f"    within-sample  : mean={within.mean():.4f}  std={within.std():.4f}")
print(f"    between-sample : mean={between.mean():.4f}  std={between.std():.4f}")
print(f"    separation ratio (between/within) = {between.mean()/within.mean():.2f}")

# ペア別 試料間距離
print("\n  Between-pair mean bottleneck:")
for a, b in itertools.combinations(TAGS, 2):
    ds = [D[i, j] for i in range(n) for j in range(n)
          if i < j and {labels[i], labels[j]} == {a, b}]
    print(f"    {LAB[a]:9s} vs {LAB[b]:9s}: {np.mean(ds):.4f}")

# ── V2 図: 距離ヒートマップ + 古典MDS ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im = axes[0].imshow(D, cmap='viridis')
axes[0].set_title(f'H{DEG} PD bottleneck distance matrix\n(8 replicates x 3 meshes)',
                  fontsize=10)
ticks = [B*i + B//2 for i in range(3)]
axes[0].set_xticks(ticks); axes[0].set_xticklabels([LAB[t] for t in TAGS])
axes[0].set_yticks(ticks); axes[0].set_yticklabels([LAB[t] for t in TAGS])
for i in range(1, 3):
    axes[0].axhline(B*i - 0.5, color='w', lw=1); axes[0].axvline(B*i - 0.5, color='w', lw=1)
plt.colorbar(im, ax=axes[0], label='bottleneck distance')

# 古典MDS (double-centering)
J = np.eye(n) - np.ones((n, n))/n
Bm = -0.5 * J @ (D**2) @ J
w, V = np.linalg.eigh(Bm)
order = np.argsort(w)[::-1]
emb = V[:, order[:2]] * np.sqrt(np.clip(w[order[:2]], 0, None))
for t in TAGS:
    m = np.array([lab == t for lab in labels])
    axes[1].scatter(emb[m, 0], emb[m, 1], s=80, color=COL[t], label=LAB[t],
                    edgecolors='k', alpha=0.8)
axes[1].set_title('Classical MDS of PD distances\n(replicates cluster by mesh = robust)',
                  fontsize=10)
axes[1].set_xlabel('MDS-1'); axes[1].set_ylabel('MDS-2')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
fig.suptitle('V2 PD-distance validation: within-sample << between-sample', fontsize=12)
plt.tight_layout()
o2 = os.path.join(OUT, 'validate_pd_distance.png')
plt.savefig(o2, dpi=150); plt.close(fig); print(f"\n  -> {o2}")
print("\nAll done.")
