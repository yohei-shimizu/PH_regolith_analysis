"""
検証の前段: 各メッシュから B 個の独立サブサンプルを取り、それぞれ alpha
filtration で H1/H2 の PD を計算してキャッシュする。
(V1 ブートストラップ安定性 / V2 PD間距離 で使用)
"""
from pathlib import Path
import os, time, gc
import numpy as np
import homcloud.interface as hc

_MESH = Path(__file__).resolve().parents[1]
DATA  = str(_MESH / 'data')
CACHE = str(_MESH / 'cache' / 'replicates')
os.makedirs(CACHE, exist_ok=True)
TAGS  = ['mesh01', 'mesh02', 'mesh03']
N_SUB = 30000
B     = 8           # 試料あたり複製数

for tag in TAGS:
    pts = np.loadtxt(os.path.join(DATA, f'{tag}.txt'))
    for r in range(B):
        out = os.path.join(CACHE, f'{tag}_r{r}.npz')
        if os.path.exists(out):
            continue
        rng = np.random.default_rng(1000 + r)   # 複製ごとに別シード
        idx = rng.choice(len(pts), N_SUB, replace=False)
        t0 = time.time()
        pl = hc.PDList.from_alpha_filtration(pts[idx], save_boundary_map=False)
        rec = {}
        for deg in [1, 2]:
            pd = pl.dth_diagram(deg)
            rec[f'b{deg}'] = np.asarray(pd.births)
            rec[f'd{deg}'] = np.asarray(pd.deaths)
        np.savez_compressed(out, **rec)
        print(f"{tag} r{r}: H1={len(rec['b1']):,} H2={len(rec['b2']):,} "
              f"({time.time()-t0:.0f}s)")
        del pl; gc.collect()
    del pts; gc.collect()
print("Replicates done.")
