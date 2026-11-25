"""Score LLM runs against classical baselines on exactly the same hours."""
import sys, json, glob, pathlib, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = pathlib.Path(__file__).resolve().parents[1]
s = pd.read_parquet(ROOT / 'data/hours.parquet', columns=['hour', 'target', 'split']); f = pd.read_parquet(ROOT / 'data/hour_features.parquet')
d = s.merge(f, on='hour'); tr = d[d.split == 'train']; d['hour'] = d.hour.astype(str)
mean_c = [c for c in f.columns if c.endswith('|mean')]
base = {'train mean': lambda X: np.full(len(X), tr.target.mean())}
for nm, cols in [('ridge iron-only', ['% Iron Concentrate|mean']), ('ridge hourly means', mean_c)]:
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(tr[cols], tr.target); base[nm] = (lambda m, c: lambda X: m.predict(X[c]))(m, cols)
files = sys.argv[1:] or sorted(glob.glob(str(ROOT / 'results/inkling2s_*.jsonl')))
runs = {pathlib.Path(p).stem: pd.DataFrame([json.loads(l) for l in open(p)]).drop_duplicates('hour', keep='last') for p in files}
common = set.intersection(*[set(r.hour) for r in runs.values()])
def met(y, p): e = p - y; return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()), bias=e.mean(), corr=np.corrcoef(y, p)[0, 1] if len(y) > 2 else np.nan)
rows = []
for nm, r in runs.items():
    ok = r[r.pred.notna()]; rows.append(dict(model=nm, parse_rate=len(ok) / len(r), **met(ok.target.values, ok.pred.values)))
    rows[-1].update(out_tok=np.mean([u.get('output_tokens', 0) for u in r.usage if u]) if 'usage' in r else None)
X = d[d.hour.isin(common)]
for nm, fn in base.items(): rows.append(dict(model=f'[baseline] {nm} (on common {len(X)})', **met(X.target.values, fn(X))))
for nm, r in runs.items():
    ok = r[r.hour.isin(common) & r.pred.notna()]; rows.append(dict(model=f'{nm} (on common)', **met(ok.target.values, ok.pred.values)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
