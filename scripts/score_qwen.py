import json, sys, glob, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
s = pd.read_parquet('data/hours.parquet', columns=['hour', 'target', 'split']); f = pd.read_parquet('data/hour_features.parquet')
d = s.merge(f, on='hour'); tr = d[d.split == 'train']; d['hour'] = d.hour.astype(str)
mc = [c for c in f.columns if c.endswith('|mean')]
rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(tr[mc], tr.target)
rf = make_pipeline(StandardScaler(), RidgeCV()).fit(tr[['% Iron Concentrate|mean']], tr.target)
rows = []
for fn in sys.argv[1:]:
    R = pd.DataFrame([json.loads(l) for l in open(fn)]); ok = R[R.pred.notna()]
    row = dict(run=fn.split('/')[-1].replace('.jsonl', ''), parsed=R.pred.notna().mean(), trunc=(R.finish == 'length').mean(), tok=R.n_tokens.mean())
    if len(ok) > 2:
        e = ok.pred - ok.target; hm = ok.groupby('hour').agg(pred=('pred', 'median'), target=('target', 'first'))
        row.update(MAE=e.abs().mean(), medAE=e.abs().median(), bias=e.mean(), corr=np.corrcoef(ok.target, ok.pred)[0, 1],
                   MAE_med4=(hm.pred - hm.target).abs().mean(), corr_med4=np.corrcoef(hm.target, hm.pred)[0, 1], hours=len(hm),
                   within_sd=ok.groupby('hour').pred.std().median())
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
X = d[d.hour.isin(R.hour.unique())]
print('baselines on same 200 h: train-mean MAE', round(abs(tr.target.mean() - X.target).mean(), 3), '| ridge Fe', round(abs(rf.predict(X[['% Iron Concentrate|mean']]) - X.target).mean(), 3),
      '| ridge means', round(abs(rg.predict(X[mc]) - X.target).mean(), 3))
