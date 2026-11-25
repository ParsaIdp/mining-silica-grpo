"""Stronger per-hour baselines (no lagged labels — same information as the LLM gets), tuned with time-series CV on train only."""
import pandas as pd, numpy as np, itertools
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error, r2_score
import lightgbm as lgb
s = pd.read_parquet('data/hours.parquet', columns=['hour', 'target', 'split', 'interpolated']); f = pd.read_parquet('data/hour_features.parquet')
d = s.merge(f, on='hour'); tr, te = d[d.split == 'train'], d[d.split == 'test']
allc = [c for c in f.columns if c != 'hour']; mean_c = [c for c in allc if c.endswith('|mean')]
fe = ['% Iron Concentrate|mean']; assays = ['% Iron Concentrate|mean', '% Iron Feed|mean', '% Silica Feed|mean']
def ev(nm, p):
    y = te.target.values; return dict(model=nm, MAE=MAE(y, p), RMSE=mean_squared_error(y, p) ** .5, R2=r2_score(y, p))
res = []
def ridge(cols): return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(tr[cols], tr.target)
for nm, cols in [('ridge Fe only', fe), ('ridge 3 assays', assays), ('ridge hourly means', mean_c), ('ridge all aggs', allc)]:
    res.append(ev(nm, ridge(cols).predict(te[cols])))
m = make_pipeline(StandardScaler(), HuberRegressor(alpha=1e-3, max_iter=2000)).fit(tr[mean_c], tr.target); res.append(ev('huber hourly means', m.predict(te[mean_c])))
m = make_pipeline(SplineTransformer(n_knots=6), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(tr[mean_c], tr.target); res.append(ev('spline-additive (GAM-like) means', m.predict(te[mean_c])))
# LightGBM tuned via TimeSeriesSplit on train
tss = TimeSeriesSplit(5); best = None
for nl, lr, ne, mcs in itertools.product([4, 8, 16], [0.02, 0.05], [200, 600], [20, 80]):
    sc = []
    for a, b in tss.split(tr):
        g = lgb.LGBMRegressor(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mcs, subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_lambda=1, verbose=-1, random_state=0)
        g.fit(tr.iloc[a][mean_c], tr.iloc[a].target); sc.append(MAE(tr.iloc[b].target, g.predict(tr.iloc[b][mean_c])))
    if best is None or np.mean(sc) < best[0]: best = (np.mean(sc), dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mcs))
g = lgb.LGBMRegressor(**best[1], subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_lambda=1, verbose=-1, random_state=0).fit(tr[mean_c], tr.target)
pg = g.predict(te[mean_c]); res.append(ev(f'lightgbm TS-CV tuned {best[1]}', pg))
pr = ridge(mean_c).predict(te[mean_c]); res.append(ev('ensemble ridge+lgbm', (pr + pg) / 2))
# oracle-ish reference: ridge trained on the test period itself (in-sample upper bound of linear signal)
m = make_pipeline(StandardScaler(), RidgeCV()).fit(te[mean_c], te.target); res.append(ev('[ceiling] ridge fit IN-SAMPLE on test', m.predict(te[mean_c])))
r = pd.DataFrame(res).round(3); print(r.to_string(index=False)); r.to_csv('results/strong_baselines.csv', index=False)
