import pandas as pd, numpy as np, json
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error, r2_score
import lightgbm as lgb
s = pd.read_parquet('data/hours.parquet', columns=['hour', 'target', 'split', 'interpolated'])
f = pd.read_parquet('data/hour_features.parquet')
d = s.merge(f, on='hour')
tr, te = d[d.split == 'train'], d[d.split == 'test']
allc = [c for c in f.columns if c != 'hour']
mean_c = [c for c in allc if c.endswith('|mean')]
noiron = [c for c in allc if not c.startswith('% Iron Concentrate')]
iron_only = ['% Iron Concentrate|mean']
def ev(name, p, mask=None):
    y = te.target.values
    r = dict(model=name, MAE=MAE(y, p), RMSE=mean_squared_error(y, p) ** .5, R2=r2_score(y, p))
    m = ~te.interpolated.values
    r['MAE_nointerp'] = MAE(y[m], p[m]); return r
res = [ev('train mean', np.full(len(te), tr.target.mean()))]
for nm, cols in [('ridge iron-conc only', iron_only), ('ridge hourly means', mean_c), ('ridge all aggs', allc), ('ridge all aggs, no iron conc', noiron)]:
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(tr[cols], tr.target)
    res.append(ev(nm, m.predict(te[cols])))
for nm, cols in [('lightgbm all aggs', allc), ('lightgbm no iron conc', noiron)]:
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0).fit(tr[cols], tr.target)
    res.append(ev(nm, m.predict(te[cols])))
# stoichiometric guess (no training): SiO2 ~ 100 - Fe*1.4297 (hematite) - other gangue; report fit of slope only
r = pd.DataFrame(res).round(3); print(r.to_string(index=False)); r.to_csv('results/baselines.csv', index=False)
