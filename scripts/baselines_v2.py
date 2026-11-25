"""Stronger classical baselines. All model/feature/hyperparameter choices are made with time-series CV
(5 expanding folds, 24 h gap) on the TRAIN period only; the test period (>= 2017-08-01) is touched once at the end.
Setting A (per-hour, same info as the LLM): current-hour sensors + feed assays + % Iron Concentrate, no past labels.
Setting B (reference only, forecasting): A + lab values from previous hours."""
import warnings, itertools, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, catboost as cb
warnings.filterwarnings('ignore')
T, FE = '% Silica Concentrate', '% Iron Concentrate'
raw = pd.read_csv('data/raw/MiningProcess_Flotation_Plant_Database.csv', decimal=',', parse_dates=['date'])
lab_cols = ['% Iron Feed', '% Silica Feed', FE]
sens = [c for c in raw.columns if c not in ['date', T] + lab_cols]
g = raw.groupby('date')
# ---- features ----
pos = g.cumcount(); n = g[T].transform('size'); raw['_third'] = (pos * 3 // n).clip(0, 2); raw['_t'] = pos / n
F = {}
for c in sens:
    s = g[c]
    F[c + '|mean'] = s.mean(); F[c + '|std'] = s.std(); F[c + '|min'] = s.min(); F[c + '|max'] = s.max()
    F[c + '|q10'] = s.quantile(.1); F[c + '|q50'] = s.median(); F[c + '|q90'] = s.quantile(.9)
    F[c + '|first3'] = raw[raw._third == 0].groupby('date')[c].mean(); F[c + '|last3'] = raw[raw._third == 2].groupby('date')[c].mean()
    F[c + '|last10m'] = g.tail(30).groupby('date')[c].mean()
for c in lab_cols: F[c + '|mean'] = g[c].mean()
X = pd.DataFrame(F); X.index.name = 'hour'
y = g[T].mean().rename('target'); interp = (g[T].nunique() > 1) | (g[FE].nunique() > 1)
X['FeConc^2'] = X[FE + '|mean'] ** 2
for c in ['Amina Flow', 'Starch Flow', 'Ore Pulp pH', 'Ore Pulp Density', 'Ore Pulp Flow']: X[f'FeConc*{c}'] = X[FE + '|mean'] * X[c + '|mean']
X['FeFeed-FeConc'] = X[FE + '|mean'] - X['% Iron Feed|mean']; X['amina/feedSi'] = X['Amina Flow|mean'] / X['% Silica Feed|mean']
X['starch/flow'] = X['Starch Flow|mean'] / X['Ore Pulp Flow|mean']; X['amina/flow'] = X['Amina Flow|mean'] / X['Ore Pulp Flow|mean']
# forecasting lags (setting B): previous *available* hours' labs (gaps -> NaN)
full = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq='h')); fefull = X[FE + '|mean'].reindex(full.index)
LAG = pd.DataFrame({f'silica_lag{k}': full.shift(k) for k in (1, 2, 3)} | {f'fe_lag{k}': fefull.shift(k) for k in (1, 2)}).reindex(X.index)
LAG['silica_lag_mean3'] = LAG[['silica_lag1', 'silica_lag2', 'silica_lag3']].mean(axis=1)
tr = X.index < '2017-08-01'; te = ~tr
cols_mean = [c for c in X.columns if c.endswith('|mean')]
cols_rich = list(X.columns)
cols_lin = cols_mean + [c for c in X.columns if c.startswith('FeConc') or '/' in c or '-' in c] + [c + '|last3' for c in sens] + [c + '|std' for c in sens]
SETS = {'means': cols_mean, 'lin+': cols_lin, 'rich': cols_rich}
tss = TimeSeriesSplit(5, gap=24)
def wts(idx, end, hl):
    if hl is None or (isinstance(hl, float) and np.isnan(hl)): return None
    age = np.asarray((end - idx).total_seconds()) / 86400; return np.asarray(0.5 ** (age / hl), dtype=float)
def fit_pred(make, Xtr, ytr, Xte, w):
    m = make()
    if w is None: m.fit(Xtr, ytr)
    elif hasattr(m, 'steps'): m.fit(Xtr, ytr, **{f'{m.steps[-1][0]}__sample_weight': w})
    else: m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xte)
def cv(make, cols, hl=None, Xd=None):
    Xd = X if Xd is None else Xd; Xt, yt = Xd.loc[tr, cols], y[tr]; err, oof = [], pd.Series(np.nan, index=Xt.index)
    for a, b in tss.split(Xt):
        p = fit_pred(make, Xt.iloc[a], yt.iloc[a], Xt.iloc[b], wts(Xt.index[a], Xt.index[a][-1], hl)); oof.iloc[b] = p
        mb = ~interp[Xt.index[b]].values; err.append(np.abs(p - yt.iloc[b].values)[mb].mean())
    return np.mean(err), oof
def test(make, cols, hl=None, Xd=None):
    Xd = X if Xd is None else Xd
    return pd.Series(fit_pred(make, Xd.loc[tr, cols], y[tr], Xd.loc[te, cols], wts(X.index[tr], X.index[tr][-1], hl)), index=X.index[te])
CANDS = []
for alpha in [0.1, 1, 10, 100, 1000]:
    for fs in SETS: CANDS.append((f'ridge a={alpha} [{fs}]', (lambda a=alpha: make_pipeline(StandardScaler(), Ridge(alpha=a))), fs))
for a, l1 in itertools.product([0.003, 0.01, 0.03, 0.1], [0.2, 0.8]):
    CANDS.append((f'enet a={a} l1={l1} [lin+]', (lambda a=a, l1=l1: make_pipeline(StandardScaler(), ElasticNet(alpha=a, l1_ratio=l1, max_iter=20000))), 'lin+'))
CANDS.append(('huber [lin+]', lambda: make_pipeline(StandardScaler(), HuberRegressor(alpha=1e-2, max_iter=5000)), 'lin+'))
for C, gm in itertools.product([0.3, 1, 3], [0.003, 0.01, 0.03]):
    CANDS.append((f'svr C={C} g={gm} [means]', (lambda C=C, gm=gm: make_pipeline(StandardScaler(), SVR(C=C, gamma=gm, epsilon=0.1))), 'means'))
for nl, lr, ne in itertools.product([4, 8, 16], [0.02, 0.05], [200, 500]):
    for fs in ['means', 'rich']:
        CANDS.append((f'lgbm nl={nl} lr={lr} ne={ne} [{fs}]', (lambda nl=nl, lr=lr, ne=ne: lgb.LGBMRegressor(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.6, reg_lambda=1, verbose=-1, random_state=0, n_jobs=4)), fs))
for d_, ne in itertools.product([2, 3, 4], [300, 800]):
    CANDS.append((f'xgb d={d_} ne={ne} [rich]', (lambda d_=d_, ne=ne: xgb.XGBRegressor(max_depth=d_, n_estimators=ne, learning_rate=0.03, subsample=.8, colsample_bytree=.6, min_child_weight=10, reg_lambda=2, n_jobs=4)), 'rich'))
    CANDS.append((f'catboost d={d_+2} it={ne} [rich]', (lambda d_=d_, ne=ne: cb.CatBoostRegressor(depth=d_ + 2, iterations=ne, learning_rate=0.03, loss_function='MAE', verbose=0, thread_count=4, random_seed=0)), 'rich'))
for msl in [5, 20]:
    CANDS.append((f'extratrees msl={msl} [rich]', (lambda msl=msl: ExtraTreesRegressor(400, min_samples_leaf=msl, max_features=0.3, n_jobs=8, random_state=0)), 'rich'))
CANDS.append(('histgb [rich]', lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=400, l2_regularization=1, random_state=0), 'rich'))
for h in [(64,), (128, 64)]:
    CANDS.append((f'mlp {h} [means]', (lambda h=h: make_pipeline(StandardScaler(), MLPRegressor(h, alpha=1e-2, max_iter=2000, early_stopping=True, random_state=0))), 'means'))
res = []; t0 = time.time()
for name, make, fs in CANDS:
    for hl in [None, 60, 30]:
        if hl is not None and not (name.startswith('ridge') or name.startswith('lgbm') or name.startswith('catboost')): continue
        try: m, oof = cv(make, SETS[fs], hl)
        except Exception as e: print('fail', name, e); continue
        res.append(dict(name=name + (f' hl={hl}d' if hl else ''), base=name, fs=fs, hl=hl, cv=m, make=make, oof=oof))
print(f'{len(res)} configs in {time.time()-t0:.0f}s', flush=True)
R = pd.DataFrame(res).sort_values('cv')
# best per family
R['family'] = R.base.str.split().str[0]
best = R.groupby('family').head(1)
yt, mt = y[te], ~interp[X.index[te]].values
def score(p):
    e = (p - yt.values); return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum(), MAE_nointerp=np.abs(e)[mt].mean())
out, preds = [], {}
for _, r in best.iterrows():
    p = test(r.make, SETS[r.fs], r.hl); preds[r['name']] = p; out.append(dict(model=r['name'], cv_MAE=r.cv, **score(p.values)))
# stacking: average of top-k families chosen by CV, and non-negative linear stack on OOF
top = best.head(5)
for k in [3, 5]:
    p = pd.concat([preds[n] for n in top.name.head(k)], axis=1).mean(axis=1)
    oofm = pd.concat([top.oof.iloc[i] for i in range(k)], axis=1).mean(axis=1).dropna()
    cvk = np.abs(oofm - y[oofm.index])[~interp[oofm.index].values].mean()
    out.append(dict(model=f'mean of top-{k} families by CV: ' + ', '.join(top.name.head(k)), cv_MAE=cvk, **score(p.values)))
O = pd.DataFrame(out).sort_values('cv_MAE'); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print('\n=== Setting A: per-hour (same information as the LLM). Ranked by TRAIN CV MAE; test = Aug-Sep 2017, 960 h ==='); print(O.round(3).to_string(index=False))
O.to_csv('results/baselines_v2_settingA.csv', index=False)
# ---- Setting B reference: forecasting with previous-hour labs ----
XB = pd.concat([X, LAG], axis=1); lagc = list(LAG.columns)
outB = []
p = LAG.loc[te, 'silica_lag1'].fillna(y[tr].mean()); outB.append(dict(model='persistence (previous hour silica)', **score(p.values)))
for nm, make, cols in [('ridge [means + lags]', lambda: make_pipeline(StandardScaler(), Ridge(10)), cols_mean + lagc),
                       ('lgbm [rich + lags]', lambda: lgb.LGBMRegressor(num_leaves=8, learning_rate=0.03, n_estimators=500, min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.6, verbose=-1, random_state=0), cols_rich + lagc)]:
    Xf = XB.copy(); Xf[lagc] = Xf[lagc].fillna(y[tr].mean())
    m, _ = cv(make, cols, Xd=Xf); outB.append(dict(model=nm, cv_MAE=m, **score(test(make, cols, Xd=Xf).values)))
print('\n=== Setting B (reference only, NOT the LLM task): previous-hour lab values allowed ==='); print(pd.DataFrame(outB).round(3).to_string(index=False))
pd.DataFrame(outB).to_csv('results/baselines_v2_settingB.csv', index=False)
pd.DataFrame(preds).to_parquet('results/baselines_v2_test_preds.parquet')
