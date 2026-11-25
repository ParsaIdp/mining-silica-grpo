"""Build one sample per hour. Target = hourly % Silica Concentrate. Time split: test >= 2017-08-01."""
import pandas as pd, numpy as np, json
df = pd.read_csv('data/raw/MiningProcess_Flotation_Plant_Database.csv', decimal=',', parse_dates=['date'])
T = '% Silica Concentrate'
g = df.groupby('date')
interp = (g[T].nunique() > 1) | (g['% Iron Concentrate'].nunique() > 1)
rows = []
for ts, d in g:
    d = d.drop(columns=['date']).reset_index(drop=True)
    rows.append(dict(hour=ts, n_rows=len(d), target=float(d[T].mean()), interpolated=bool(interp[ts]),
                     table=d.drop(columns=[T]).to_csv(index=False)))
s = pd.DataFrame(rows)
s['split'] = np.where(s.hour >= '2017-08-01', 'test', 'train')
s.to_parquet('data/hours.parquet', index=False)
# hourly aggregate features for classical baselines
feat = df.drop(columns=[T]).groupby('date').agg(['mean', 'std', 'min', 'max', 'first', 'last'])
feat.columns = [f'{a}|{b}' for a, b in feat.columns]
feat = feat.reset_index().rename(columns={'date': 'hour'})
feat.to_parquet('data/hour_features.parquet', index=False)
print(s.groupby('split').agg(n=('target', 'size'), interp=('interpolated', 'sum'), mean=('target', 'mean'), std=('target', 'std')))
