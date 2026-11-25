import pandas as pd, numpy as np
df = pd.read_csv('data/raw/MiningProcess_Flotation_Plant_Database.csv', decimal=',', parse_dates=['date'])
print(df.shape, df.date.min(), df.date.max())
g = df.groupby('date')
n = g.size(); print('hours', len(n)); print(n.value_counts().head())
full = pd.date_range(df.date.min(), df.date.max(), freq='h'); missing = full.difference(n.index)
print('missing hours', len(missing)); 
gaps = pd.Series(missing).diff().ne(pd.Timedelta('1h')).cumsum(); print(pd.Series(missing).groupby(gaps).agg(['min','max','count']))
# constant-within-hour columns
nun = g.nunique().max(); print('max nunique within hour per col:\n', nun)
h = g.mean()
y = h['% Silica Concentrate']; print(y.describe())
print('corr w/ iron conc', y.corr(h['% Iron Concentrate']))
print('lag1 autocorr', y.autocorr(1))
# fraction of hours where silica differs within hour
print('hours with >1 silica value', (g['% Silica Concentrate'].nunique()>1).mean())
print('hours with >1 iron feed', (g['% Iron Feed'].nunique()>1).mean())
# runs of constant feed
fe = h['% Iron Feed']; print('feed changes per hour frac', (fe.diff()!=0).mean())
print(h.index.to_series().dt.to_period('M').value_counts().sort_index())
