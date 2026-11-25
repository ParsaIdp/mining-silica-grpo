CONTEXT = """You are a process engineer at an iron ore plant. The data come from the reverse cationic flotation stage of an iron ore concentrator (column flotation, 7 columns): silica-bearing gangue is floated off with amine (collector), while starch depresses the iron oxides, which report to the concentrate.

You are given one hour of plant data, recorded every 20 seconds (~180 rows). Columns:
- % Iron Feed, % Silica Feed: grade of the ore pulp fed to flotation (lab assays, updated infrequently)
- Starch Flow (m3/h), Amina Flow (m3/h): reagent dosages
- Ore Pulp Flow (t/h), Ore Pulp pH, Ore Pulp Density (kg/cm3): feed pulp conditions
- Flotation Column 01-07 Air Flow (Nm3/h) and Level (mm)
- % Iron Concentrate: lab assay of the iron grade of the final concentrate for this hour

Task: estimate % Silica Concentrate, the lab-measured silica grade (wt% SiO2) of the final concentrate for this same hour."""

ANSWER = "Reason as needed, then end your reply with your final estimate as a single number in the form <answer>X.XX</answer>."

def no_harness(table_csv: str) -> str:
    return f"{CONTEXT}\n\nData (CSV):\n{table_csv}\n{ANSWER}"

def harness_task() -> str:
    return (f"{CONTEXT}\n\nThe hour's data are in the CSV file `hour.csv` in the current working directory. "
            "Python 3 with pandas and numpy is available. Inspect and analyse the data with shell commands as you see fit. "
            "There is no other data available. When finished, give your final estimate.")

# ---- compact per-hour summary prompt (for RL-scale rollouts) ----
import io as _io
import pandas as _pd
SUMMARY_INTRO = CONTEXT.replace("You are given one hour of plant data, recorded every 20 seconds (~180 rows). Columns:",
                                "You are given a summary of one hour of plant data (recorded every 20 seconds, ~180 readings). Columns:")
RANGE_HINT = "For reference, the concentrate silica grade at this plant typically lies between about 0.5 and 6 wt% SiO2."
def _fmt(x):
    return f"{x:.4g}"
def summarize(table_csv: str) -> str:
    df = _pd.read_csv(_io.StringIO(table_csv))
    const, lines = [], []
    for c in df.columns:
        s = df[c]
        if s.nunique() == 1:
            const.append(f"- {c}: {_fmt(s.iloc[0])}")
        else:
            third = max(len(s) // 3, 1)
            lines.append(f"| {c} | {_fmt(s.mean())} | {_fmt(s.std())} | {_fmt(s.min())} | {_fmt(s.max())} | "
                         f"{_fmt(s.iloc[:third].mean())} | {_fmt(s.iloc[-third:].mean())} |")
    out = "Lab assays and values constant over the hour:\n" + "\n".join(const) + "\n\n"
    out += ("Process sensors over the hour (first/last third = mean of the first and last 20 minutes):\n"
            "| column | mean | std | min | max | first third | last third |\n|---|---|---|---|---|---|---|\n" + "\n".join(lines))
    return out
def lag_text(lags):
    """lags: list of (hours_back, silica or None)."""
    if len(lags[0]) == 3:  # (k, silica, iron)
        parts = [f"{k} h earlier: " + ("not available" if v is None else f"% Silica Concentrate {v:.2f}, % Iron Concentrate {fe:.2f}")
                 for k, v, fe in lags]
        return "Lab assays of the concentrate in the preceding hours (" + "; ".join(parts) + ")."
    parts = [f"{k} h earlier: {'not available' if v is None else f'{v:.2f}'}" for k, v in lags]
    return "Lab-assayed % Silica Concentrate in the preceding hours (" + "; ".join(parts) + ")."
BRIEF = "Reason briefly and concretely from the given values (a few short steps, no long derivations). "
def summary_prompt(table_csv: str, hint: bool = False, lags=None, brief: bool = False) -> str:
    return f"{SUMMARY_INTRO}\n\n{summarize(table_csv)}\n\n" + (lag_text(lags) + "\n" if lags else "") + (RANGE_HINT + "\n" if hint else "") + \
        (BRIEF if brief else "Keep your reasoning concise. ") + ANSWER

def lag_lookup(hours_df, n=3, iron=None):
    """Map hour -> [(k, silica at hour-k or None)] (plus iron concentrate if `iron` dict given)."""
    import pandas as pd
    t = dict(zip(pd.to_datetime(hours_df.hour), hours_df.target))
    if iron is None:
        return lambda h: [(k, t.get(pd.Timestamp(h) - pd.Timedelta(hours=k))) for k in range(1, n + 1)]
    return lambda h: [(k, t.get(pd.Timestamp(h) - pd.Timedelta(hours=k)), iron.get(pd.Timestamp(h) - pd.Timedelta(hours=k)))
                      for k in range(1, n + 1)]
def iron_lookup(root):
    import pandas as pd
    f = pd.read_parquet(root / 'data/hour_features.parquet')
    return dict(zip(pd.to_datetime(f.hour), f['% Iron Concentrate|mean']))
