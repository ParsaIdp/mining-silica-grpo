import re, json, os, time, pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
def load_env():
    f = ROOT / '.env'
    if not f.exists():
        return
    for l in f.read_text().splitlines():
        if '=' in l: k, v = l.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
def parse_answer(text):
    m = re.findall(r'<answer>\s*([-+]?\d*\.?\d+)\s*</answer>', text or '')
    return float(m[-1]) if m else None
def eval_subset(n=200, seed=0):
    s = pd.read_parquet(ROOT / 'data/hours.parquet')
    t = s[(s.split == 'test') & ~s.interpolated]
    return t.sample(n, random_state=seed).reset_index(drop=True)  # random order so partial runs are unbiased
