"""Shared spend ledger with a hard cap (gpt-6-astra list prices, USD per 1M tokens)."""
import json, threading, pathlib, fcntl
ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'results/spend_ledger.jsonl'
PRICE = {'gpt-6-astra': dict(inp=10.0, cached=1.0, out=50.0)}
def cost(model, inp, cached, out):
    p = PRICE[model]; return ((inp - cached) * p['inp'] + cached * p['cached'] + out * p['out']) / 1e6
def spent():
    if not LEDGER.exists(): return 0.0
    return sum(json.loads(l)['usd'] for l in LEDGER.open())
def record(**kw):
    with LEDGER.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(kw) + '\n')
