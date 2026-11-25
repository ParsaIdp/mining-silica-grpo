import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, eval_subset
from cc_run import run
ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['tools', 'notools'], required=True)
ap.add_argument('--n', type=int, default=200); ap.add_argument('--limit', type=int, default=10**9); ap.add_argument('--effort', default='high')
ap.add_argument('--out'); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
out = ROOT / (a.out or f'results/inkling2s_cc_{a.mode}_{a.effort}.jsonl'); out.parent.mkdir(exist_ok=True)
done = {json.loads(l)['hour'] for l in out.open() if json.loads(l).get('pred') is not None} if out.exists() else set()
todo = [r for r in eval_subset(a.n).itertuples() if str(r.hour) not in done][:a.limit]
import threading
from concurrent.futures import ThreadPoolExecutor
ap2 = a; lock = threading.Lock()
def job(r):
    rec = run(r, a.mode, ROOT / 'runs/cc', effort=a.effort)
    with lock:
        with out.open('a') as f: f.write(json.dumps(rec) + '\n')
    return rec
with ThreadPoolExecutor(a.workers) as ex:
  for rec in ex.map(job, todo):
    u = rec['usage'] or {}
    print(rec['hour'], rec['target'], rec['pred'], 'tools', rec['n_tool_calls'], 'msgs', rec['n_assistant_msgs'], 'in', u.get('input_tokens'), 'out', u.get('output_tokens'), rec['secs'], 'rc', rec['rc'], rec['is_error'], rec['stderr'][-200:], flush=True)
