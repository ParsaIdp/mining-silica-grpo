import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, eval_subset
from codex_run import run
ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['tools', 'notools'], required=True)
ap.add_argument('--n', type=int, default=200); ap.add_argument('--limit', type=int, default=10**9); ap.add_argument('--effort', default='high')
ap.add_argument('--out'); a = ap.parse_args()
out = ROOT / (a.out or f'results/inkling2s_codex_{a.mode}.jsonl'); out.parent.mkdir(exist_ok=True)
done = {json.loads(l)['hour'] for l in out.open() if json.loads(l).get('pred') is not None} if out.exists() else set()
todo = [r for r in eval_subset(a.n).itertuples() if str(r.hour) not in done][:a.limit]
for r in todo:
    rec = run(r, a.mode, ROOT / 'runs/codex', effort=a.effort)
    with out.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(rec['hour'], rec['target'], rec['pred'], 'cmds', rec['n_commands'], rec['usage'], rec['secs'], rec['rc'], rec['errors'][:1], flush=True)
