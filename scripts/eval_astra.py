"""gpt-6-astra eval. notools: Responses API, table in prompt. tools: Codex CLI (OpenAI provider) in bwrap with hour.csv.
Stops before starting a sample once ledger spend >= --cap."""
import argparse, json, sys, pathlib, time, threading, os, shutil, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, eval_subset, load_env, parse_answer
from prompts import no_harness, summary_prompt, lag_lookup, iron_lookup
from budget import cost, spent, record
from codex_run import tools_prompt, CODEX
from concurrent.futures import ThreadPoolExecutor
MODEL = 'gpt-6-astra'
CONF = f'''model = "{MODEL}"
model_provider = "oa"
model_reasoning_effort = "{{effort}}"
approval_policy = "never"
[model_providers.oa]
name = "OpenAI API"
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
'''
PROMPT = lambda r: no_harness(r.table)
def notools(r, effort):
    from openai import OpenAI
    t0 = time.time()
    resp = OpenAI().responses.create(model=MODEL, input=PROMPT(r), reasoning={'effort': effort, 'summary': 'auto'}, max_output_tokens=64000)
    u = resp.usage; c = cost(MODEL, u.input_tokens, u.input_tokens_details.cached_tokens, u.output_tokens)
    summ = [s.text for o in resp.output if o.type == 'reasoning' for s in (o.summary or [])]
    return dict(pred=parse_answer(resp.output_text), last=resp.output_text, reasoning_summary=summ, usd=c, secs=round(time.time() - t0, 1),
                usage=dict(input_tokens=u.input_tokens, cached=u.input_tokens_details.cached_tokens, output_tokens=u.output_tokens, reasoning=u.output_tokens_details.reasoning_tokens))
def tools(r, effort, work_root):
    w = pathlib.Path(work_root) / f"astra_tools_{str(r.hour).replace(' ', 'T').replace(':', '')}"
    if w.exists(): shutil.rmtree(w)
    (w / 'work').mkdir(parents=True); (w / 'home').mkdir()
    (w / 'home/config.toml').write_text(CONF.format(effort=effort)); (w / 'work/hour.csv').write_text(r.table)
    prompt = tools_prompt(); (w / 'prompt.txt').write_text(prompt)
    ro = ['/usr', '/lib', '/lib64', '/bin', '/etc/ssl', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/pki', os.environ.get('SANDBOX_PYTHON_ROOT', os.path.expanduser('~/miniconda'))]
    cmd = ['bwrap', '--die-with-parent', '--unshare-all', '--share-net', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    for p in ro:
        if os.path.exists(p): cmd += ['--ro-bind', p, p]
    cmd += ['--ro-bind', str(ROOT / '.venv'), '/venv', '--ro-bind', str(pathlib.Path(CODEX).parents[1]), '/opt/codexvendor', '--bind', str(w / 'work'), '/work',
            '--bind', str(w / 'home'), '/codexhome', '--chdir', '/work', '--clearenv', '--setenv', 'PATH', '/venv/bin:/usr/bin:/bin',
            '--setenv', 'HOME', '/codexhome', '--setenv', 'CODEX_HOME', '/codexhome', '--setenv', 'OPENAI_API_KEY', os.environ['OPENAI_API_KEY'],
            '/opt/codexvendor/bin/codex', 'exec', '--json', '--skip-git-repo-check', '--ephemeral', '--ignore-rules', '--dangerously-bypass-approvals-and-sandbox',
            '-o', '/codexhome/last.txt', '-']
    t0 = time.time()
    try: p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=2400); out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e: out, err, rc = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), '', 'timeout'
    (w / 'events.jsonl').write_text(out); (w / 'stderr.txt').write_text(err)
    ev = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    us = [e['usage'] for e in ev if e.get('type') == 'turn.completed' and e.get('usage')]
    inp = sum(u.get('input_tokens', 0) for u in us); cached = sum(u.get('cached_input_tokens', 0) for u in us); outt = sum(u.get('output_tokens', 0) for u in us)
    last = (w / 'home/last.txt').read_text() if (w / 'home/last.txt').exists() else ''
    ncmd = sum(1 for e in ev if e.get('type') == 'item.completed' and e.get('item', {}).get('type') == 'command_execution')
    errs = [e for e in ev if e.get('type') in ('error', 'turn.failed')][:3]
    return dict(pred=parse_answer(last), last=last[-3000:], n_commands=ncmd, usd=cost(MODEL, inp, cached, outt), rc=rc, errors=errs,
                usage=dict(input_tokens=inp, cached=cached, output_tokens=outt), secs=round(time.time() - t0, 1), dir=str(w))
ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['tools', 'notools'], required=True); ap.add_argument('--limit', type=int, default=10)
ap.add_argument('--n', type=int, default=200); ap.add_argument('--effort', default='high'); ap.add_argument('--workers', type=int, default=3)
ap.add_argument('--cap', type=float, default=40.0)
ap.add_argument('--prompt', default='raw', choices=['raw', 'summary_hint']); ap.add_argument('--split', default='test', choices=['test', 'val'])
ap.add_argument('--lags', type=int, default=0); ap.add_argument('--lag_iron', type=int, default=0); ap.add_argument('--brief', type=int, default=0); a = ap.parse_args(); load_env()
tag = '' if (a.prompt, a.split, a.lags) == ('raw', 'test', 0) else f'_{a.prompt}_{a.split}' + (f'_lag{a.lags}' if a.lags else '') + ('fe' if a.lag_iron else '') + ('_brief' if a.brief else '')
out = ROOT / f'results/astra_{a.mode}_{a.effort}{tag}.jsonl'
if a.prompt == 'summary_hint':
    import pandas as pd
    L = lag_lookup(pd.read_parquet(ROOT / 'data/hours.parquet'), a.lags, iron_lookup(ROOT) if a.lag_iron else None) if a.lags else None
    PROMPT = lambda r: summary_prompt(r.table, hint=True, lags=L(r.hour) if L else None, brief=bool(a.brief))
if a.split == 'val':
    sys.path.insert(0, str(ROOT / 'rl')); from eval_ckpt import val_rows
    eval_subset = lambda n: val_rows()
done = {json.loads(l)['hour'] for l in out.open() if json.loads(l).get('pred') is not None} if out.exists() else set()
todo = [r for r in eval_subset(a.n).itertuples() if str(r.hour) not in done][:a.limit]
lock = threading.Lock()
def job(r):
    if spent() >= a.cap: print('budget cap reached, skipping', r.hour, flush=True); return
    try: rec = notools(r, a.effort) if a.mode == 'notools' else tools(r, a.effort, ROOT / 'runs/astra')
    except Exception as e: rec = dict(pred=None, error=repr(e)[:500], usd=0.0)
    rec = dict(hour=str(r.hour), target=r.target, mode=a.mode, effort=a.effort, model=MODEL, **rec)
    record(usd=rec['usd'], hour=rec['hour'], mode=a.mode, effort=a.effort)
    with lock:
        with out.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(rec['hour'], round(rec['target'], 2), rec['pred'], f"${rec['usd']:.2f}", rec.get('usage'), rec.get('secs'), rec.get('n_commands', ''), rec.get('error', '') or rec.get('errors', ''), f'| total ${spent():.2f}', flush=True)
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(job, todo))
