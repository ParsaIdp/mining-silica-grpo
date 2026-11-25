"""Run one sample through Codex CLI (OpenRouter provider) inside a bwrap sandbox.
The sandbox sees only its own work dir (+ read-only system/python); network is shared for the API call.
mode=tools: hour.csv in work dir, shell allowed.  mode=notools: table in prompt, no data files, model told not to run commands."""
import subprocess, json, os, shutil, pathlib, time, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, load_env, parse_answer
from prompts import CONTEXT, ANSWER, no_harness
CODEX = os.environ.get('CODEX_BIN', 'codex')  # path to the codex binary
CONF = '''model = "{model}"
model_provider = "openrouter"
model_reasoning_effort = "{effort}"
approval_policy = "never"
[model_providers.openrouter]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
[model_providers.openrouter.auth]
command = "sh"
args = ["-c", "echo $OPENROUTER_API_KEY"]
'''
def tools_prompt():
    return (f"{CONTEXT}\n\nThe hour's data are in `hour.csv` in the current directory. `python3` with pandas/numpy is available. "
            f"Analyse the data with shell commands as you see fit; no other data exist.\n{ANSWER}")
def notools_prompt(table):
    return no_harness(table) + "\nDo not run any commands or use any tools; answer directly from the data above."
def run(row, mode, work_root, model='thinkingmachines/inkling-2-small:free', effort='high', timeout=1800):
    load_env()
    w = pathlib.Path(work_root) / f"{mode}_{str(row.hour).replace(' ', 'T').replace(':', '')}"
    if w.exists(): shutil.rmtree(w)
    (w / 'work').mkdir(parents=True); (w / 'home').mkdir()
    (w / 'home/config.toml').write_text(CONF.format(model=model, effort=effort))
    if mode == 'tools': (w / 'work/hour.csv').write_text(row.table); prompt = tools_prompt()
    else: prompt = notools_prompt(row.table)
    (w / 'prompt.txt').write_text(prompt)
    ro = ['/usr', '/lib', '/lib64', '/bin', '/etc/ssl', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/pki', os.environ.get('SANDBOX_PYTHON_ROOT', os.path.expanduser('~/miniconda'))]
    cmd = ['bwrap', '--die-with-parent', '--unshare-all', '--share-net', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    for p in ro:
        if os.path.exists(p): cmd += ['--ro-bind', p, p]
    cmd += ['--ro-bind', str(ROOT / '.venv'), '/venv', '--ro-bind', CODEX, '/opt/codex',
            '--bind', str(w / 'work'), '/work', '--bind', str(w / 'home'), '/codexhome', '--chdir', '/work',
            '--clearenv', '--setenv', 'PATH', '/venv/bin:/usr/bin:/bin', '--setenv', 'HOME', '/codexhome', '--setenv', 'CODEX_HOME', '/codexhome',
            '--setenv', 'OPENROUTER_API_KEY', os.environ['OPENROUTER_API_KEY'],
            '/opt/codex', 'exec', '--json', '--skip-git-repo-check', '--ephemeral', '--ignore-rules',
            '--dangerously-bypass-approvals-and-sandbox', '-o', '/codexhome/last.txt', '-']
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout); rc = p.returncode; out, err = p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = 'timeout', (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), ''
    (w / 'events.jsonl').write_text(out); (w / 'stderr.txt').write_text(err)
    last = (w / 'home/last.txt').read_text() if (w / 'home/last.txt').exists() else ''
    ev = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    n_cmd = sum(1 for e in ev if e.get('type') == 'item.completed' and e.get('item', {}).get('type') == 'command_execution')
    usage = [e.get('usage') for e in ev if e.get('type') == 'turn.completed']
    errs = [e for e in ev if e.get('type') in ('error', 'turn.failed')]
    return dict(hour=str(row.hour), target=row.target, mode=mode, pred=parse_answer(last), last=last, n_commands=n_cmd,
                usage=usage[-1] if usage else None, rc=rc, secs=round(time.time() - t0, 1), errors=errs[:3], dir=str(w))
