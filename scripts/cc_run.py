"""Run one sample of Inkling 2 Small through Claude Code (headless) on OpenRouter, inside a bwrap sandbox.
The sandbox exposes only its own work dir plus read-only system/python; network is shared for the API.
mode=notools: full table in prompt, no tools at all (--tools "").
mode=tools:   hour.csv in work dir; Bash/Read/Write/Edit/Glob/Grep allowed; python3+pandas available."""
import subprocess, json, os, shutil, pathlib, time, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, load_env, parse_answer
from prompts import CONTEXT, ANSWER, no_harness
CLAUDE = os.path.realpath(os.path.expanduser('~/.local/bin/claude'))
MODEL = 'thinkingmachines/inkling-2-small:free'
def tools_prompt():
    return (f"{CONTEXT}\n\nThe hour's data are in `hour.csv` in the current directory. `python3` with pandas/numpy is available. "
            f"Analyse the data with shell commands as you see fit; no other data exist.\n{ANSWER}")
def run(row, mode, work_root, effort='high', timeout=2400, model=MODEL):
    load_env()
    w = pathlib.Path(work_root) / f"{mode}_{str(row.hour).replace(' ', 'T').replace(':', '')}"
    if w.exists(): shutil.rmtree(w)
    (w / 'work').mkdir(parents=True); (w / 'cfg').mkdir()
    if mode == 'tools':
        (w / 'work/hour.csv').write_text(row.table); prompt = tools_prompt(); tools = 'Bash,Read,Write,Edit,Glob,Grep'
    else:
        prompt = no_harness(row.table); tools = ''
    (w / 'prompt.txt').write_text(prompt)
    ro = ['/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc/ssl', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/pki', '/etc/passwd', os.environ.get('SANDBOX_PYTHON_ROOT', os.path.expanduser('~/miniconda'))]
    cmd = ['bwrap', '--die-with-parent', '--unshare-all', '--share-net', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    for p in ro:
        if os.path.exists(p): cmd += ['--ro-bind', p, p]
    env = {'PATH': '/venv/bin:/usr/bin:/bin', 'HOME': '/cfg', 'CLAUDE_CONFIG_DIR': '/cfg', 'ANTHROPIC_BASE_URL': 'https://openrouter.ai/api',
           'ANTHROPIC_AUTH_TOKEN': os.environ['OPENROUTER_API_KEY'], 'ANTHROPIC_API_KEY': '', 'ANTHROPIC_MODEL': model,
           'ANTHROPIC_SMALL_FAST_MODEL': model, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': model, 'DISABLE_TELEMETRY': '1',
           'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_AUTOUPDATER': '1', 'TERM': 'dumb'}
    cmd += ['--ro-bind', str(ROOT / '.venv'), '/venv', '--ro-bind', CLAUDE, '/opt/claude',
            '--bind', str(w / 'work'), '/work', '--bind', str(w / 'cfg'), '/cfg', '--chdir', '/work', '--clearenv']
    for k, v in env.items(): cmd += ['--setenv', k, v]
    cmd += ['/opt/claude', '-p', '--bare', '--model', model, '--effort', effort, '--tools', tools,
            '--permission-mode', 'bypassPermissions', '--no-session-persistence', '--output-format', 'stream-json', '--verbose']
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout); rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = 'timeout', e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), ''
    (w / 'events.jsonl').write_text(out); (w / 'stderr.txt').write_text(err)
    ev = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    res = next((e for e in reversed(ev) if e.get('type') == 'result'), {})
    n_tool = sum(1 for e in ev if e.get('type') == 'assistant' for c in e['message'].get('content', []) if c.get('type') == 'tool_use')
    n_api = sum(1 for e in ev if e.get('type') == 'assistant')
    last = res.get('result', '') or ''
    return dict(hour=str(row.hour), target=row.target, mode=mode, effort=effort, pred=parse_answer(last), last=last[-3000:],
                n_tool_calls=n_tool, n_assistant_msgs=n_api, usage=res.get('usage'), num_turns=res.get('num_turns'),
                is_error=res.get('is_error'), rc=rc, secs=round(time.time() - t0, 1), stderr=err[-500:], dir=str(w))
