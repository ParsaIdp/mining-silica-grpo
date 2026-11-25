"""No-tool eval: full 20 s table in the prompt; resumable JSONL output."""
import argparse, json, time, sys
from openai import OpenAI
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import load_env, parse_answer, eval_subset, ROOT
from prompts import no_harness
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=200); ap.add_argument('--limit', type=int, default=10**9)
ap.add_argument('--model', default='thinkingmachines/inkling-2-small:free'); ap.add_argument('--effort', default='high')
ap.add_argument('--out', default='results/inkling2s_noharness.jsonl'); a = ap.parse_args()
load_env(); import os
cli = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'])
out = ROOT / a.out; done = set()
if out.exists(): done = {json.loads(l)['hour'] for l in out.open() if json.loads(l).get('pred') is not None}
todo = [r for r in eval_subset(a.n).itertuples() if str(r.hour) not in done][:a.limit]
for r in todo:
    t0 = time.time()
    try:
        resp = cli.chat.completions.create(model=a.model, messages=[{'role': 'user', 'content': no_harness(r.table)}],
                                           temperature=1.0, max_tokens=32000, extra_body={'reasoning': {'effort': a.effort}})
        msg = resp.choices[0].message; text = msg.content or ''
        reasoning = getattr(msg, 'reasoning', None) or ''
        rec = dict(hour=str(r.hour), target=r.target, pred=parse_answer(text), content=text, reasoning=reasoning,
                   usage=resp.usage.model_dump() if resp.usage else None, secs=round(time.time() - t0, 1), model=a.model, effort=a.effort)
    except Exception as e:
        rec = dict(hour=str(r.hour), target=r.target, pred=None, error=repr(e)[:500])
    with out.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(rec['hour'], rec['target'], rec.get('pred'), rec.get('usage', {}) and rec['usage'].get('prompt_tokens'), rec.get('usage', {}) and rec['usage'].get('completion_tokens'), rec.get('secs'), rec.get('error', ''), flush=True)
