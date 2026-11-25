"""Local zero-shot eval of Qwen3.5-4B with vLLM: same no-tools prompt, n samples per hour."""
import argparse, json, os, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, eval_subset, parse_answer
from prompts import no_harness, summary_prompt
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default=os.environ.get('QWEN_PATH', 'models/Qwen3.5-4B'))
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--samples', type=int, default=4); ap.add_argument('--max_tokens', type=int, default=16384)
    ap.add_argument('--thinking', type=int, default=1); ap.add_argument('--out', default='results/qwen35_4b_think.jsonl'); ap.add_argument('--prompt', default='raw', choices=['raw', 'summary', 'summary_hint']); a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = list(eval_subset(a.n).itertuples())
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': no_harness(r.table) if a.prompt == 'raw' else summary_prompt(r.table, hint=a.prompt == 'summary_hint')}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=bool(a.thinking)) for r in rows]
    plen = [len(tok(p).input_ids) for p in prompts]; print('prompt tokens min/mean/max', min(plen), sum(plen) / len(plen), max(plen), flush=True)
    llm = LLM(a.model, max_model_len=max(plen) + a.max_tokens + 64, gpu_memory_utilization=0.9, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(n=a.samples, temperature=0.6 if a.thinking else 0.7, top_p=0.95 if a.thinking else 0.8, top_k=20, max_tokens=a.max_tokens, seed=0)
    out = ROOT / a.out
    done = {json.loads(l)['hour'] for l in out.open()} if out.exists() else set()
    todo = [(r, p) for r, p in zip(rows, prompts) if str(r.hour) not in done]
    print('todo hours', len(todo), flush=True)
    for k in range(0, len(todo), 25):
        chunk = todo[k:k + 25]; t0 = time.time()
        outs = llm.generate([p for _, p in chunk], sp)
        with out.open('a') as f:
            for (r, _), o in zip(chunk, outs):
                for i, c in enumerate(o.outputs):
                    ans = c.text.split('</think>')[-1]
                    f.write(json.dumps(dict(hour=str(r.hour), target=r.target, sample=i, pred=parse_answer(ans), n_tokens=len(c.token_ids),
                                            finish=c.finish_reason, text=c.text)) + '\n')
        print(f'chunk {k // 25} done {len(chunk)} hours in {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    main()
