"""Evaluate a GRPO checkpoint (or the base model) with vLLM, using the training protocol.

The LoRA is merged into a full checkpoint (VL key layout). Sampling uses temperature T
without top-p/top-k, like the training policy. A response that hits --budget gets the
forced suffix "Final estimate: <answer>" and up to 10 more tokens.
Writes per-sample JSONL and prints MAE (mean over samples and median-of-n per hour).
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import eval_subset, parse_answer  # noqa: E402
from prompts import summary_prompt, lag_lookup, iron_lookup  # noqa: E402
BRIEF = False
LAGS = None  # set by --lags N: previous N hours' lab silica in the prompt

BASE = os.environ.get('QWEN_PATH', 'models/Qwen3.5-4B')
MERGED = Path(os.environ.get('MERGED_DIR', str(ROOT / 'runs/merged')))
FORCE_TEXT = '\n\nFinal estimate: <answer>'
THINK = False  # --thinking 1: Qwen thinking mode; forced suffix closes the think block
FORCE_TEXT_THINK = '\n</think>\n\nFinal estimate: <answer>'


def merge(ckpt, alpha_over_r):
    import torch
    from safetensors.torch import load_file, save_file
    dst = MERGED / (Path(ckpt).parent.name + '_' + Path(ckpt).name)
    if (dst / 'DONE').exists():
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    lora = torch.load(Path(ckpt) / 'state.pt', weights_only=False)['lora']
    pairs = {}
    for n, v in lora.items():
        key = n.replace('base_model.model.model.', 'model.language_model.')
        key = key.replace('.lora_A.default.weight', '.weight').replace('.lora_B.default.weight', '.weight')
        pairs.setdefault(key, {})['A' if 'lora_A' in n else 'B'] = v.float()
    used = 0
    for f in sorted(Path(BASE).glob('model.safetensors-*.safetensors')):
        sd = load_file(str(f))
        for k in list(sd):
            if k in pairs:
                ab = pairs[k]
                sd[k] = (sd[k].float() + alpha_over_r * ab['B'] @ ab['A']).to(sd[k].dtype)
                used += 1
        save_file(sd, str(dst / f.name), metadata={'format': 'pt'})
    if used != len(pairs):
        raise RuntimeError(f'merged {used} of {len(pairs)} LoRA modules')
    for f in [f for f in Path(BASE).iterdir() if f.is_file()]:
        if not f.name.startswith('model.safetensors-') or f.name.endswith('index.json'):
            if not (dst / f.name).exists():
                shutil.copy(f, dst / f.name)
    (dst / 'DONE').write_text('ok')
    return str(dst)


def val_rows():
    import pandas as pd
    d = pd.read_parquet(ROOT / 'data/hours.parquet')
    d = d[~d.interpolated]
    return d[(d.hour >= '2017-07-15') & (d.split == 'train')].sample(100, random_state=0).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='', help='runs/<run>/ckpt_XXXXX; empty = base model')
    ap.add_argument('--split', default='val', choices=['val', 'test', 'astra20'])
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--budget', type=int, default=384)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--alpha_over_r', type=float, default=2.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lags', type=int, default=0)
    ap.add_argument('--lag_iron', type=int, default=0)
    ap.add_argument('--brief', type=int, default=0)
    ap.add_argument('--thinking', type=int, default=0)
    a = ap.parse_args()
    global LAGS, BRIEF, THINK, FORCE_TEXT
    BRIEF = bool(a.brief)
    THINK = bool(a.thinking)
    if THINK:
        FORCE_TEXT = FORCE_TEXT_THINK
    if a.lags:
        import pandas as pd
        LAGS = lag_lookup(pd.read_parquet(ROOT / 'data/hours.parquet'), a.lags, iron_lookup(ROOT) if a.lag_iron else None)
    model = merge(a.ckpt, a.alpha_over_r) if a.ckpt else BASE
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    if a.split == 'astra20':
        import pandas as pd
        hrs = {json.loads(l)['hour'] for l in (ROOT / 'results/astra_notools_high.jsonl').open() if json.loads(l).get('pred') is not None}
        t = eval_subset(200); rows = [r for r in t.itertuples() if str(r.hour) in hrs]
    else:
        rows = list((val_rows() if a.split == 'val' else eval_subset(a.n)).itertuples())
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': summary_prompt(r.table, hint=True, lags=LAGS(r.hour) if LAGS else None, brief=BRIEF)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=THINK) for r in rows]
    llm = LLM(model, tokenizer=BASE, max_model_len=max(4096, 2200 + a.budget + 64), gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(n=a.samples, temperature=a.temperature, top_p=1.0, top_k=-1, max_tokens=a.budget, seed=0)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    recs, follow = [], []
    for r, p, o in zip(rows, prompts, outs):
        for i, c in enumerate(o.outputs):
            rec = dict(hour=str(r.hour), target=float(r.target), sample=i, text=c.text, n_tokens=len(c.token_ids), forced=c.finish_reason == "length")
            recs.append(rec)
            if rec['forced']:
                follow.append((rec, p + c.text + FORCE_TEXT))
    if follow:
        sp2 = SamplingParams(n=1, temperature=a.temperature, top_p=1.0, top_k=-1, max_tokens=10, stop=['</answer>'], seed=0)
        outs2 = llm.generate([f[1] for f in follow], sp2)
        for (rec, _), o in zip(follow, outs2):
            rec['text'] += FORCE_TEXT + o.outputs[0].text + '</answer>'
    for rec in recs:
        rec['pred'] = parse_answer(rec['text'])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for rec in recs:
            f.write(json.dumps(rec) + '\n')
    import numpy as np, pandas as pd
    d = pd.DataFrame(recs)
    ok = d.dropna(subset=['pred'])
    per_hour = ok.groupby('hour').agg(target=('target', 'first'), med=('pred', 'median'))
    summ = dict(ckpt=a.ckpt or 'base', split=a.split, lags=a.lags, hours=len(rows), parsed=len(ok) / len(d), forced=float(d.forced.mean()),
                mae_samples=float((ok.pred - ok.target).abs().mean()), mae_median=float((per_hour.med - per_hour.target).abs().mean()),
                corr_median=float(np.corrcoef(per_hour.med, per_hour.target)[0, 1]), secs=time.time() - t0)
    print(json.dumps(summ), flush=True)
    with (ROOT / 'results/rl_evals.jsonl').open('a') as f:
        f.write(json.dumps(summ) + '\n')
    if a.ckpt and os.environ.get('KEEP_MERGED') != '1':
        shutil.rmtree(model, ignore_errors=True)


if __name__ == '__main__':
    main()
