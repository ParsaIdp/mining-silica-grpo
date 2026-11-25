"""GRPO on Qwen3.5-4B with bitwise-identical sampling and training forwards.

Sampling and training run the same HF Qwen3.5 model, ops, shapes, and batch
composition. That is the "shared cached" design from BioReasonCell
(bioreason_cell/rl/exact_policy), extended here to a lockstep group of G
rollouts of one prompt.
  * Sampling: autograd-enabled forward, backward-only saves discarded; every
    step's full [G, vocab] logits and log-probs are hashed.
  * Replay: the same prefill + per-token cached decode with gradients, checkpointed
    in blocks. Every step re-checks the full-vocab hashes (also during backward
    recomputation); any mismatch aborts before the optimizer step.
Rows that finish keep feeding a pad token, so shapes never change.
When a row hits the reasoning budget, a fixed suffix "<answer>" is teacher-forced
(loss-masked), and the row then samples up to ANSWER_TOKENS tokens of the number.
The policy is softmax(logits / T); no top-p/top-k.
Data parallel: one process per GPU (torchrun). LoRA grads are all-reduced, and
weights are checked to be identical across ranks after each step.
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, os.environ.get('BIOREASONCELL', str(ROOT.parent / 'BioReasonCell')))
from common import parse_answer  # noqa: E402
from prompts import summary_prompt, lag_lookup, iron_lookup  # noqa: E402
BRIEF = False
LAGS = None  # set by --lags N: previous N hours' lab silica in the prompt
from bioreason_cell.rl.exact_policy.functional_cell_cache import FunctionalCellCache, install_functional_decode  # noqa: E402
from bioreason_cell.rl.exact_policy.checkpointed_cell_replay import cache_after_block  # noqa: E402
from bioreason_cell.rl.exact_policy.policy_identity import tensor_digest  # noqa: E402
from bioreason_cell.rl.exact_policy.sampling_memory import sampling_saved_tensors  # noqa: E402

MODEL = os.environ.get('QWEN_PATH', 'models/Qwen3.5-4B')  # local Qwen3.5-4B checkpoint dir
FORCE_TEXT = '\n\nFinal estimate: <answer>'
THINK = False  # --thinking 1: Qwen thinking mode; forced suffix closes the think block
FORCE_TEXT_THINK = '\n</think>\n\nFinal estimate: <answer>'
ANSWER_TOKENS = 10


# ---------------------------------------------------------------- model
def load_model(path, lora_rank, lora_alpha, device):
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model
    cfg = AutoConfig.from_pretrained(path).text_config
    cfg._attn_implementation = 'flash_attention_2'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(cfg)
    state = {}
    for f in sorted(Path(path).glob('model.safetensors-*.safetensors')):
        for k, v in load_file(str(f)).items():
            if k.startswith('model.language_model.'):
                state['model.' + k[len('model.language_model.'):]] = v
            elif k.startswith('lm_head.'):
                state[k] = v
    if 'lm_head.weight' not in state:
        state['lm_head.weight'] = state['model.embed_tokens.weight']
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [m for m in missing if 'rotary' not in m]
    if missing or unexpected:
        raise RuntimeError(f'weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    model.tie_weights()
    # Non-persistent buffers (rotary inv_freq) are still on meta; rebuild them.
    for name, m in list(model.named_modules()):
        if any(b.is_meta for b in m.buffers(recurse=False)):
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            setattr(parent, name.rsplit('.', 1)[-1], type(m)(config=m.config, device='cpu'))
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f'meta tensors left: {left[:5]}')
    model = model.to(device=device, dtype=torch.bfloat16)
    lcfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.train()
    if any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
        raise ValueError('dropout would break replay identity')
    base = model.get_base_model()
    install_functional_decode(base)
    return model, base


class GroupForward:
    """Cached forward over one prompt and a lockstep group of rollouts."""

    def __init__(self, base, temperature):
        self.base, self.T = base, temperature

    def logprobs(self, logits):
        return (logits.float() / self.T).log_softmax(-1)

    def prefill(self, prompt_ids, cache):
        ids = torch.tensor([prompt_ids], device='cuda')
        pos = torch.arange(len(prompt_ids), device='cuda').unsqueeze(0)
        out = self.base(input_ids=ids, position_ids=pos, attention_mask=None, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        return out.logits[:, -1, :]  # [1, V]

    def step(self, tokens, position, cache):
        ids = tokens.view(-1, 1)
        pos = torch.full_like(ids, position)
        out = self.base(input_ids=ids, position_ids=pos, attention_mask=None, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        return out.logits[:, -1, :]  # [G, V]


# ---------------------------------------------------------------- sampling
@torch.enable_grad()
def sample_group(fwd, prompt_ids, G, budget, eos, pad, force_ids, answer_close, decode, generator):
    """Returns a trajectory dict: inputs[t] = tokens fed at decode step t (t>=1), masks, lps, hashes."""
    cache = FunctionalCellCache(fwd.base.config)
    with sampling_saved_tensors():
        logits = fwd.prefill(prompt_ids, cache)
    logits = logits.detach()
    lp = fwd.logprobs(logits)
    hashes = [tensor_digest(logits) + tensor_digest(lp)]
    first = torch.multinomial(lp.exp().expand(G, -1), 1, replacement=True, generator=generator).view(-1)
    cache = cache.fork(detach=True).repeat_single_prefix(G)
    cache = cache.fork(detach=True)
    tokens = [[int(t)] for t in first.tolist()]
    masks = [[1] for _ in range(G)]
    lps = [[float(lp[0, t])] for t in first.tolist()]
    sampled = [1] * G
    mode = ['sample'] * G
    queue = [[] for _ in range(G)]
    answer_count = [0] * G
    forced = [False] * G

    def transition(i, tok, was_sampled):
        if mode[i] == 'sample':
            if tok in eos:
                mode[i] = 'done'
            elif sampled[i] >= budget:
                mode[i], queue[i], forced[i] = 'force', list(force_ids), True
        elif mode[i] == 'force':
            if not queue[i]:
                mode[i] = 'answer'
        elif mode[i] == 'answer':
            answer_count[i] += 1
            if tok in eos or answer_close in decode(tokens[i][-4:]) or answer_count[i] >= ANSWER_TOKENS:
                mode[i] = 'done'

    for i in range(G):
        transition(i, tokens[i][0], True)
    inputs = [first.tolist()]
    pos = len(prompt_ids)
    while any(m != 'done' for m in mode):
        feed = torch.tensor(inputs[-1], device='cuda')
        with sampling_saved_tensors():
            logits = fwd.step(feed, pos, cache)
        logits = logits.detach()
        cache = cache.fork(detach=True)
        lp = fwd.logprobs(logits)
        hashes.append(tensor_digest(logits) + tensor_digest(lp))
        draw = torch.multinomial(lp.exp(), 1, generator=generator).view(-1).tolist()
        nxt = []
        for i in range(G):
            if mode[i] == 'done':
                t, m, l = pad, 0, 0.0
            elif mode[i] == 'force':
                t, m, l = queue[i].pop(0), 0, 0.0
            else:
                t, m, l = draw[i], 1, float(lp[i, draw[i]])
                sampled[i] += 1
            if mode[i] != 'done':
                tokens[i].append(t); masks[i].append(m); lps[i].append(l)
            nxt.append(t)
            if mode[i] != 'done':
                transition(i, t, m)
        inputs.append(nxt)
        pos += 1
    # Row-wise text and per-step alignment: row i's token at step s is inputs[s][i]
    # while the row is alive (s < len(tokens[i])).
    return dict(prompt=list(prompt_ids), inputs=inputs, hashes=hashes, tokens=tokens, masks=masks,
                lps=lps, forced=forced, texts=[decode(t) for t in tokens])


# ---------------------------------------------------------------- replay
def replay_group(fwd, traj, block_size):
    """Differentiable replay; returns lp tensor [S, G] where S = len(inputs) (step s gives lp of inputs[s])."""
    inputs, hashes, G = traj['inputs'], traj['hashes'], len(traj['tokens'])
    S = len(inputs)
    tok = torch.tensor(inputs, device='cuda')  # [S, G]

    def check(s, logits, lp):
        if tensor_digest(logits) + tensor_digest(lp) != hashes[s]:
            raise RuntimeError(f'full-vocab logits/logprobs differ from sampling at step {s}')

    def prefill_fn(*_unused):
        cache = FunctionalCellCache(fwd.base.config)
        logits = fwd.prefill(traj['prompt'], cache)
        lp = fwd.logprobs(logits)
        check(0, logits, lp)
        sel = lp[0].gather(0, tok[0])  # [G]
        _, _, tensors = cache.split_tensors()
        return (sel, *tensors)

    template0, names0, _ = FunctionalCellCache(fwd.base.config).split_tensors()
    dummy = torch.zeros(1, device='cuda', requires_grad=True)
    out = checkpoint(prefill_fn, dummy, use_reentrant=False, preserve_rng_state=False)
    sels = [out[0].unsqueeze(0)]
    cache = cache_after_block(template0, names0, out[1:], initial=True).repeat_single_prefix(G)
    pos0 = len(traj['prompt'])

    def make_block(template, names, start, stop):
        def run(*tensors):
            c = FunctionalCellCache.join_tensors(template, names, tensors)
            out = []
            for s in range(start, stop):
                logits = fwd.step(tok[s - 1], pos0 + s - 1, c)
                lp = fwd.logprobs(logits)
                check(s, logits, lp)
                out.append(lp.gather(1, tok[s].view(-1, 1)).view(-1))
            _, _, end = c.split_tensors()
            return (torch.stack(out), *end)
        return run

    with torch.autograd.graph.save_on_cpu(pin_memory=True), set_checkpoint_early_stop(False):
        for start in range(1, S, block_size):
            stop = min(start + block_size, S)
            template, names, tensors = cache.split_tensors()
            out = checkpoint(make_block(template, names, start, stop), *tensors,
                             use_reentrant=False, preserve_rng_state=False)
            sels.append(out[0])
            cache = FunctionalCellCache.join_tensors(template, names, out[1:])
    return torch.cat(sels)  # [S, G]


def row_masks(traj):
    S, G = len(traj['inputs']), len(traj['tokens'])
    mask = torch.zeros(S, G)
    old = torch.zeros(S, G)
    for i in range(G):
        n = len(traj['tokens'][i])
        mask[:n, i] = torch.tensor(traj['masks'][i], dtype=torch.float)
        old[:n, i] = torch.tensor(traj['lps'][i])
    return mask, old


# ---------------------------------------------------------------- reward
def reward_fn(text, target, forced, scale, forced_penalty=0.1):
    pred = parse_answer(text)
    if pred is None or not math.isfinite(pred):
        return -0.5, None
    return math.exp(-abs(pred - target) / scale) - (forced_penalty if forced else 0.0), pred


# ---------------------------------------------------------------- data
def build_rows(tok, split):
    import pandas as pd
    d = pd.read_parquet(ROOT / 'data/hours.parquet')
    d = d[~d.interpolated]
    if split == 'train':
        d = d[d.hour < '2017-07-15']
    elif split == 'val':
        d = d[(d.hour >= '2017-07-15') & (d.split == 'train')].sample(100, random_state=0)
    rows = []
    for r in d.itertuples():
        text = tok.apply_chat_template([{'role': 'user', 'content': summary_prompt(r.table, hint=True, lags=LAGS(r.hour) if LAGS else None, brief=BRIEF)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
        rows.append(dict(hour=str(r.hour), target=float(r.target), ids=tok(text, add_special_tokens=False).input_ids))
    return rows


# ---------------------------------------------------------------- checkpoint
def lora_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def params_digest(params):
    h = hashlib.sha256()
    for p in params:
        h.update(tensor_digest(p).encode())
    return h.hexdigest()


def save_ckpt(out, step, model, opt, gen, cursor, order):
    d = out / f'ckpt_{step:05d}'
    tmp = out / f'.tmp_ckpt_{step:05d}'
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=step, lora=lora_state(model), opt=opt.state_dict(), gen=gen.get_state(),
                    cpu_rng=torch.get_rng_state(), cursor=cursor, order=order), tmp / 'state.pt')
    if d.exists():
        import shutil; shutil.rmtree(d)
    tmp.rename(d)
    (out / 'latest').write_text(d.name)
    # keep every 10th plus the latest two
    ck = sorted(out.glob('ckpt_*'))
    for c in ck[:-2]:
        if int(c.name.split('_')[1]) % 10:
            import shutil; shutil.rmtree(c)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--prompts_per_rank', type=int, default=2)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--budget', type=int, default=384)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--lora_rank', type=int, default=32)
    ap.add_argument('--reward_scale', type=float, default=0.5)
    ap.add_argument('--block_size', type=int, default=32)
    ap.add_argument('--max_grad_norm', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lags', type=int, default=0)
    ap.add_argument('--lag_iron', type=int, default=0)
    ap.add_argument('--brief', type=int, default=0)
    ap.add_argument('--thinking', type=int, default=0)
    ap.add_argument('--forced_penalty', type=float, default=0.1)
    ap.add_argument('--adv_std', type=int, default=0, help='divide advantages by group reward std (standard GRPO)')
    ap.add_argument('--adv_eps', type=float, default=0.05)
    ap.add_argument('--deadline', type=float, default=0, help='unix time; stop before starting a step that may not finish')
    a = ap.parse_args()

    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC', '1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (out / 'args.json').write_text(json.dumps(vars(a), indent=1))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = {tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id} - {None}
    pad = tok.convert_tokens_to_ids('<|im_end|>')
    global LAGS, BRIEF, THINK, FORCE_TEXT
    BRIEF = bool(a.brief)
    THINK = bool(a.thinking)
    if THINK:
        FORCE_TEXT = FORCE_TEXT_THINK
    if a.lags:
        import pandas as pd
        LAGS = lag_lookup(pd.read_parquet(ROOT / 'data/hours.parquet'), a.lags, iron_lookup(ROOT) if a.lag_iron else None)
    force_ids = tok(FORCE_TEXT, add_special_tokens=False).input_ids
    rows = build_rows(tok, 'train')
    torch.manual_seed(a.seed)
    model, base = load_model(MODEL, a.lora_rank, 2 * a.lora_rank, 'cuda')
    params = [p for p in model.parameters() if p.requires_grad]
    if ddp:
        for p in params:
            dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    gen = torch.Generator(device='cuda')
    step, cursor = 0, 0
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    latest = out / 'latest'
    if latest.exists():
        st = torch.load(out / latest.read_text() / 'state.pt', weights_only=False)
        named = dict(model.named_parameters())
        for n, v in st['lora'].items():
            named[n].data.copy_(v.to(named[n].device))
        opt.load_state_dict(st['opt'])
        step, cursor, order = st['step'], st['cursor'], st['order']
        gen.set_state(st['gen'])
        torch.set_rng_state(st['cpu_rng'])
    else:
        gen.manual_seed(a.seed)
    # distinct sampling streams per rank, derived deterministically from the shared state
    rank_gen = torch.Generator(device='cuda')
    fwd = GroupForward(base, a.temperature)
    log = (out / 'train.jsonl').open('a') if rank == 0 else None
    step_secs = []
    while step < a.steps:
        if a.deadline and step_secs and time.time() + 1.3 * max(step_secs[-3:]) > a.deadline:
            break
        t0 = time.time()
        rank_gen.manual_seed(int(torch.randint(0, 2**62, (1,), generator=gen, device='cuda')) + rank)
        n_global = a.prompts_per_rank * world
        idx = [order[(cursor + k) % len(order)] for k in range(n_global)]
        mine = idx[rank * a.prompts_per_rank:(rank + 1) * a.prompts_per_rank]
        groups = []
        for j in mine:
            r = rows[j]
            traj = sample_group(fwd, r['ids'], a.group, a.budget, eos, pad, force_ids, '</answer>',
                                lambda t: tok.decode(t), rank_gen)
            rew, preds = zip(*[reward_fn(t, r['target'], f, a.reward_scale, a.forced_penalty) for t, f in zip(traj['texts'], traj['forced'])])
            groups.append((r, traj, list(rew), list(preds)))
        t_sample = time.time() - t0
        # global token count for normalization
        ntok = torch.tensor([sum(sum(m) for m in g[1]['masks']) for g in groups], device='cuda').sum().double()
        if ddp:
            dist.all_reduce(ntok)
        opt.zero_grad(set_to_none=True)
        stats = []
        for r, traj, rew, preds in groups:
            rw = torch.tensor(rew)
            adv = rw - rw.mean()
            if a.adv_std:
                adv = adv / (rw.std() + a.adv_eps)
            mask, old = row_masks(traj)
            if adv.abs().max() > 1e-6:
                cur = replay_group(fwd, traj, a.block_size)
                c = cur.detach().cpu()
                if not torch.equal(c * mask, old.float() * mask):
                    # old lps were float() of the same fp32 value; compare via float32
                    if not torch.equal((c * mask).float(), (old * mask).float()):
                        raise RuntimeError('selected log-prob mismatch')
                m = mask.to(cur.device); A = adv.to(cur.device).view(1, -1)
                loss = -((cur - cur.detach()).exp() * A * m).sum() / ntok
                loss.backward()
            errs = [abs(p - r['target']) for p in preds if p is not None]
            stats.append(dict(hour=r['hour'], target=r['target'], reward=float(rw.mean()), reward_std=float(rw.std()),
                              mae=sum(errs) / len(errs) if errs else None, parsed=len(errs) / len(preds),
                              forced=sum(traj['forced']) / len(preds),
                              len=sum(len(t) for t in traj['tokens']) / len(preds),
                              median_pred=sorted(p for p in preds if p is not None)[len(errs) // 2] if errs else None))
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if ddp:
            for p in params:
                dist.all_reduce(p.grad)
        gnorm = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        if not torch.isfinite(gnorm):
            raise RuntimeError('nonfinite grad')
        opt.step()
        step += 1
        cursor += n_global
        dig = params_digest(params)
        if ddp:
            all_d = [None] * world
            dist.all_gather_object(all_d, dig)
            all_stats = [None] * world
            dist.all_gather_object(all_stats, stats)
            if len(set(all_d)) != 1:
                raise RuntimeError('ranks diverged')
            stats = [s for ss in all_stats for s in ss]
        step_secs.append(time.time() - t0)
        if rank == 0:
            maes = [s['mae'] for s in stats if s['mae'] is not None]
            rec = dict(step=step, time=time.time(), secs=step_secs[-1], sample_secs=t_sample, grad_norm=float(gnorm),
                       tokens=int(ntok), reward=sum(s['reward'] for s in stats) / len(stats),
                       mae=sum(maes) / len(maes) if maes else None,
                       parsed=sum(s['parsed'] for s in stats) / len(stats),
                       forced=sum(s['forced'] for s in stats) / len(stats),
                       len=sum(s['len'] for s in stats) / len(stats), weights=dig[:16], groups=stats,
                       sample_text=groups[0][1]['texts'][0][-600:])
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(json.dumps({k: v for k, v in rec.items() if k not in ('groups', 'sample_text')}), flush=True)
            save_ckpt(out, step, model, opt, gen, cursor, order)
        if ddp:
            dist.barrier()
    if rank == 0 and step >= a.steps:
        (out / 'DONE').write_text(str(step))
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
