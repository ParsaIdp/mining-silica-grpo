# Silica from flotation data: LLM reasoning + exact-policy GRPO

Can a small open model, trained with RL, learn to estimate a flotation plant's lab assay from its sensor data?
This repo uses the Kaggle **Quality Prediction in a Mining Process** dataset (iron-ore reverse cationic flotation,
Mar–Sep 2017). Each prompt is one hour of 20 s sensor readings plus the hour's other lab assays, and the model must
give that hour's `% Silica Concentrate`.

The repo covers:
- classical baselines, tuned with time-series CV;
- zero-shot frontier LLMs (gpt-6-astra, Inkling 2 Small) run in a sandboxed harness;
- **GRPO on Qwen3.5-4B** with bitwise-identical sampling and training.

Full write-up with charts: **[Silica GRPO Report](https://claude.ai/artifact/HRSK7LDUEr5w2uAobXjftB)**.

## Headline results

All rows score the same 100 held-out validation hours (2017-07-15 to 07-31). Error is MAE in wt% SiO₂; lower is better.

| Model | Information in the prompt | Val MAE |
|---|---|---|
| Train mean | none | 0.77 |
| Base Qwen3.5-4B, no thinking | this hour | 0.83 |
| GRPO Qwen, round 2 | this hour | 0.59 (near-constant guess) |
| gpt-6-astra (high effort) | this hour | 0.52 |
| Ridge, current-hour Fe assay only | this hour | 0.43 |
| Repeat last hour's silica | lags | 0.47 |
| gpt-6-astra | 3 lags of silica | 0.475 |
| GRPO Qwen, round 1 | 3 lags of silica | 0.49 |
| **GRPO Qwen, round 2 (step 130)** | **3 lags of silica + Fe** | **0.356** |
| Ridge, hourly means + lags | lags | 0.36 |
| Base Qwen, **thinking mode**, 6k-token cap | 3 lags of silica + Fe | 0.337 |
| gpt-6-astra | 3 lags of silica + Fe | 0.316 |
| One-parameter iron-change rule | last silica + ΔFe | 0.319 (test 0.366) |

Qwen rows use the median of 4 sampled answers. Astra gives one answer per hour.

What we learned:
1. **Silica is almost entirely explained by the same-hour iron assay** (Fe and SiO₂ are measured on the same sample).
   Process sensors add almost nothing. Without the Fe assay, no model beats the train mean on a time split.
2. With past assays in context, the best predictor is *last silica − 0.85 × ΔFe*. The in-context information ceiling is about 0.31.
3. **RL works when the signal is in the prompt.** Round-2 GRPO took Qwen from 0.72 to 0.356. Its reasoning became short
   and input-grounded: it compares this hour's Fe with previous hours. It learned the right sign of the iron rule but
   about a third of its slope.
4. **RL collapses to a constant guess when the prompt has only weak signal** (this-hour-only runs). Under GRPO, group
   variance is the only learning signal, and a constant guess removes it.
5. **Thinking mode alone reaches 0.337** with a 6k-token cap and forced answer. The planned thinking-mode GRPO run
   (`--thinking 1 --budget 6144`) is implemented but never got cluster time.

## Layout

```
scripts/   data prep, classical baselines, prompts, API-model harnesses (sandboxed), scoring
rl/        exact-policy GRPO trainer, LoRA-merge + vLLM evaluation, Slurm wrappers
results/   baseline CSVs, Astra/Inkling outputs, rl_evals.jsonl (every checkpoint eval), spend ledger
```

Not committed: the Kaggle data, checkpoints (`runs/`), logs, and large raw Qwen generations.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env              # API keys, only for the API-model evals
cp cluster.env.example cluster.env  # paths to conda env, Qwen3.5-4B checkpoint, BioReasonCell checkout
```

Data: download `MiningProcess_Flotation_Plant_Database.csv` from Kaggle
(`edumagalhaes/quality-prediction-in-a-mining-process`) to `data/raw/`, then run `python scripts/prepare.py`.

`prepare.py` writes:
- `data/hours.parquet`: one row per hour, the raw CSV table without the target;
- `data/hour_features.parquet`: hourly aggregates.

Data facts:
- **Split:** train < 2017-08-01 (3,137 h), test ≥ 2017-08-01 (960 h).
- **Interpolated hours:** 320 hours have lab values interpolated within the hour. They are kept for training and excluded from LLM evals.
- **RL split:** RL trains on hours < 2017-07-15. Validation is 100 non-interpolated hours from 2017-07-15 to 07-31. Test is `common.eval_subset(200)`.

## Classical baselines

- `scripts/baselines.py`, `scripts/strong_baselines.py`: see `results/baselines.csv` and `results/strong_baselines.csv`.
- `scripts/baselines_v2.py` selects models by time-series CV on train only:
  - Setting A (per-hour) reaches test MAE 0.51 (LGBM + CatBoost + ElasticNet ensemble).
  - Setting B (lags allowed) reaches 0.39 with ridge + lags; persistence scores 0.41.

## API-model evals

- `scripts/eval_astra.py`: gpt-6-astra through the Codex CLI. Prompt options: `--prompt raw|summary_hint`, `--split test|val`, `--lags 3 --lag_iron 1 --brief 1`.
- `scripts/eval_cc.py` + `scripts/cc_run.py`: Inkling 2 Small through headless Claude Code pointed at OpenRouter. The free tier serves this model only to agent harnesses.
- **Sandbox:** every sample runs in a `bwrap` sandbox that sees only its own work directory, so labels are unreachable.
- **Spend:** about $35 in total, logged in `results/spend_ledger.jsonl`.

## RL: exact-policy GRPO (`rl/`)

`rl/exact_grpo.py` runs GRPO on Qwen3.5-4B with LoRA (r=32, α=64, fp32 adapters, AdamW lr 2e-5).

**Bitwise identity.** Sampling and training use the same HF forward. The functional GDN/attention cache comes from
BioReasonCell's `exact_policy` module, extended to lockstep groups of G rollouts.
- Every sampling step hashes the full-vocabulary logits and log-probs.
- Replay and backward recomputation re-check those hashes, under checkpointed blocks.
- A mismatch aborts the step before the optimizer runs.
- With DDP, LoRA weight digests are also compared across ranks.

Identity held on every step of every run.

**Protocol.**
- The policy is softmax(logits/T) with no top-p.
- When the reasoning budget runs out, the trainer appends a forced suffix, `Final estimate: <answer>`, which is loss-masked. In thinking mode the suffix closes `</think>` first.
- **Reward:** exp(−|err|/scale), minus a penalty if the answer was forced; −0.5 if the answer can't be parsed.
- **Advantage:** Dr-GRPO (r − group mean), optionally divided by the group std (`--adv_std 1`).

**Dependency:** the four `bioreason_cell.rl.exact_policy` modules come from a BioReasonCell checkout
(set `BIOREASONCELL`). That repo is not public yet, so the RL trainer needs access to it. Everything else here stands alone.

```bash
# round-2 best config (lags + iron, brief reasoning, 256-token cap)
OUT=runs/grpo_v2_lagfe NGPU=4 EXTRA="--steps 300 --prompts_per_rank 4 --group 4 --budget 256 --block_size 32 \
  --temperature 1.0 --reward_scale 0.25 --adv_std 1 --forced_penalty 0.2 --brief 1 --lags 3 --lag_iron 1" sbatch rl/train.slurm
# evaluate a checkpoint with the same protocol (merges LoRA, runs vLLM)
ARGS="--ckpt runs/grpo_v2_lagfe/ckpt_00130 --split val --budget 256 --temperature 1.0 --brief 1 --lags 3 --lag_iron 1 \
  --out results/rl/x.jsonl" sbatch rl/eval.slurm
# or auto-evaluate every 10th checkpoint
bash rl/watch_evals.sh runs/grpo_v2_lagfe "--budget 256 --temperature 1.0 --brief 1 --lags 3 --lag_iron 1"
```

Slurm and memory:
- **Requeue:** `train.slurm` checkpoints every step and requeues itself on preemption (`--signal=B:USR1@300`).
- **Host memory:** replay keeps one KV snapshot per checkpoint block, so host memory grows roughly with T²/block.
  A 12k-token thinking budget exceeded 335 GB per job. The 6k config uses `--block_size 32` and about 700 GB.

## Runs

| Run | Prompt | Steps | Best val MAE |
|---|---|---|---|
| `grpo_v1` | this hour, 384-token cap | 200 | 0.57 (step 80); otherwise ~0.63, a constant guess |
| `grpo_lag3` | + 3 silica lags | 137 | 0.49 |
| `grpo_v2_hour` | this hour, sharper reward | 85 | 0.59, a constant guess |
| `grpo_v2_lagfe` | + 3 silica and Fe lags, brief | 137 | **0.356** (step 130) |
| `grpo_v3_think6k` | thinking mode, 6k cap | – | queued, never scheduled |

Every checkpoint evaluation is in `results/rl_evals.jsonl`.
