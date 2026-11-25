#!/usr/bin/env bash
#SBATCH --job-name=mining_qwen_summ
#SBATCH --account=cell_reason
#SBATCH --partition=quick_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/qwen_eval_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f cluster.env ] && . ./cluster.env  # site paths, see cluster.env.example
env=${CONDA_ENV:?set CONDA_ENV in cluster.env}
export PYTHONPATH=${EXTRA_PYTHONPATH:-} PATH="$env/bin:$PATH" CUDA_HOME="$env" LD_PRELOAD="$env/lib/libstdc++.so.6" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python scripts/eval_qwen_vllm.py --prompt summary --thinking 1 --max_tokens 4096 --out results/qwen35_4b_summary_think4k.jsonl
python scripts/eval_qwen_vllm.py --prompt summary_hint --thinking 1 --max_tokens 4096 --out results/qwen35_4b_summaryhint_think4k.jsonl
