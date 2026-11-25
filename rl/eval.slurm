#!/usr/bin/env bash
#SBATCH --job-name=mining_rleval
#SBATCH --account=cell_reason
#SBATCH --partition=quick_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/rleval_%j.log
# Usage: ARGS="--ckpt runs/x/ckpt_00010 --split val --out results/rl/x_10_val.jsonl" sbatch rl/eval.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f cluster.env ] && . ./cluster.env  # site paths, see cluster.env.example
env=${CONDA_ENV:?set CONDA_ENV in cluster.env}
export PYTHONPATH=${EXTRA_PYTHONPATH:-} PATH="$env/bin:$PATH" CUDA_HOME="$env" LD_PRELOAD="$env/lib/libstdc++.so.6" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python rl/eval_ckpt.py $ARGS
