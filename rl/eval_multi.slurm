#!/usr/bin/env bash
#SBATCH --job-name=mining_rlevalm
#SBATCH --account=cell_reason
#SBATCH --partition=quick_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rlevalm_%j.log
# Usage: sbatch rl/eval_multi.slurm FILE  (one eval_ckpt.py argument line per row)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f cluster.env ] && . ./cluster.env  # site paths, see cluster.env.example
env=${CONDA_ENV:?set CONDA_ENV in cluster.env}
export PYTHONPATH=${EXTRA_PYTHONPATH:-} PATH="$env/bin:$PATH" CUDA_HOME="$env" LD_PRELOAD="$env/lib/libstdc++.so.6" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
while read -r line; do [ -n "$line" ] && python rl/eval_ckpt.py $line; done < "$1"
