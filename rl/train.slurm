#!/usr/bin/env bash
#SBATCH --job-name=mining_grpo
#SBATCH --account=cell_reason
#SBATCH --partition=quick_preemptible
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/grpo_%j.log
# Usage: OUT=... NGPU=4 EXTRA="--steps 200" sbatch [--gres=gpu:N] rl/train.slurm
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f cluster.env ] && . ./cluster.env  # site paths, see cluster.env.example
env=${CONDA_ENV:?set CONDA_ENV in cluster.env}
export PATH="$env/bin:$PATH" CUDA_HOME="$env" LD_PRELOAD="$env/lib/libstdc++.so.6"
export PYTHONPATH=${EXTRA_PYTHONPATH:-} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1
export TRITON_CACHE_DIR=/tmp/mining_grpo_${SLURM_JOB_ID}/triton TORCHINDUCTOR_CACHE_DIR=/tmp/mining_grpo_${SLURM_JOB_ID}/inductor
NGPU=${NGPU:-4}
trap 'echo "USR1: requeue"; kill -TERM $PID 2>/dev/null; wait $PID; [ -f "$OUT/DONE" ] || scontrol requeue $SLURM_JOB_ID; exit 0' USR1
END=$(squeue -h -j $SLURM_JOB_ID -o %e); DEADLINE=$(( $(date -d "$END" +%s) - 420 ))
torchrun --standalone --nproc_per_node=$NGPU rl/exact_grpo.py --out "$OUT" --deadline $DEADLINE $EXTRA &
PID=$!
wait $PID; rc=$?
echo "exit $rc"
if [ $rc -eq 0 ] && [ ! -f "$OUT/DONE" ]; then scontrol requeue $SLURM_JOB_ID; fi
