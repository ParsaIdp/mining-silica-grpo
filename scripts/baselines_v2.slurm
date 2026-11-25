#!/usr/bin/env bash
#SBATCH --job-name=mining_baselines
#SBATCH --account=cell_reason
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baselines_v2_%j.log
cd "${SLURM_SUBMIT_DIR:-.}" && .venv/bin/python scripts/baselines_v2.py
