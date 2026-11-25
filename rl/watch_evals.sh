#!/usr/bin/env bash
# Submit a val eval for each new ckpt_XXXX0 of a run. Usage: rl/watch_evals.sh runs/grpo_v1
RUN=$1; EXTRA_EVAL=${2:-}; NAME=$(basename $RUN); cd "$(dirname "$0")/.."
while true; do
  for c in $RUN/ckpt_*0; do
    [ -d "$c" ] || continue
    s=$(basename $c); flag=$RUN/.eval_$s
    [ -f $flag ] && continue
    ARGS="--ckpt $c --split val $EXTRA_EVAL --out results/rl/${NAME}_${s}_val.jsonl" sbatch --job-name=ev_${NAME}_$s rl/eval.slurm && touch $flag
  done
  [ -f $RUN/DONE ] && break
  sleep 600
done
