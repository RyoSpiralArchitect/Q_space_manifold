#!/usr/bin/env zsh
set -e
set -u
set -o pipefail

cd "$(dirname "$0")/.."

MODELS='mistral_base=mlx:mlx-community/Mistral-7B-v0.3-4bit,mistral_it=mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit,llama3_base=mlx:mlx-community/Meta-Llama-3-8B-4bit,llama3_it=mlx:mlx-community/Meta-Llama-3-8B-Instruct-4bit,gemma2_2b_base=mlx:mlx-community/gemma-2-2b-4bit,gemma2_2b_it=mlx:mlx-community/gemma-2-2b-it-4bit'

run_sweep() {
  local stage="$1"
  shift

  local common_flags=(
    --batch-models "$MODELS"
    --activation-space k
    --q-capture-stage "$stage"
    --pool-last-k-sweep 1,3,5
    --target-layer-fraction 0.35
    --target-head 4
    --projection pca
    --detail-best-layer-head
    --label-permutation-n 200
    --linear-probe-permutation-n 50
    --probe-linear
    --high-d-flow-metrics
    --projection-diagnostics
    --head-similarity
    --drop-special-tokens
    --flow-start-token-index 1
    --max-stored-tokens 5
    --stored-token-selection tail
    --token-q-storage-dtype float16
    --no-plots
    --resume-existing
  )

  print "\n=== K-space ${stage}: SUBJ n1000 ==="
  ./q_space_manifold_monolith.py \
    --dataset-source subj \
    --dataset-split train \
    --samples-per-class 1000 \
    "${common_flags[@]}" \
    --output-dir "${HOME}/q_space_runs/k_space_subj_n1000_6models_${stage}_pool_sweep"

  print "\n=== K-space ${stage}: prompted SST-2 n1000 ==="
  ./q_space_manifold_monolith.py \
    --dataset-source sst2 \
    --dataset-split train \
    --text-template $'Review: {text}\nSentiment:' \
    --samples-per-class 1000 \
    "${common_flags[@]}" \
    --output-dir "${HOME}/q_space_runs/k_space_sst2_prompted_n1000_6models_${stage}_pool_sweep"

  print "\n=== K-space ${stage}: TREC coarse n1000ish ==="
  ./q_space_manifold_monolith.py \
    --dataset-source hf \
    --hf-dataset-name omkar334/trec \
    --dataset-split train \
    --text-column text \
    --label-column coarse_label \
    --samples-per-class 1000 \
    "${common_flags[@]}" \
    --output-dir "${HOME}/q_space_runs/k_space_trec_coarse_n1000ish_6models_${stage}_pool_sweep"

  print "\n=== K-space ${stage}: CodeXGLUE code language n1000 len64 ==="
  ./q_space_manifold_monolith.py \
    --dataset-source json \
    --dataset-json "${HOME}/q_space_runs/datasets/codexglue_codesearchnet_code_language_validation_n1000.json" \
    --samples-per-class 0 \
    --max-token-length 64 \
    "${common_flags[@]}" \
    --output-dir "${HOME}/q_space_runs/k_space_codexglue_code_language_n1000_6models_${stage}_pool_sweep_len64"
}

for stage in pre-rope post-rope; do
  run_sweep "$stage"
done

print "\n=== K-space sweeps complete ==="
