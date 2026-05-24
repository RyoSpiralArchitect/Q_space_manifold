# MBP Dense Follow-Up Handoff

Date: 2026-05-25

Use this file to restart the Q-space manifold work in a fresh thread on the
MacBook Pro.

## Fresh Thread Prompt

Paste this into the new thread after opening the repository:

```text
We are continuing Ryo's Q-space manifold probe in
/Users/ryohiga/SpiralReality/Q_space_manifold.

Read these first:
- README.md
- docs/research_notes/n1000_3d_matrix.md
- docs/research_notes/base_vs_instruct_subj.md
- docs/research_notes/sst2_base_vs_prompted.md

The latest completed run is the n=1000/class 3D matrix over Mistral, Llama 3,
and Gemma 2 2B base/instruct 4bit checkpoints. Post-RoPE Q capture has now been
implemented with `--q-capture-stage post-rope` for MLX RoPE models. The next
goal is to compare pre-RoPE vs post-RoPE on the strongest 4bit heads, then
repeat the same SUBJ and prompted SST-2 matrix on dense same-family checkpoints
on this MacBook Pro, keeping all analysis flags identical where possible.
```

## Current Repository State

Important commits on `main`:

- `a7ea15d Replace representative flows with 3D plots`
- `706edf8 Add linear probe permutation control`

Expected local noise on the MacBook Air source checkout:

```text
?? .DS_Store
?? examples/.DS_Store
```

Those files are not part of the experiment.

## Completed Large Runs

SUBJ:

```text
output_dir: /tmp/q_space_subj_n1000_base_vs_it_3d
sample: 1000/class, 2000 rows/model
models: 6 base/instruct 4bit configs
plots: 3D, plot sample limit 200
```

Best heads:

```text
mistral_base:   L10/H6   score 0.2270
mistral_it:     L7/H15   score 0.2154
llama3_base:    L11/H6   score 0.1758
llama3_it:      L20/H31  score 0.2187
gemma2_2b_base: L21/H4   score 0.1663
gemma2_2b_it:   L1/H0    score 0.0345
```

Prompted SST-2:

```text
output_dir: /tmp/q_space_sst2_prompted_n1000_base_vs_it_3d
sample: 1000/class, 2000 rows/model
template: Review: {text}\nSentiment:
models: 6 base/instruct 4bit configs
pool sweep: 1,3,5
plots: 3D, plot sample limit 200
```

Strongest rows by model:

```text
mistral_base:   k=5 L10/H21  score 0.0978
mistral_it:     k=1 L23/H30  score 0.1726
llama3_base:    k=1 L20/H24  score 0.1205
llama3_it:      k=1 L18/H28  score 0.2246
gemma2_2b_base: k=5 L12/H4   score 0.0587
gemma2_2b_it:   k=5 L12/H3   score 0.0266
```

Compact tracked tables live in `examples/n1000_3d_matrix/`.

## Dense Follow-Up Shape

Before dense, run a small post-RoPE comparison on the strongest 4bit heads. The
dense follow-up should then be a same-family 12-cell matrix:

```text
3 model families x 2 tuning states x 2 task framings
```

Task framings:

- SUBJ, no template
- SST-2 with `Review: {text}\nSentiment:`

Keep these flags:

```text
--samples-per-class 1000
--target-layer-fraction 0.35
--target-head 4
--projection pca
--detail-best-layer-head
--label-permutation-n 200
--high-d-flow-metrics
--projection-diagnostics
--probe-linear
--linear-probe-permutation-n 0
--head-similarity
--drop-special-tokens
--flow-start-token-index 1
--plot-3d
--plot-sample-limit 200
```

For post-RoPE comparison, add:

```text
--q-capture-stage post-rope
```

The default is still pre-RoPE:

```text
--q-capture-stage pre-rope
```

For SST-2 also keep:

```text
--pool-last-k-sweep 1,3,5
--text-template $'Review: {text}\nSentiment:'
```

## Dense Command Templates

Resolve the exact dense model IDs on the MacBook Pro before downloading. Keep
the aliases stable so the CSVs compare cleanly with the 4bit run.

SUBJ template:

```bash
./q_space_manifold_monolith.py \
  --dataset-source subj \
  --dataset-split train \
  --samples-per-class 1000 \
  --batch-models mistral_dense_base=mlx:<DENSE_MISTRAL_BASE>,mistral_dense_it=mlx:<DENSE_MISTRAL_IT>,llama3_dense_base=mlx:<DENSE_LLAMA3_BASE>,llama3_dense_it=mlx:<DENSE_LLAMA3_IT>,gemma2_2b_dense_base=mlx:<DENSE_GEMMA2_2B_BASE>,gemma2_2b_dense_it=mlx:<DENSE_GEMMA2_2B_IT> \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --projection pca \
  --detail-best-layer-head \
  --label-permutation-n 200 \
  --high-d-flow-metrics \
  --projection-diagnostics \
  --probe-linear \
  --linear-probe-permutation-n 0 \
  --head-similarity \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --plot-3d \
  --plot-sample-limit 200 \
  --output-dir /tmp/q_space_subj_n1000_dense_base_vs_it_3d
```

Prompted SST-2 template:

```bash
./q_space_manifold_monolith.py \
  --dataset-source sst2 \
  --dataset-split train \
  --samples-per-class 1000 \
  --text-template $'Review: {text}\nSentiment:' \
  --batch-models mistral_dense_base=mlx:<DENSE_MISTRAL_BASE>,mistral_dense_it=mlx:<DENSE_MISTRAL_IT>,llama3_dense_base=mlx:<DENSE_LLAMA3_BASE>,llama3_dense_it=mlx:<DENSE_LLAMA3_IT>,gemma2_2b_dense_base=mlx:<DENSE_GEMMA2_2B_BASE>,gemma2_2b_dense_it=mlx:<DENSE_GEMMA2_2B_IT> \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --pool-last-k-sweep 1,3,5 \
  --projection pca \
  --detail-best-layer-head \
  --label-permutation-n 200 \
  --high-d-flow-metrics \
  --projection-diagnostics \
  --probe-linear \
  --linear-probe-permutation-n 0 \
  --head-similarity \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --plot-3d \
  --plot-sample-limit 200 \
  --output-dir /tmp/q_space_sst2_prompted_n1000_dense_base_vs_it_3d
```

## What To Compare

Primary comparison:

- Do the strongest 4bit heads survive post-RoPE, or does positional phase
  smear/relocate the stance geometry?
- Does Mistral remain an early/mid stable stance band?
- Does Llama 3 instruct still migrate deeper and strengthen?
- Does Gemma 2 2B-it remain flat/diffuse in dense form?
- Do prompted SST-2 IT peaks survive without 4bit quantization?

Secondary comparison:

- Head similarity CKA/RSA at best layers
- `pool_last_k` sensitivity
- 3D flow readability at the strongest heads
- Linear probe accuracy and label-permutation controls

If dense results match the 4bit pattern, the architecture/tuning hypothesis
becomes much stronger. If dense results differ, quantization becomes a real
experimental axis rather than a nuisance detail.
