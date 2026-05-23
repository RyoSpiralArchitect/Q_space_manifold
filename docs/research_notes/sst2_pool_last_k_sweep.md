# SST-2 Pool-Last-K Sweep

Date: 2026-05-24

This note records an instruction-tuned cross-model sweep on GLUE/SST-2 with
`pool_last_k = 1,3,5`. The goal is to test whether the Q-space stance signal is
mostly a final-token artifact, and whether sentiment polarity behaves like the
earlier SUBJ subjectivity/objectivity signal.

The scan used:

- dataset: `glue/sst2`
- split: `train`
- sample: `100 negative + 100 positive`
- backend: MLX
- projection: PCA
- Q capture: pre-RoPE Q projection output
- models: Mistral-7B-Instruct, Llama-3-8B-Instruct, Gemma-2-2B-it
- controls: random-label silhouette, linear probe, projection diagnostics, and
  best-layer head similarity

Raw compact tables are in `examples/sst2_pool_last_k_it_3models/`.

Reproduction command:

```bash
./q_space_manifold_monolith.py \
  --dataset-source sst2 \
  --dataset-split train \
  --samples-per-class 100 \
  --batch-models mistral_it=mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit,llama3_it=mlx:mlx-community/Meta-Llama-3-8B-Instruct-4bit,gemma2_2b_it=mlx:mlx-community/gemma-2-2b-it-4bit \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --projection pca \
  --pool-last-k-sweep 1,3,5 \
  --detail-best-layer-head \
  --label-permutation-n 100 \
  --high-d-flow-metrics \
  --projection-diagnostics \
  --probe-linear \
  --head-similarity \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --no-plots \
  --output-dir /tmp/q_space_sst2_pool_sweep_it_3models
```

## Headline Pattern

Compared with SUBJ, SST-2 produces a much weaker single-head Q-space separation
signal:

```text
SUBJ: strong subjectivity/objectivity stance geometry
SST-2: weaker polarity geometry, still above random-label controls
```

The result argues against a simple final-token artifact, because increasing
`pool_last_k` does not collapse the signal. But it also suggests that sentiment
polarity is not as cleanly localized in a single Q head as subjective/objective
framing.

## Best Layer / Head by Pooling Window

Best layer/head by high-dimensional cosine silhouette:

| model | pool_last_k | best layer/head | relative depth | silhouette |
| --- | ---: | ---: | ---: | ---: |
| Mistral-7B-Instruct | 1 | L15/H6 | 0.484 | 0.0790 |
| Mistral-7B-Instruct | 3 | L15/H6 | 0.484 | 0.0858 |
| Mistral-7B-Instruct | 5 | L15/H6 | 0.484 | 0.0863 |
| Llama-3-8B-Instruct | 1 | L14/H10 | 0.452 | 0.0621 |
| Llama-3-8B-Instruct | 3 | L14/H27 | 0.452 | 0.0516 |
| Llama-3-8B-Instruct | 5 | L14/H27 | 0.452 | 0.0532 |
| Gemma-2-2B-it | 1 | L12/H3 | 0.480 | 0.0165 |
| Gemma-2-2B-it | 3 | L15/H0 | 0.600 | 0.0227 |
| Gemma-2-2B-it | 5 | L15/H0 | 0.600 | 0.0264 |

## Probe and Control Diagnostics

| model | pool_last_k | best head | LOO linear probe | random-label p | mean CKA to best head | mean RSA to best head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral-7B-Instruct | 1 | L15/H6 | 0.755 | 0.0099 | 0.576 | 0.475 |
| Mistral-7B-Instruct | 3 | L15/H6 | 0.745 | 0.0099 | 0.619 | 0.533 |
| Mistral-7B-Instruct | 5 | L15/H6 | 0.745 | 0.0099 | 0.757 | 0.585 |
| Llama-3-8B-Instruct | 1 | L14/H10 | 0.750 | 0.0099 | 0.716 | 0.947 |
| Llama-3-8B-Instruct | 3 | L14/H27 | 0.715 | 0.0099 | 0.845 | 0.970 |
| Llama-3-8B-Instruct | 5 | L14/H27 | 0.755 | 0.0099 | 0.899 | 0.974 |
| Gemma-2-2B-it | 1 | L12/H3 | 0.555 | 0.0099 | 0.809 | 0.750 |
| Gemma-2-2B-it | 3 | L15/H0 | 0.620 | 0.0099 | 0.566 | 0.523 |
| Gemma-2-2B-it | 5 | L15/H0 | 0.575 | 0.0099 | 0.590 | 0.575 |

The p-value floor is `1 / (100 + 1) = 0.0099`, so these controls should be read
as "stronger than all 100 random-label samples", not as a precise significance
estimate.

## Working Interpretation

For Mistral, the best SST-2 head is stable across pooling windows at L15/H6, and
the silhouette rises slightly from `0.0790` to `0.0863`. This makes a pure
final-punctuation explanation less likely.

For Llama 3, the best layer remains stable around relative depth `0.45`, but the
best head changes from H10 to H27 when pooling over more tokens. The linear probe
remains useful, while RSA/CKA indicate a highly redundant head neighborhood.

For Gemma 2 2B-it, the signal remains weak, though pooling over more tokens
raises the best silhouette from `0.0165` to `0.0264`. This is consistent with the
earlier "weak or distributed single-head geometry" reading.

The contrast with SUBJ is the important part: subjective/objective framing looks
much more visible in Q-space than positive/negative sentiment polarity. A
reasonable next hypothesis is that Q-space heads may expose discourse stance or
information-seeking posture more cleanly than valence itself. SST-2 polarity may
require a prompt-style classification framing, a larger sample, post-RoPE
capture, or probes over other internal surfaces such as residual stream, MLP
activations, or logits.

## Next Checks

- Run the same SST-2 sweep on base checkpoints.
- Compare prompted SST-2 variants such as `Review: ... Sentiment:`.
- Increase permutation count beyond 100 for the weaker SST-2 effects.
- Compare pre-RoPE and post-RoPE Q capture.
- Revisit SST-2 on larger or dense Gemma checkpoints.
