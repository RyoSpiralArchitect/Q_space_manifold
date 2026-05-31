# Pre/Post-RoPE SUBJ Pilot

Date: 2026-05-25

This note records the first targeted pre/post-RoPE comparison for the Mistral
instruction-tuned SUBJ probe.

## Claim Hygiene Update

Update added 2026-06-01: the compact headline below,
`pre-RoPE: stronger, sharper, earlier` and `post-RoPE: weaker, broader, later`,
should be read as a description of this Mistral-IT N=100 pilot only. It does
not generalize cleanly to the later N=1000 cross-family matrix. The larger
matrix supports a weaker and safer claim: Q-space task signals can survive
RoPE, but exact layer/head shifts and small score deltas are single-run
variation until reproduced across seeds, samples, and matching full
pre/post-RoPE sweeps.

The goal is to separate two experimental surfaces:

```text
pre-RoPE Q  = raw query posture before rotary positional phase
post-RoPE Q = position-aware query used before attention scoring
```

The working question is whether the stance-separating geometry seen in Q-space
is only a positional artifact, or whether it survives after the model's rotary
position phase is applied.

## Setup

- model: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- backend: MLX
- dataset: `SetFit/subj`
- split: `train`
- samples: `100 subjective + 100 objective`
- projection: PCA
- main recurring heads from earlier probes: `L10/H6`, `L11/H22`

The post-RoPE implementation captures the query tensor from the model's actual
RoPE call before attention scaling/scoring.

## Quick Pool-1 Check

The first post-RoPE check used the same basic SUBJ setting as the representative
pre-RoPE run, with `pool_last_k=1`.

| capture | target layer/head | best layer/head | best score | notable layer score |
| --- | ---: | ---: | ---: | --- |
| pre-RoPE | L11/H22 | L11/H22 | 0.2290 | L11/H22 = 0.2290 |
| post-RoPE | L11/H22 | L10/H6 | 0.2177 | L11/H14 = 0.2009; L11/H22 = 0.1669 |

The signal does not disappear after RoPE. However, the head identity and layer
peak reorganize: the original `L11/H22` weakens, while `L10/H6` becomes the
best overall post-RoPE head.

## Matched Pool-3 / Drop-Special Check

To reduce dependence on a single final punctuation or control token position,
the next pass used:

```text
--drop-special-tokens
--pool-last-k 3
```

| capture | target layer/head | best layer/head | best score | target/local score |
| --- | ---: | ---: | ---: | ---: |
| pre-RoPE | L10/H6 | L11/H22 | 0.1801 | L10/H6 = 0.1578 |
| post-RoPE | L10/H6 | L16/H26 | 0.1538 | L10/H6 = 0.1468 |
| post-RoPE | L11/H22 | L16/H26 | 0.1538 | L11/H22 = 0.1166 |

The matched comparison gives a useful middle result:

```text
signal survives, but reorganizes
```

It is not a complete match between pre-RoPE and post-RoPE. That would make RoPE
look irrelevant to the probe. It is also not a collapse to noise. That would
push the interpretation toward a pure positional artifact. Instead, the current
pilot shows a weaker and broader post-RoPE geometry, with peak separation
drifting later.

## Interpretation

The current best short description is:

```text
pre-RoPE:   stronger, sharper, earlier
post-RoPE: weaker, broader, later
```

This is consistent with the following working distinction:

- pre-RoPE Q is closer to content-conditioned retrieval posture: what the head
  is preparing to search for.
- post-RoPE Q mixes that posture with sequence-position structure and
  autoregressive addressing: where or when that search is being situated.

The recurring appearance of `L10/H6` and `L11/H22` across SUBJ, prompted SST-2,
pooling checks, and pre/post-RoPE comparisons suggests that the Mistral-IT
signal is not a pure positional phenomenon. A cautious phrase for the current
finding is:

```text
Mistral-IT appears to expose a local stance-query band around layers 10-11.
RoPE does not erase the band, but it rotates the readout into a broader,
partly later position-aware geometry.
```

This is still correlational. Causal ablation and dense same-family comparisons
are needed before making stronger claims.

## Reproduction Commands

Post-RoPE, pool-1:

```bash
./q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source subj \
  --samples-per-class 100 \
  --target-layer 11 \
  --target-head 22 \
  --q-capture-stage post-rope \
  --projection pca \
  --output-dir /tmp/q_space_post_rope_subj_mistral_l11_h22
```

Post-RoPE, pool-3, `L10/H6`:

```bash
./q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source subj \
  --samples-per-class 100 \
  --target-layer 10 \
  --target-head 6 \
  --q-capture-stage post-rope \
  --projection pca \
  --drop-special-tokens \
  --pool-last-k 3 \
  --output-dir /tmp/q_space_post_rope_subj_mistral_l10_h6_pool3
```

Pre-RoPE, pool-3, `L10/H6`:

```bash
./q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source subj \
  --samples-per-class 100 \
  --target-layer 10 \
  --target-head 6 \
  --q-capture-stage pre-rope \
  --projection pca \
  --drop-special-tokens \
  --pool-last-k 3 \
  --output-dir /tmp/q_space_pre_rope_subj_mistral_l10_h6_pool3
```

Post-RoPE, pool-3, `L11/H22`:

```bash
./q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source subj \
  --samples-per-class 100 \
  --target-layer 11 \
  --target-head 22 \
  --q-capture-stage post-rope \
  --projection pca \
  --drop-special-tokens \
  --pool-last-k 3 \
  --output-dir /tmp/q_space_post_rope_subj_mistral_l11_h22_pool3
```

## Next Step

Run the same pre/post-RoPE comparison across:

```text
3 families x 2 tuning states x 2 task framings
```

using the n=1000/class 3D matrix settings. This should be done before the dense
MacBook Pro pass, so the dense run can decide whether the observed pre/post
reorganization is family-level, tuning-level, or quantization-sensitive.
