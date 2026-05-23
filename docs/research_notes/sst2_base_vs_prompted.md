# SST-2 Base-vs-Instruct and Prompt Framing

Date: 2026-05-24

This note records two follow-ups to the first SST-2 `pool_last_k` sweep:

1. a base-vs-instruction-tuned SST-2 sweep on naked review sentences;
2. a prompted SST-2 sweep using `Review: {text}\nSentiment:` as a framing
   template.

The goal is to test whether weak SST-2 polarity separation means that polarity
is absent from Q-space, or whether the probe needs a task frame that makes the
model form an explicit sentiment-query stance.

Raw compact tables are in:

- `examples/sst2_base_vs_instruct_pool_last_k/`
- `examples/sst2_prompted_pool_last_k_it_3models/`

## Reproduction Commands

Base-vs-instruct naked SST-2:

```bash
./q_space_manifold_monolith.py \
  --dataset-source sst2 \
  --dataset-split train \
  --samples-per-class 100 \
  --batch-models mistral_base=mlx:mlx-community/Mistral-7B-v0.3-4bit,mistral_it=mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit,llama3_base=mlx:mlx-community/Meta-Llama-3-8B-4bit,llama3_it=mlx:mlx-community/Meta-Llama-3-8B-Instruct-4bit,gemma2_2b_base=mlx:mlx-community/gemma-2-2b-4bit,gemma2_2b_it=mlx:mlx-community/gemma-2-2b-it-4bit \
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
  --output-dir /tmp/q_space_sst2_pool_sweep_base_vs_it
```

Prompted instruction-tuned SST-2:

```bash
./q_space_manifold_monolith.py \
  --dataset-source sst2 \
  --dataset-split train \
  --samples-per-class 100 \
  --text-template $'Review: {text}\nSentiment:' \
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
  --output-dir /tmp/q_space_sst2_prompted_pool_sweep_it_3models
```

## Headline Pattern

Naked SST-2 remains weak. Prompted SST-2 becomes much clearer for Mistral and
Llama 3:

```text
Mistral naked k=1:  L15/H6   silhouette 0.0790
Mistral prompt k=1: L23/H30  silhouette 0.1681

Llama3 naked k=1:   L14/H10  silhouette 0.0621
Llama3 prompt k=1:  L20/H9   silhouette 0.2239

Gemma2-it naked k=1:  L12/H3 silhouette 0.0165
Gemma2-it prompt k=1: L1/H6  silhouette 0.0240
```

This supports the idea that the earlier weak SST-2 result was not simply
"polarity is absent from Q-space." Instead, naked review sentences may not force
a clean sentiment-query stance at the sampled position. Adding a task frame
appears to light up a separable query-routing geometry, especially in Mistral
and Llama 3.

## Naked SST-2: Base vs Instruct

Best layer/head by high-dimensional cosine silhouette:

| model | pool_last_k | best layer/head | relative depth | silhouette |
| --- | ---: | ---: | ---: | ---: |
| Mistral base | 1 | L15/H6 | 0.484 | 0.0772 |
| Mistral base | 3 | L15/H6 | 0.484 | 0.0837 |
| Mistral base | 5 | L15/H6 | 0.484 | 0.0847 |
| Mistral instruct | 1 | L15/H6 | 0.484 | 0.0790 |
| Mistral instruct | 3 | L15/H6 | 0.484 | 0.0858 |
| Mistral instruct | 5 | L15/H6 | 0.484 | 0.0863 |
| Llama3 base | 1 | L14/H10 | 0.452 | 0.0636 |
| Llama3 base | 3 | L14/H27 | 0.452 | 0.0662 |
| Llama3 base | 5 | L14/H27 | 0.452 | 0.0696 |
| Llama3 instruct | 1 | L14/H10 | 0.452 | 0.0621 |
| Llama3 instruct | 3 | L14/H27 | 0.452 | 0.0516 |
| Llama3 instruct | 5 | L14/H27 | 0.452 | 0.0532 |
| Gemma2 2B base | 1 | L14/H7 | 0.560 | 0.0408 |
| Gemma2 2B base | 3 | L14/H7 | 0.560 | 0.0464 |
| Gemma2 2B base | 5 | L14/H7 | 0.560 | 0.0471 |
| Gemma2 2B-it | 1 | L12/H3 | 0.480 | 0.0165 |
| Gemma2 2B-it | 3 | L15/H0 | 0.600 | 0.0227 |
| Gemma2 2B-it | 5 | L15/H0 | 0.600 | 0.0264 |

Interpretation:

- **Mistral** is stable under instruction tuning: same best head and nearly the
  same score.
- **Llama 3** does not show the SUBJ-style instruction-tuned boost on naked
  SST-2. The base model is slightly stronger for `pool_last_k=3,5`.
- **Gemma 2 2B** again weakens after instruction tuning, matching the earlier
  SUBJ pattern qualitatively.

## Prompted SST-2

Best layer/head by high-dimensional cosine silhouette:

| model | pool_last_k | best layer/head | relative depth | silhouette | LOO linear probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mistral instruct | 1 | L23/H30 | 0.742 | 0.1681 | 0.825 |
| Mistral instruct | 3 | L20/H18 | 0.645 | 0.0981 | 0.750 |
| Mistral instruct | 5 | L10/H21 | 0.323 | 0.0976 | 0.860 |
| Llama3 instruct | 1 | L20/H9 | 0.645 | 0.2239 | 0.845 |
| Llama3 instruct | 3 | L20/H9 | 0.645 | 0.1426 | 0.875 |
| Llama3 instruct | 5 | L8/H15 | 0.258 | 0.1131 | 0.750 |
| Gemma2 2B-it | 1 | L1/H6 | 0.040 | 0.0240 | 0.590 |
| Gemma2 2B-it | 3 | L1/H6 | 0.040 | 0.0366 | 0.590 |
| Gemma2 2B-it | 5 | L12/H1 | 0.480 | 0.0309 | 0.595 |

All prompted best-head silhouette scores beat the 100-sample random-label
controls. The p-value floor is `1 / (100 + 1) = 0.0099`.

The strongest prompted effects are late and concentrated:

- Mistral moves from naked L15/H6 to prompted L23/H30 for `pool_last_k=1`.
- Llama 3 moves from naked L14/H10 to prompted L20/H9 for `pool_last_k=1`.
- Gemma 2 2B-it remains weak and does not show a comparable prompt-induced
  stance head.

## Working Interpretation

The prompt appears to create an explicit sentiment-query position. The final
token in `Review: {text}\nSentiment:` is not a label leak; it is the same task
cue for every sample. But its Q vector is allowed to condition on the preceding
review, so it can become a query for the evidence needed to complete the
sentiment field.

This suggests a sharper distinction:

```text
naked sentence Q-space:
  weak polarity geometry; the model is reading, not necessarily classifying

prompted sentiment Q-space:
  explicit query-routing geometry; the model is preparing to classify
```

That fits the broader stance-phase hypothesis better than the first SST-2 result
alone. The probe may be more sensitive to *how the model is being asked to use a
text* than to the latent topic or valence of the text in isolation.

## Cautions

- Prompted `pool_last_k=1` may be especially sensitive to the final task cue
  token. This is useful, but it should be interpreted as "classification stance"
  rather than generic sentence meaning.
- Prompt wording is now an experimental factor. Variants such as
  `Question: Is this review positive or negative? Answer:` should be tested.
- The Gemma 2 2B prompted result remains weak; larger Gemma checkpoints may
  behave differently.

## Next Checks

- Run prompted SST-2 on base checkpoints.
- Sweep prompt templates and compare final-token versus answer-token positions.
- Add post-RoPE Q capture for the prompted run.
- Add a label-balanced causal ablation test for Mistral L23/H30 and Llama3
  L20/H9.
