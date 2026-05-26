# Base vs Instruction-Tuned SUBJ Scan

Date: 2026-05-24

This note records an early base-vs-instruction-tuned comparison on the SUBJ
dataset. The goal is to test whether the measured stance-separating Q-space
geometry changes with model family and instruction tuning.

The scan used:

- dataset: `SetFit/subj`
- split: `train`
- sample: `100 subjective + 100 objective`
- backend: MLX
- projection: PCA
- Q capture: pre-RoPE Q projection output
- controls: random-label silhouette, linear probe, projection diagnostics, and
  best-layer head similarity

Raw tables are in `examples/base_vs_instruct_subj/`.

## Headline Pattern

The observed pattern is:

```text
Mistral: preserve + specialize
Llama 3: strengthen + migrate deeper
Gemma 2 2B: weaken + flatten/diffuse
```

This suggests that instruction tuning does not uniformly amplify stance
separation. Instead, it may preserve, relocate, or diffuse the Q-space stance
phase depending on model family.

## Best Layer / Head

Best layer/head by high-dimensional cosine silhouette:

| model | best layer/head | relative depth | silhouette |
| --- | ---: | ---: | ---: |
| Mistral-7B base | L10/H6 | 0.323 | 0.2307 |
| Mistral-7B instruct | L11/H22 | 0.355 | 0.2290 |
| Llama-3-8B base | L13/H27 | 0.419 | 0.1899 |
| Llama-3-8B instruct | L20/H31 | 0.645 | 0.2276 |
| Gemma-2-2B base | L21/H4 | 0.840 | 0.1815 |
| Gemma-2-2B-it | L12/H1 | 0.480 | 0.0410 |

Interpretation:

- **Mistral** is stable: subjectivity/objectivity separation is already strong
  in the base model and remains strong after instruction tuning.
- **Llama 3** strengthens and moves later: the top head shifts from relative
  depth `0.419` to `0.645` and the silhouette increases.
- **Gemma 2 2B** changes most sharply: the base model has a late single-head
  stance axis, while the instruction-tuned model loses most single-head
  separability.

## Head Similarity

Head similarity was measured in each model's best layer with pairwise linear
CKA and RSA over head-level Q-space geometry.

| model | best head | mean CKA | mean RSA | nearest heads by CKA |
| --- | ---: | ---: | ---: | --- |
| Mistral-7B base | L10/H6 | 0.666 | 0.687 | H4, H5, H9 |
| Mistral-7B instruct | L11/H22 | 0.577 | 0.638 | H19, H23, H20 |
| Llama-3-8B base | L13/H27 | 0.696 | 0.732 | H26, H6, H23 |
| Llama-3-8B instruct | L20/H31 | 0.655 | 0.772 | H30, H20, H23 |
| Gemma-2-2B base | L21/H4 | 0.748 | 0.824 | H0, H1, H5 |
| Gemma-2-2B-it | L12/H1 | 0.804 | 0.877 | H3, H0, H5 |

This adds texture to the headline pattern:

- **Mistral preserve + specialize**: silhouette is preserved while mean CKA/RSA
  drop after instruction tuning, suggesting the stance axis remains but head
  geometry becomes less redundant or more specialized.
- **Llama strengthen + migrate deeper**: silhouette increases and the best
  layer shifts later. CKA drops slightly while RSA rises, suggesting a later
  stance cluster whose sample-distance geometry becomes more shared.
- **Gemma weaken + flatten/diffuse**: silhouette drops sharply while CKA/RSA
  rise. The instruction-tuned model appears more head-redundant and less
  single-head-localized.

## Working Interpretation

These observations are not enough to rule out every artifact. They are best read
as a structured hypothesis generator: the probe sees different family/tuning
interactions that should be checked against random-label nulls, pre/post-RoPE
capture, sample-size stability, and causal ablation.

```text
Mistral: alignment preserves the stance phase.
Llama 3: alignment relocates and strengthens the stance phase.
Gemma 2 2B: alignment diffuses or flattens the single-head stance phase.
```

The evidence is still correlational. The stronger claim requires post-RoPE
comparison beyond the current pilot, sample-size stability, and causal ablation.

## Next Checks

- Run the same base-vs-instruct scan on SST-2.
- Run `pool_last_k` robustness checks for `1,3,5`.
- Add post-RoPE Q capture and compare whether the pre-RoPE stance phase remains
  visible in actual attention queries.
- Test larger Gemma variants, especially Gemma 2 9B, when more memory is
  available.
- Add causal ablation of candidate heads after the geometry is stable across
  datasets.
