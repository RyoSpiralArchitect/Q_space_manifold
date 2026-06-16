# CodeXGLUE Code-Language Pre/Post-RoPE Sweep

Date: 2026-06-14/15

This note records the first medium-scale code-language Q-space sweep with
matched pre-RoPE and post-RoPE captures. The task is not sentiment,
subjectivity, or question type. It asks whether final-token Q geometry contains
a language-family routing signal for code snippets.

The note began as a Mistral-7B-Instruct pilot. It now also includes the matched
six-model base/instruction-tuned sweep across Mistral, Llama 3, and Gemma 2 2B.

Compact tracked artifacts:

- `examples/codexglue_code_language_6models_pre_post_rope_n1000/pre_rope_pool_last_k_sweep_summary.csv`
- `examples/codexglue_code_language_6models_pre_post_rope_n1000/post_rope_pool_last_k_sweep_summary.csv`
- `examples/codexglue_code_language_6models_pre_post_rope_n1000/pre_post_pool_last_k_comparison.csv`
- `examples/codexglue_code_language_6models_pre_post_rope_n1000/pre_post_best_per_model_comparison.csv`
- `examples/codexglue_code_language_n1000_mistral_it_pre_rope/pool_last_k_sweep_summary.csv`
- `examples/codexglue_code_language_n1000_mistral_it_post_rope/pool_last_k_sweep_summary.csv`

Large full outputs, including `q_space_vectors.npz`, remain outside the repo:

- `~/q_space_runs/codexglue_code_language_n1000_6models_pre_rope_len64_capped`
- `~/q_space_runs/codexglue_code_language_n1000_6models_post_rope_len64_capped`
- `~/q_space_runs/codexglue_code_language_n1000_mistral_it_pre_rope_len64_capped_sweep`
- `~/q_space_runs/codexglue_code_language_n1000_mistral_it_post_rope_len64_capped_sweep`

## Settings

- dataset: `google/code_x_glue_ct_code_to_text`
- split: `validation`
- task framing: code-language routing
- classes: `python`, `java`, `javascript`, `go`, `php`, `ruby`
- sample: `1000` rows per class, `6000` rows total
- models: Mistral-7B base/instruct, Llama-3-8B base/instruct, Gemma-2-2B
  base/instruct, all current `mlx-community/*-4bit` checkpoints
- backend: MLX
- Q captures: pre-RoPE Q projection output and post-RoPE Q before attention
  scoring
- pooling: `--pool-last-k-sweep 1,3,5`
- token caps: `--max-token-length 64`, `--max-stored-tokens 5`,
  `--stored-token-selection tail`, `--token-q-storage-dtype float16`
- projection: PCA
- plots: disabled for this run with `--no-plots`
- controls: 100 silhouette label permutations, top-1 layer/head null, linear
  probe without permutation nulls

### Sampling Policy

The source JSON was built from the Hugging Face Dataset Viewer rows API and
stores balanced counts:

```text
python 1000 + java 1000 + javascript 1000 + go 1000 + php 1000 + ruby 1000
= 6000 rows
```

There is no class imbalance in this tracked CodeXGLUE matrix. Each language
contributes exactly 1000 validation rows. The six-model pre/post runs reuse the
same JSON file, so model-family and pre/post comparisons are over the same
balanced sample set.

The code text had already been capped in the dataset file to 3000 characters.
The run further capped model input to 64 tokens and retained only the last 5
token Q records for token-flow outputs. The final-token or pooled-final-token
Q vectors still use all 6000 examples.

This cap is an important caveat. The current late-routing interpretation is a
result under `--max-token-length 64`; it may depend on how much of each snippet
is visible and where the retained tail falls after truncation. Relaxing the
token-length cap on a larger machine is required before treating the late
code-language profile as a property of uncapped code contexts.

## Six-Model Headline

The first surprise was that the Mistral-IT result generalized. At
`pool_last_k=5`, all six model aliases show a late code-language readout. The
exact head is family-specific, but the depth is consistently late relative to
the earlier natural-language stance probes.

| model | pre-RoPE best | pre sil | pre probe | post-RoPE best | post sil | post probe | post/pre |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral-7B base | L21/H18 | 0.1011 | 0.9765 | L21/H18 | 0.0940 | 0.9737 | 0.93 |
| Mistral-7B instruct | L21/H18 | 0.0924 | 0.9733 | L21/H18 | 0.0860 | 0.9712 | 0.93 |
| Llama-3-8B base | L19/H30 | 0.2031 | 0.9818 | L19/H30 | 0.1783 | 0.9810 | 0.88 |
| Llama-3-8B instruct | L19/H30 | 0.1807 | 0.9788 | L19/H30 | 0.1492 | 0.9792 | 0.83 |
| Gemma-2-2B base | L19/H4 | 0.0929 | 0.9800 | L20/H5 | 0.0718 | 0.9815 | 0.77 |
| Gemma-2-2B-it | L25/H5 | 0.0492 | 0.9662 | L25/H0 | 0.0313 | 0.9593 | 0.64 |

The six-model version strengthens three parts of the reading:

- **late band**: all best rows are at relative depth `0.61-1.00`;
- **pooling amplification**: widening from `pool_last_k=1` to `5` increases
  silhouette for every model in both capture stages;
- **RoPE survival**: post-RoPE is usually weaker, but Mistral and Llama 3 keep
  the same best layer/head at `pool_last_k=5`.

Gemma 2 2B remains the contrast case. The base model keeps a late readable
code-language row, while the instruction-tuned model has a weaker final-layer
surface. Even there, the best-head linear probe remains high, so the cautious
phrasing is weak single-head clustering, not absence of code-language
information.

## Mistral-IT Pooling Detail

Across pre-RoPE and post-RoPE, and across all three pooling values, the same
best layer/head appears in the original Mistral-IT pilot:

| capture | pool_last_k | best layer/head | relative depth | silhouette | null mean | null z | p>=actual | LOO probe acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-RoPE | 1 | L21/H18 | 0.677 | 0.0345 | -0.0066 | 54.70 | 0.0099 | 0.8930 |
| pre-RoPE | 3 | L21/H18 | 0.677 | 0.0683 | -0.0076 | 74.20 | 0.0099 | 0.9540 |
| pre-RoPE | 5 | L21/H18 | 0.677 | 0.0924 | -0.0080 | 87.21 | 0.0099 | 0.9733 |
| post-RoPE | 1 | L21/H18 | 0.677 | 0.0324 | -0.0065 | 52.27 | 0.0099 | 0.8907 |
| post-RoPE | 3 | L21/H18 | 0.677 | 0.0637 | -0.0077 | 67.02 | 0.0099 | 0.9517 |
| post-RoPE | 5 | L21/H18 | 0.677 | 0.0860 | -0.0081 | 84.37 | 0.0099 | 0.9712 |

The single-head silhouette is modest in absolute terms, but the random-label
null is strongly separated. More importantly, the best head is stable under
pooling and under the pre/post-RoPE capture change. The score still increases
monotonically as the final-token readout is pooled over the last 1, 3, and 5
tokens.

## Late Band

Top layer/head rows for `pool_last_k=5`:

| rank | pre-RoPE layer/head | pre silhouette | post-RoPE layer/head | post silhouette |
| ---: | ---: | ---: | ---: | ---: |
| 1 | L21/H18 | 0.0924 | L21/H18 | 0.0860 |
| 2 | L22/H10 | 0.0778 | L22/H10 | 0.0763 |
| 3 | L23/H10 | 0.0750 | L23/H29 | 0.0718 |
| 4 | L23/H28 | 0.0745 | L22/H27 | 0.0700 |
| 5 | L23/H29 | 0.0740 | L22/H26 | 0.0698 |
| 6 | L22/H27 | 0.0728 | L23/H10 | 0.0690 |
| 7 | L22/H26 | 0.0727 | L23/H28 | 0.0669 |
| 8 | L24/H29 | 0.0720 | L22/H17 | 0.0669 |
| 9 | L22/H17 | 0.0714 | L22/H21 | 0.0657 |
| 10 | L25/H24 | 0.0711 | L25/H24 | 0.0647 |

This looks more like a late code-language band than a lone magic head. In
Mistral the best single readout is L21/H18, while neighboring top rows cluster
across roughly L21-L25 in both capture stages. The six-model run extends that
band-level framing: Llama 3 peaks at L19/H30, Gemma 2 2B base around L19-L20,
and Gemma 2 2B-it at the final layer.

## Interpretation

For natural-language SUBJ/SST/TREC runs, useful Q-space readouts often appeared
in earlier or middle bands depending on model family and task. This code
language run moves the strongest readout later. In the six-model `pool_last_k=5`
matrix, the best rows sit at relative depth `0.61-1.00`.

The matched post-RoPE run weakens the metric slightly but does not reorganize
the location:

```text
pre-RoPE:  L21/H18, silhouette 0.0345 -> 0.0683 -> 0.0924
post-RoPE: L21/H18, silhouette 0.0324 -> 0.0637 -> 0.0860
```

A cautious interpretation is:

```text
In these 4bit CodeXGLUE code-language runs, Q-space contains a late,
pooling-amplified code-language routing signal. Unlike the earlier
natural-language stance probes, this capped code-language readout is largely
RoPE-stable in Mistral and Llama 3: the same best layer/head survives post-RoPE
with only a modest score reduction. Gemma 2 2B remains weaker and more diffuse,
especially after instruction tuning.
```

This should not yet be generalized to uncapped code contexts or dense
checkpoints. It is a six-model 4bit run, capped to 64 model tokens, with only
the tail token-flow records retained. The next checks are dense checkpoints,
less aggressive token caps on a larger machine, and causal ablations.

## Deeper Reading

This run is a useful counterpoint to the earlier natural-language sweeps. In
SUBJ, SST-2, and TREC, the most visible Q-space readouts looked like stance,
polarity, or question-frame probes, and their strongest bands could move by
model family, instruction tuning, or capture stage. Here the label is not a
discourse stance. It is code language identity, and the strongest readout moves
later while remaining fairly stable across pre-RoPE and post-RoPE capture.

That suggests a more task-dependent picture:

```text
natural-language stance probes:
  often expose earlier or middle query-posture bands

code-language routing probe:
  exposes a later, pooled-tail readout after local syntax and lexical evidence
  has accumulated
```

The recurrence of Mistral L21/H18 is especially important because it survives
all six matched Mistral conditions: pre/post capture crossed with
`pool_last_k=1`, `3`, and `5`. The six-model rerun adds a second exact
recurrence in Llama 3 at L19/H30. The right unit of interpretation is therefore
probably not "one special head", but a family-specific late code-language
readout band with one strongest local probe.

The monotonic pooling effect sharpens that reading. The final token alone
already carries the signal, but pooling the last 3 or 5 tokens makes it much
more visible. This is consistent with a tail-local aggregate readout rather
than a single terminal-token artifact.

The modest silhouette should also be read carefully. The classes are not
forming large compact blobs in raw cosine geometry; the silhouette values are
small. But the null-separated silhouette and high leave-one-out linear probe
accuracy say the information is strongly linearly readable. In other words,
the code-language signal may be distributed across several directions rather
than expressed as clean spherical clusters.

One practical lesson transfers back to the broader atlas work: the
`--target-layer-fraction 0.35` natural-language heuristic is not reliable for
code. At the target layer around L11, the headline heads are weak or even
negative in the top-layer/head tables, while the scan finds the readout late.
Code-language experiments need a full layer/head scan, not only a
stance-inspired mid-layer pin.

## Practical Note

At `6000` samples, full layer/head scanning plus random-label nulls is slow.
The matched post-RoPE sweep completed with:

```text
--top-layer-head-null-rank-limit 1
```

This keeps a null statistic for the headline best row while avoiding an
expensive top-5 null pass on every pooling value.
