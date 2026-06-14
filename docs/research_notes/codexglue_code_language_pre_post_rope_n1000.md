# CodeXGLUE Code-Language Pre/Post-RoPE Sweep

Date: 2026-06-14/15

This note records the first medium-scale code-language Q-space sweep with
matched pre-RoPE and post-RoPE captures. The task is not sentiment,
subjectivity, or question type. It asks whether final-token Q geometry contains
a language-family routing signal for code snippets.

Compact tracked artifacts:

- `examples/codexglue_code_language_n1000_mistral_it_pre_rope/pool_last_k_sweep_summary.csv`
- `examples/codexglue_code_language_n1000_mistral_it_post_rope/pool_last_k_sweep_summary.csv`

Large full outputs, including `q_space_vectors.npz`, remain outside the repo:

- `~/q_space_runs/codexglue_code_language_n1000_mistral_it_pre_rope_len64_capped_sweep`
- `~/q_space_runs/codexglue_code_language_n1000_mistral_it_post_rope_len64_capped_sweep`

## Settings

- dataset: `google/code_x_glue_ct_code_to_text`
- split: `validation`
- task framing: code-language routing
- classes: `python`, `java`, `javascript`, `go`, `php`, `ruby`
- sample: `1000` rows per class, `6000` rows total
- model: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
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

The source JSON was built from the Hugging Face Dataset Viewer rows API and
stores balanced counts:

```text
python 1000 + java 1000 + javascript 1000 + go 1000 + php 1000 + ruby 1000
= 6000 rows
```

The code text had already been capped in the dataset file to 3000 characters.
The run further capped model input to 64 tokens and retained only the last 5
token Q records for token-flow outputs. The final-token or pooled-final-token
Q vectors still use all 6000 examples.

## Headline Result

Across pre-RoPE and post-RoPE, and across all three pooling values, the same
best layer/head appears:

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

This looks more like a late code-language band than a lone magic head. The
best single readout is L21/H18, while neighboring top rows cluster across
roughly L21-L25 in both capture stages.

## Interpretation

For natural-language SUBJ/SST/TREC runs, useful Q-space readouts often appeared
in earlier or middle bands depending on model family and task. This code
language run moves the strongest readout later: relative depth `0.677`, with a
broader late band through roughly L21-L25.

The matched post-RoPE run weakens the metric slightly but does not reorganize
the location:

```text
pre-RoPE:  L21/H18, silhouette 0.0345 -> 0.0683 -> 0.0924
post-RoPE: L21/H18, silhouette 0.0324 -> 0.0637 -> 0.0860
```

A cautious interpretation is:

```text
In this Mistral-IT 4bit CodeXGLUE code-language run, Q-space contains a late,
pooling-stable code-language routing signal. Unlike the earlier Mistral SUBJ
pilot where post-RoPE appeared weaker and somewhat reorganized, this capped
code-language readout is largely RoPE-stable: the same best layer/head and
late band survive post-RoPE with only a modest score reduction.
```

This should not yet be generalized across model families or uncapped code
contexts. It is a single-model run, capped to 64 model tokens, with only the
tail token-flow records retained. The next checks are other model families,
base-vs-instruct comparisons, and a less aggressive token cap on a larger
machine.

## Deeper Reading

This run is a useful counterpoint to the earlier natural-language sweeps. In
SUBJ, SST-2, and TREC, the most visible Q-space readouts looked like stance,
polarity, or question-frame probes, and their strongest bands could move by
model family, instruction tuning, or capture stage. Here the label is not a
discourse stance. It is code language identity, and the strongest readout moves
later while remaining nearly invariant across pre-RoPE and post-RoPE capture.

That suggests a more task-dependent picture:

```text
natural-language stance probes:
  often expose earlier or middle query-posture bands

code-language routing probe:
  exposes a later, pooled-tail readout after local syntax and lexical evidence
  has accumulated
```

The recurrence of L21/H18 is especially important because it survives all six
matched conditions in this run: pre/post capture crossed with `pool_last_k=1`,
`3`, and `5`. The surrounding top rows also stay in a late L21-L25 band. So the
right unit of interpretation is probably not "one special head", but a late
code-language readout band with one strongest local probe.

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
negative in the top-layer/head tables, while the scan finds the readout near
relative depth 0.68. Code-language experiments need a full layer/head scan, not
only a stance-inspired mid-layer pin.

## Practical Note

At `6000` samples, full layer/head scanning plus random-label nulls is slow.
The matched post-RoPE sweep completed with:

```text
--top-layer-head-null-rank-limit 1
```

This keeps a null statistic for the headline best row while avoiding an
expensive top-5 null pass on every pooling value.
