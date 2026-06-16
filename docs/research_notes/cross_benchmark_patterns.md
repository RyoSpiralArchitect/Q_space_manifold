# Cross-Benchmark Q-Space Patterns

Date: 2026-06-16

This note summarizes the patterns that have emerged across the current
medium-scale benchmark set:

- SUBJ subjectivity/objectivity
- prompted SST-2 sentiment polarity
- TREC coarse question type
- CodeXGLUE / CodeSearchNet code language

It is a synthesis note, not a final claim. The evidence is still geometric and
predictive rather than causal, and the benchmark-level patterns below are
single-seed observations unless a section explicitly says otherwise. All model
rows discussed here use the current MLX 4bit checkpoints unless otherwise
stated.

## Compact Artifacts

The main tracked tables used here are:

- `examples/n1000_3d_matrix/n1000_3d_pre_post_headline_comparison.csv`
- `examples/n1000_3d_matrix/subj_n1000_3d_batch_top_layer_heads.csv`
- `examples/n1000_3d_matrix/sst2_prompted_n1000_3d_pool_last_k_sweep_summary.csv`
- `examples/n1000_3d_matrix/sst2_prompted_n1000_3d_post_rope_pool_last_k_sweep_summary.csv`
- `examples/trec_coarse_pre_post_rope_n1000ish/pre_post_best_per_model_comparison.csv`
- `examples/trec_coarse_pre_post_rope_n1000ish/pre_post_pool_last_k_comparison.csv`
- `examples/codexglue_code_language_6models_pre_post_rope_n1000/pre_post_best_per_model_comparison.csv`
- `examples/codexglue_code_language_6models_pre_post_rope_n1000/pre_post_pool_last_k_comparison.csv`

## Benchmark Matrix

| benchmark | task readout | sample shape | representative strength | strongest depth pattern | pooling pattern | pre/post-RoPE behavior |
| --- | --- | --- | --- | --- | --- | --- |
| SUBJ | subjective/objective stance | `1000/class`, 2 classes | max sil `0.227`; tracked probe up to `0.935` in the smaller base/IT probe artifact | Mistral early/mid, Llama-IT late, Gemma-base late, Gemma-IT weak/diffuse | `pool_last_k=1` in the tracked headline | signal survives post-RoPE; exact heads may shift |
| prompted SST-2 | sentiment polarity under explicit prompt frame | `1000/class`, 2 classes | max sil `0.225` pre, `0.198` post; retained post probe `0.908` on Llama3-IT | Mistral/Llama-IT late or mid/late strongest; Gemma-IT weak | pooling changes both score and location | signal survives, but prompt/cue position makes the readout more mobile |
| TREC coarse | answer-type routing for questions | `4817` rows, 6 imbalanced classes | max sil `0.102` pre, `0.089` post; matching probe `0.839/0.831` | mostly mid or mid/late around L17-L18 for Mistral/Llama/Gemma-base | `pool_last_k=1` is usually strongest | very stable pre/post at headline level |
| CodeXGLUE code language | code-language routing | `1000/class`, 6 balanced languages | max sil `0.203` pre, `0.178` post; matching probe `0.982/0.981` | late across all families; Mistral L21/H18, Llama L19/H30, Gemma late/final | `pool_last_k=1 -> 3 -> 5` strengthens every model | largely stable in Mistral/Llama; weaker and more diffuse in Gemma |

The probe values in this table are representative tracked values, not a single
uniformly recomputed probe table across every benchmark. TREC and CodeXGLUE use
matching rows from their pre/post comparison artifacts. SUBJ and prompted SST-2
use the retained probe artifacts available for their notes.

The strongest cross-benchmark lesson is not that there is one universal
Q-space head. The pattern is closer to:

```text
task-conditioned Q-space readouts recur,
but their depth, pooling behavior, and head localization depend on the task
and model family.
```

## Common Patterns

### 1. Q-space carries task-conditioned readouts

Across all four benchmark types, the headline rows are not random-label
artifacts. The most visible signals differ by task, but the same probe family
keeps finding non-random layer/head surfaces:

- SUBJ: subjectivity/objectivity stance;
- prompted SST-2: sentiment-query stance under an explicit prompt cue;
- TREC: answer-type routing;
- CodeXGLUE: code-language routing.

The safest wording is "readout" rather than "mechanistic unit". These are
places where the relevant variable is visible in Q-space; causal ablation is
still needed before assigning responsibility.

### 2. RoPE usually weakens or shifts; it rarely erases

The early Mistral-IT SUBJ pilot suggested a "weaker, broader, later" post-RoPE
story. The larger matrix makes that too strong as a general headline. A better
summary is:

```text
headline task signals usually survive the pre/post-RoPE capture change,
while exact scores and head identities can shift.
```

TREC and CodeXGLUE are the cleanest survival cases. TREC preserves most
headline rows exactly. CodeXGLUE preserves Mistral L21/H18 and Llama L19/H30
through post-RoPE at `pool_last_k=5`, with modest score reductions.

### 3. Silhouette and probe accuracy measure different geometry

Multi-class tasks often have modest silhouette but high linear probe accuracy.
This is especially clear in TREC and CodeXGLUE. The working interpretation is
that some Q-space task variables are linearly readable without forming compact,
spherical class clusters under cosine silhouette.

That means a weak silhouette is not automatically "no information". It may mean
"not a clean single-head manifold cluster".

### 4. Pooling is diagnostic, not just robustness

`pool_last_k` changes what part of the readout is being measured:

- TREC is strongest at `k=1`, consistent with the final question-token stance
  asking what answer type should be retrieved next.
- prompted SST-2 moves under pooling because the final `Sentiment:` cue and the
  neighboring review tokens are different readout positions.
- CodeXGLUE strengthens monotonically from `k=1` to `k=5`, suggesting a
  tail-local aggregate of code-language evidence rather than a single terminal
  token artifact.

Pooling should therefore be recorded as part of the experimental condition, not
only as a stability check.

### 5. Full layer/head scans remain necessary

The convenient `--target-layer-fraction 0.35` anchor is useful for first-look
plots, but it is not a reliable evidence shortcut. It misses the CodeXGLUE late
band and can understate prompted SST-2 or TREC rows depending on pooling.

Claim-facing tables should continue to come from the full layer x head
silhouette scan, with target-layer plots treated as diagnostics.

## Fine Differences

### SUBJ: stance separation

SUBJ is the cleanest stance-style benchmark so far. Mistral stays relatively
stable around early/mid bands. Llama 3 instruction tuning moves the strongest
readout later. Gemma 2 2B base has a late readable axis, while Gemma 2 2B-it is
weak and rank-unstable.

This makes SUBJ useful as the first "stance formation" probe, but it should not
be used as the template for every task.

### Prompted SST-2: cue-sensitive sentiment readout

Prompted SST-2 is more sensitive to prompt framing and pooling. Mistral-IT and
Llama3-IT show strong sentiment-query rows when the input ends with an explicit
`Sentiment:` cue. This is useful, but it also creates a confound: instruction
tuning may be improving prompt-following and cue use, not only abstract
sentiment representation.

### TREC: answer-type routing

TREC is weaker by raw silhouette than SUBJ, but it is stable and linearly
readable. Its strongest rows usually prefer `pool_last_k=1`, and most
pre/post-RoPE headline rows stay fixed.

The main caveat is dataset imbalance: `ABBR` contributes only 86 rows in the
tracked `n1000ish` run. The ABBR-excluded check did not erase the headline
readouts, but absolute six-class silhouette comparisons should still be read
with that imbalance in mind.

### CodeXGLUE: late code-language routing

CodeXGLUE is the strongest task-dependent counterpoint. The readout moves late
across all six model aliases, strengthens monotonically as the final-token pool
widens, and survives post-RoPE in Mistral and Llama 3 with the same best
layer/head at `pool_last_k=5`.

The tracked CodeXGLUE dataset is balanced:

```text
python 1000 + java 1000 + javascript 1000 + go 1000 + php 1000 + ruby 1000
= 6000 rows
```

The source JSON caps code text at 3000 characters, and the current six-model
run additionally uses `--max-token-length 64`. Therefore the late-routing
interpretation is provisional until the token-length cap is relaxed. The current
result shows a late readout under this capped condition; it does not yet prove
that the same depth profile holds for long, uncapped code contexts.

The current reading is:

```text
code-language identity is not behaving like early/mid stance formation;
it looks more like a late pooled-tail routing/readout after syntax and lexical
evidence has accumulated.
```

## Working Hypothesis

A compact hypothesis that fits the current benchmark set is:

```text
Q-space task readouts appear where the model has enough context to form the
next attention query for that task.

Stance and prompt-cue tasks can appear in early/mid or cue-local bands.
Answer-type questions often sharpen at the final question stance.
Code-language identity appears later and strengthens when the tail context is
pooled.
```

This is intentionally weaker than a mechanistic claim. It says where the
variable is readable, not yet which head causes downstream behavior.

## Next Checks

- Run a dedicated silhouette-vs-probe geometry audit for TREC and CodeXGLUE:
  pairwise class centroid distances, within/between-class scatter,
  one-vs-rest margins, kNN accuracy, and probe accuracy as a function of PCA
  dimension. This should test whether the signal is a compact manifold cluster
  or a distributed linear code.
- Run dense checkpoints for the same CodeXGLUE matrix.
- Relax the CodeXGLUE `--max-token-length 64` cap on a larger machine.
- Add K-space and V-space scans to separate Q-specific routing from generic
  representation separability.
- Add causal ablation for recurring heads or bands, especially Mistral L21/H18
  and Llama L19/H30 on CodeXGLUE.
- Repeat key matrices across dataset seeds before interpreting small
  pre/post-RoPE score deltas.
