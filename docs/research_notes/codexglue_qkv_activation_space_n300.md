# CodeXGLUE Q/K/V Activation-Space Sweep

Date: 2026-06-21

This note records a matched pre-RoPE activation-space comparison for the
CodeXGLUE / CodeSearchNet code-language benchmark. It follows the earlier
six-model Q-space pre/post-RoPE sweep and asks a narrower question:

```text
Is the late code-language readout specific to Q-space, or is it also visible in
the K and V projection spaces?
```

## Artifacts

Compact tracked artifact:

- `examples/codexglue_qkv_activation_space_n300/qkv_best_layer_heads.csv`
- `examples/codexglue_qkv_activation_space_n300/resid_pre_comparison.csv`

Large full outputs, including vector bundles, remain outside the repo:

- `~/q_space_runs/codexglue_code_language_n300_6models_qkv_pre_rope_pool5_len64/q`
- `~/q_space_runs/codexglue_code_language_n300_6models_qkv_pre_rope_pool5_len64/k`
- `~/q_space_runs/codexglue_code_language_n300_6models_qkv_pre_rope_pool5_len64/v`
- `~/q_space_runs/codexglue_code_language_n300_6models_resid_pre_pool5_len64`

## Settings

- dataset: CodeSearchNet code-language JSON derived from the CodeXGLUE
  code-to-text validation split
- classes: `python`, `java`, `javascript`, `go`, `php`, `ruby`
- sample: `300` rows per class, `1800` rows total
- models: Mistral-7B base/instruct, Llama-3-8B base/instruct, Gemma-2-2B
  base/instruct, all current `mlx-community/*-4bit` checkpoints
- backend: MLX
- activation spaces: `--activation-space q`, `k`, `v`, and a later
  `resid_pre` baseline
- capture stage: `--q-capture-stage pre-rope`
- pooling: `--pool-last-k 5`
- token caps: `--max-token-length 64`, `--max-stored-tokens 5`,
  `--stored-token-selection tail`, `--token-q-storage-dtype float16`
- projection: PCA
- plots: disabled with `--no-plots`

For K/V rows, the reported head count is the KV head count under GQA, not the
full Q head count. Mistral and Llama 3 expose 8 K/V heads here; Gemma 2 2B
exposes 4 K/V heads. Therefore K/V head IDs should not be directly equated with
Q head IDs.

## Headline Table

| model | Q best | K best | V best |
| --- | ---: | ---: | ---: |
| Mistral-7B base | L21/H18 `0.1018` | L24/H7 `0.0723` | **L19/H7 `0.4261`** |
| Mistral-7B instruct | L21/H18 `0.0938` | L24/H7 `0.0616` | **L19/H7 `0.3930`** |
| Llama-3-8B base | L19/H30 `0.2031` | L19/H7 `0.3712` | **L19/H7 `0.5605`** |
| Llama-3-8B instruct | L19/H30 `0.1791` | L19/H7 `0.3500` | **L19/H7 `0.5456`** |
| Gemma-2-2B base | L19/H4 `0.0980` | L21/H1 `0.0798` | **L19/H2 `0.2557`** |
| Gemma-2-2B-it | L19/H4 `0.0489` | L25/H0 `0.0380` | **L19/H2 `0.1319`** |

## Residual/Input Baseline

After the matched Q/K/V pass, a `--activation-space resid_pre` baseline was run
with the same n300/class CodeXGLUE JSON, `pool_last_k=5`, and 64-token cap. This
captures the input to the attention projection path before Q/K/V projection.
The stored `H0` is a pseudo-head placeholder for the full residual/input vector,
not an attention head.

| model | Q best | V best | resid-pre best | V - resid-pre |
| --- | ---: | ---: | ---: | ---: |
| Mistral-7B base | L21/H18 `0.1018` | **L19/H7 `0.4261`** | L21/H0 `0.1085` | `0.3176` |
| Mistral-7B instruct | L21/H18 `0.0938` | **L19/H7 `0.3930`** | L21/H0 `0.0956` | `0.2973` |
| Llama-3-8B base | L19/H30 `0.2031` | **L19/H7 `0.5605`** | L20/H0 `0.1452` | `0.4153` |
| Llama-3-8B instruct | L19/H30 `0.1791` | **L19/H7 `0.5456`** | L20/H0 `0.1242` | `0.4215` |
| Gemma-2-2B base | L19/H4 `0.0980` | **L19/H2 `0.2557`** | L25/H0 `0.0911` | `0.1645` |
| Gemma-2-2B-it | L19/H4 `0.0489` | **L19/H2 `0.1319`** | L25/H0 `0.0611` | `0.0707` |

The residual/input baseline is not empty. It becomes readable in the same
late-ish region for Mistral and Llama (`L21` and `L20`) and at the final Gemma
layer (`L25`). However, it is much smaller than V-space in every row. The
residual-to-V ratio ranges from about `0.23` to `0.46`, with the largest gaps in
Mistral and Llama.

## Observations

The n300 result strongly reproduces the earlier n50 V-space pilot. In all four
Mistral/Llama rows, V-space peaks at the same late KV head:

```text
Mistral/Llama V-space: L19/H7
```

Gemma also keeps the same late V-space row across base and instruction-tuned
variants:

```text
Gemma 2 2B V-space: L19/H2
```

The absolute V-space silhouette is much larger than the matched Q-space and
K-space silhouette for every model. The contrast is especially large for
Mistral, where K-space is weak but V-space is strong. Llama 3 shows a staged
profile: Q is readable, K is stronger, and V is strongest. Gemma follows the
same ordering in a weaker form, with instruction tuning reducing all three
spaces.

The residual/input baseline adds an important constraint. Code-language
identity is already somewhat readable before Q/K/V projection, so V-space is
not creating the task variable from nothing. But the V-space separation is far
larger than the residual/input baseline, so the strongest V result is not well
explained as a simple copy of pre-existing residual geometry.

This does not mean that Q-space was irrelevant. The earlier Q-space sweep showed
that code-language identity is linearly readable from late Q vectors. This
follow-up says that, under the capped CodeXGLUE condition, the cleanest raw
cosine manifold separation lives in V-space.

## Interpretation

A cautious interpretation is:

```text
For capped CodeXGLUE code-language routing, Q-space exposes a late query-side
readout, but V-space carries a much cleaner late language-evidence geometry.
```

This is consistent with the task structure. Code-language labels can be
identified from accumulated lexical and syntactic evidence in the snippet tail.
The V projection is a plausible place for that evidence to become more directly
organized than in the query posture itself. K-space can also be strong,
especially in Llama 3, but it is not uniformly as separated as V-space.

The `resid_pre` pass refines that interpretation:

```text
late residual/input states already contain code-language information, but the
V projection appears to concentrate or reorganize it into a much cleaner raw
cosine geometry.
```

This still does not identify a causal mechanism. The comparison only says that
the V-space readout is stronger than the pre-projection baseline under the same
capped, pooled-tail condition.

The most striking part is the layer recurrence. Q-space already placed the
CodeXGLUE readout late. K/V do not move it earlier; instead, V-space concentrates
the strongest separation at late layer 19 across all six rows, with
family-specific KV heads.

## Q-to-KV Group Mapping

The compact CSV includes GQA-aware mapping columns:

```text
q_to_kv_group = floor(q_best_head * kv_head_count / query_heads)
q_group_matches_k
q_group_matches_v
```

This is important because Q and K/V head IDs are not directly comparable under
GQA. In this CodeXGLUE pass, the group mapping sharpens the family split:

| model family | Q best | Q-to-KV group | K best | V best | group match |
| --- | ---: | ---: | ---: | ---: | --- |
| Mistral base/IT | L21/H18 | 4 | L24/H7 | L19/H7 | no |
| Llama 3 base/IT | L19/H30 | 7 | L19/H7 | L19/H7 | K yes, V yes |
| Gemma 2 2B base/IT | L19/H4 | 2 | L21/H1 or L25/H0 | L19/H2 | V yes |

So the late CodeXGLUE V-space result is not just a generic "V is stronger"
statement. Llama 3 aligns Q, K, and V through the same GQA group, while Mistral
shows a strong V-space row in a different KV group from the best Q row. Gemma's
weak Q row still maps to the V-space group. These differences are good targets
for later causal or residual-baseline checks.

## Caveats

- This is an n300/class follow-up, not the original n1000/class Q-space matrix.
- It is single-seed.
- It is conditioned on `--max-token-length 64`; long uncapped code may change the
  depth profile.
- It is pre-RoPE only for K/V. The current CLI supports post-RoPE capture for
  Q-space, while this K/V comparison uses projection outputs before positional
  rotation.
- `resid_pre` uses `H0` only as a pseudo-head placeholder for the full
  residual/input vector. It should not be compared as a real head identity.
- These are geometric readouts, not causal assignments. Ablation is still needed
  before claiming that a V head causes downstream code-language behavior.

## Updated Research Direction

The next useful checks are:

- repeat the Q/K/V comparison at n1000 or with a denser machine;
- relax the 64-token cap and see whether V remains the dominant space;
- run the same residual/input baseline with relaxed token caps and dataset
  seeds to check whether the V-minus-residual gap is stable;
- extend the geometry audit to matched Q/K/V rows, especially Llama 3 where K and
  V are both strong;
- add causal ablations for late CodeXGLUE Q/K/V rows rather than only Q heads.
