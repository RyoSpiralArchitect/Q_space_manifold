# Natural-Task Q/K/V Activation-Space Sweep

Date: 2026-06-28

This note records the first matched Q/K/V activation-space pass over the three
natural-language tasks that were previously studied mostly through Q-space:
SUBJ, prompted SST-2, and TREC coarse question type.

The goal is deliberately narrow:

```text
Do the natural-language readouts that appear in Q-space also appear in K or V,
and does the dominant activation space depend on the task?
```

## Artifacts

Compact tracked artifact:

- `examples/natural_tasks_qkv_activation_space_n300/qkv_best_layer_heads.csv`

Large full outputs, including vector bundles, remain outside the repo:

- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/subj/q`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/subj/k`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/subj/v`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/sst2_prompted/q`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/sst2_prompted/k`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/sst2_prompted/v`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/trec_coarse/q`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/trec_coarse/k`
- `~/q_space_runs/natural_tasks_qkv_n300_pre_rope_pool1/trec_coarse/v`

## Settings

- datasets: SUBJ, prompted SST-2, and TREC coarse question type
- sample targets: `300` rows per class
- realized samples: SUBJ `600`, prompted SST-2 `600`, TREC coarse `1586`
- TREC imbalance caveat: the ABBR class has fewer than `300` available rows, so
  this TREC pass is not class-balanced
- models: Mistral-7B base/instruct, Llama-3-8B base/instruct, Gemma-2-2B
  base/instruct, all current `mlx-community/*-4bit` checkpoints
- backend: MLX
- activation spaces: `--activation-space q`, `k`, and `v`
- capture stage: pre-RoPE projection outputs
- pooling: `--pool-last-k 1`
- token storage: `--max-stored-tokens 5`, `--stored-token-selection tail`,
  `--token-q-storage-dtype float16`
- plots: disabled with `--no-plots`

For K/V rows, the reported head count is the KV head count under GQA, not the
full Q head count. Mistral and Llama 3 expose 8 K/V heads here; Gemma 2 2B
exposes 4 K/V heads. K/V head IDs therefore should not be directly equated with
Q head IDs.

## SUBJ Headline Table

| model | Q best | K best | V best |
| --- | ---: | ---: | ---: |
| Mistral-7B base | L10/H6 `0.2365` | L16/H6 `0.2877` | L16/H6 `0.2631` |
| Mistral-7B instruct | L11/H22 `0.2295` | L16/H6 `0.2821` | L16/H6 `0.2534` |
| Llama-3-8B base | L10/H3 `0.1856` | L14/H7 `0.2447` | L16/H1 `0.2230` |
| Llama-3-8B instruct | L20/H31 `0.2288` | L20/H7 `0.2441` | L16/H1 `0.2522` |
| Gemma-2-2B base | L21/H4 `0.1738` | L16/H0 `0.2134` | L16/H0 `0.2039` |
| Gemma-2-2B-it | L1/H0 `0.0367` | L14/H2 `0.0374` | L14/H2 `0.0287` |

## Prompted SST-2 Headline Table

| model | Q best | K best | V best |
| --- | ---: | ---: | ---: |
| Mistral-7B base | L28/H25 `0.0653` | L25/H5 `0.0502` | L25/H0 `0.1054` |
| Mistral-7B instruct | L23/H30 `0.1663` | L29/H1 `0.1241` | L22/H5 `0.2683` |
| Llama-3-8B base | L20/H24 `0.1201` | L20/H2 `0.1080` | L25/H4 `0.1661` |
| Llama-3-8B instruct | L18/H28 `0.2256` | L20/H2 `0.1932` | L25/H4 `0.2794` |
| Gemma-2-2B base | L23/H5 `0.0342` | L22/H0 `0.0341` | L22/H0 `0.0648` |
| Gemma-2-2B-it | L1/H6 `0.0168` | L12/H0 `0.0278` | L1/H0 `0.0146` |

## TREC Coarse Headline Table

| model | Q best | K best | V best |
| --- | ---: | ---: | ---: |
| Mistral-7B base | L17/H26 `0.1132` | L19/H6 `0.0936` | L21/H7 `0.0899` |
| Mistral-7B instruct | L17/H26 `0.0924` | L19/H6 `0.0951` | L16/H5 `0.0794` |
| Llama-3-8B base | L17/H24 `0.0659` | L17/H1 `0.0685` | L17/H1 `0.1052` |
| Llama-3-8B instruct | L18/H16 `0.0533` | L17/H1 `0.0438` | L17/H1 `0.0907` |
| Gemma-2-2B base | L17/H3 `0.0856` | L17/H3 `0.0929` | L19/H3 `0.1021` |
| Gemma-2-2B-it | L6/H6 `0.0021` | L5/H1 `0.0178` | L5/H1 `0.0117` |

## Q-to-KV Group Mapping

For GQA models, Q head IDs and K/V head IDs live in different index spaces.
Mistral and Llama 3 expose 32 Q heads but 8 K/V heads; Gemma 2 2B exposes 8 Q
heads but 4 K/V heads. The compact CSV therefore includes:

```text
q_to_kv_group = floor(q_best_head * kv_head_count / query_heads)
q_group_matches_k
q_group_matches_v
```

This matters immediately. The SUBJ Mistral K/V rows both land at `H6`, but the
Q best heads do not map to KV group 6. In contrast, TREC Mistral has a cleaner
GQA-aligned pattern:

| task/model | Q best | Q-to-KV group | K best | V best | group match |
| --- | ---: | ---: | ---: | ---: | --- |
| TREC / Mistral base | L17/H26 | 6 | L19/H6 | L21/H7 | K yes, V no |
| TREC / Mistral instruct | L17/H26 | 6 | L19/H6 | L16/H5 | K yes, V no |

So the TREC Mistral Q/K relationship is more suggestive than a raw head-number
comparison: the answer-type Q readout and K-space address structure may be
connected through the same GQA group. This remains a geometric observation, not
a causal claim.

## Observations

SUBJ is not Q-only. In most non-Gemma-IT rows, K-space and V-space are as strong
as or stronger than Q-space. Mistral is especially clean: both base and instruct
move from an early/mid Q readout into a stable K/V `L16/H6` row. Llama and Gemma
base also show mid-layer K/V structure, while Gemma 2 2B-it remains weak across
all three spaces.

Prompted SST-2 is the strongest V-space surprise in this pass. For Mistral-IT
and Llama3-IT, V-space is stronger than Q-space or K-space. This suggests that
under an explicit sentiment prompt, the most compact raw cosine geometry can sit
closer to value/readout evidence than to the query posture alone.

TREC coarse is more distributed. Q, K, and V all carry modest answer-type
geometry. Mistral has a readable Q/K/V sequence around mid-to-late layers,
Llama's strongest TREC row is V-space in this n300 pass, and Gemma base is
strongest in V-space while Gemma-IT is again weakly localized. The TREC result
should be read with the ABBR imbalance caveat.

## Interpretation

The main update is that the project should no longer frame the phenomenon as a
Q-only geometry. Q-space remains useful as a window into query posture, but K and
V expose complementary structure:

```text
Q-space: query / retrieval posture
K-space: addressable task or answer-type structure
V-space: accumulated evidence or readout-bearing geometry
```

This is a working decomposition, not a mechanistic claim. It is especially
plausible for prompted SST-2 and CodeXGLUE, where V-space becomes much cleaner
than Q-space. SUBJ suggests a different picture: stance geometry is visible in
Q, but the mid-layer K/V spaces can be even more separated.

## Relation to the CodeXGLUE Q/K/V Pass

The earlier CodeXGLUE n300 Q/K/V pass found a very strong late V-space readout
for code-language identity. This natural-task pass shows that V-space dominance
is not exclusive to code, but it is also not universal. It depends on the task:

- CodeXGLUE: V-space dominates strongly under the 64-token cap.
- prompted SST-2: V-space dominates for Mistral/Llama instruction-tuned rows.
- SUBJ: K/V are often stronger than Q, but the pattern is mid-layer stance-like.
- TREC: Q/K/V are all modest, with V sometimes strongest.

A cautious synthesis is:

```text
activation-space choice changes which part of the task variable is most visible;
Q-space is a posture probe, while K/V can expose address and evidence geometry.
```

## Caveats

- This is n300/class reconnaissance, not the n1000 headline matrix.
- It is single-seed.
- It is pre-RoPE only.
- It uses `pool_last_k=1`; pooling sweeps may move the strongest K/V rows.
- It does not include label-permutation or linear-probe reruns.
- It is geometric, not causal. Ablation is still required before assigning a
  projection-space row a downstream role.

## Next Checks

- repeat selected Q/K/V rows at n1000;
- run post-RoPE-compatible Q checks beside the pre-RoPE K/V pass;
- compare selected rows against `--activation-space resid_pre` to see whether
  the task variable is already readable before Q/K/V projection;
- add geometry audits for the strongest natural-task K/V rows, especially SUBJ
  Mistral `K/V L16/H6` and prompted SST-2 Mistral/Llama V rows;
- test whether Gemma 2 2B-it is genuinely diffuse by moving to larger Gemma
  checkpoints or multi-head/multi-layer summaries;
- eventually pair Q/K/V geometry with causal ablation rather than treating
  silhouette as mechanism.
