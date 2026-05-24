# Q-space Manifold

Exploratory tooling for probing the geometry of transformer Query vectors.

The central question is not only *what meaning is represented*, but how a model
forms a **stance for searching meaning**: which attention heads become sensitive
to sentiment, subjectivity, factual framing, or discourse posture, and how those
query directions evolve across layers and tokens.

This repository currently contains a self-contained monolithic probe:

```text
q_space_manifold_monolith.py
```

It extracts per-layer/per-head Q vectors, compares head and layer separation,
projects Q-space with PCA or UMAP, traces token-level Q-flow, and runs early
controls such as random-label baselines, projection diagnostics, linear probes,
and head-to-head RSA/CKA.

## Current Hypothesis

Early results suggest that transformer layers may contain a **model-family
dependent stance phase** where question or discourse stance becomes sharply
organized. The phase is not necessarily located at the same relative depth in
every architecture:

- **stance formation**: subjective, objective, positive, negative, or factual
  framing becomes separable in Q-space;
- **query routing**: specific heads appear to define attractor-like routing
  directions for what the next attention operation will seek;
- **discourse framing**: token-level Q-flow often looks less like a random walk
  and more like a structured path from initialization into local exploration and
  drift.

The current working hypothesis is:

```text
instruction-tuned decoders can develop stance-separating Q-space heads,
but the layer where that phase appears shifts by model family and may be
concentrated in one head or distributed across weaker heads.
```

This is still exploratory. The current evidence is geometric and predictive,
not causal. Causal ablation and post-RoPE Q capture are planned follow-ups.

## Research Notes

- [Base vs instruction-tuned SUBJ scan](docs/research_notes/base_vs_instruct_subj.md):
  Mistral appears stable, Llama 3 migrates deeper, and Gemma 2 2B flattens or
  diffuses its single-head Q-space stance axis after instruction tuning.
- [SST-2 pool-last-k sweep](docs/research_notes/sst2_pool_last_k_sweep.md):
  sentiment polarity is weaker than SUBJ subjectivity/objectivity in single-head
  Q-space, but the signal does not disappear when pooling over the last `1,3,5`
  tokens.
- [SST-2 base-vs-instruct and prompt framing](docs/research_notes/sst2_base_vs_prompted.md):
  naked SST-2 stays weak, but `Review: {text}\nSentiment:` strongly lights up
  late sentiment-query heads in Mistral and Llama 3.

## What Is Being Measured?

For GPT-2-style models, the script captures the Q slice from fused QKV
projection layers such as `c_attn`.

For Llama/Mistral/Gemma-style RoPE models, the script captures the output of
`q_proj` / `wq`. This means the default measurement is:

```text
pre-RoPE Q projection output
```

That is intentional for the current probe: it emphasizes the content-dependent
query direction before rotary positional phase is applied. Metadata and plot
titles now record this as `q_capture_stage:
q_projection_output_pre_attention_position_rotation`.

Interpretation:

```text
pre-RoPE Q  = content / stance routing vector
post-RoPE Q = content + positional phase query used for attention scoring
```

## Representative Run

The current representative scan used:

- dataset: `SetFit/subj`
- split: `train`
- samples: `100 subjective + 100 objective`
- model: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- backend: MLX
- projection: PCA
- target depth: `round(0.35 * (n_layers - 1)) = layer 11`

Summary:

```text
best layer/head: layer 11, head 22
relative depth: 0.3548
silhouette cosine: 0.2290
linear probe leave-one-out accuracy:
  target L11/H4  = 0.84
  best   L11/H22 = 0.86
random-label p-value: 0.0099
```

## First Cross-Model SUBJ Scan

A first 3-model scan used the same `SetFit/subj` sample
(`100 subjective + 100 objective`) and the same PCA / random-label /
linear-probe settings across:

- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit`
- `mlx-community/gemma-2-2b-it-4bit`

Best layer/head by high-dimensional cosine silhouette:

| model | best layer/head | relative depth | silhouette |
| --- | ---: | ---: | ---: |
| Mistral-7B-Instruct | L11/H22 | 0.355 | 0.2290 |
| Llama-3-8B-Instruct | L20/H31 | 0.645 | 0.2276 |
| Gemma-2-2B-it | L12/H1 | 0.480 | 0.0410 |

Interpretation:

- **Mistral** shows a broad early/mid stance-separation band, with several
  top heads between relative depth `0.23` and `0.58`.
- **Llama 3** reaches a similar maximum silhouette, but its strongest head is
  later, around relative depth `0.65`, with several secondary heads in the
  mid-to-late range.
- **Gemma 2 2B** is much weaker as a single-head Q-space separation signal:
  its top 10 heads stay near `0.03-0.04`. However, linear probes still perform
  above random-label controls, suggesting weak or distributed signal rather
  than total absence.

Linear probe leave-one-out accuracy for the best overall head:

| model | best head | LOO accuracy | random-label mean |
| --- | ---: | ---: | ---: |
| Mistral-7B-Instruct | L11/H22 | 0.86 | 0.487 |
| Llama-3-8B-Instruct | L20/H31 | 0.935 | 0.497 |
| Gemma-2-2B-it | L12/H1 | 0.64 | 0.494 |

This updates the hypothesis from "a generic middle-layer phase" to a more
specific one:

```text
Mistral-like architectures may expose an earlier concentrated Q-space stance
phase, Llama 3 may expose a later concentrated phase, and smaller Gemma models
may express the signal more weakly or more diffusely.
```

The raw summary files are in `examples/subj_3models/`.

## Head Similarity: Specialization vs Redundancy

The next diagnostic asks whether the best stance-separating heads are isolated
axes, semi-specialized clusters, or mostly redundant with the rest of the layer.
For each model's best layer, the probe computes pairwise linear CKA and RSA
correlation over head-level Q-space geometry.

| model | best head | mean off-diagonal CKA | mean off-diagonal RSA | nearest heads by CKA |
| --- | ---: | ---: | ---: | --- |
| Mistral-7B-Instruct | L11/H22 | 0.577 | 0.638 | H19, H23, H20 |
| Llama-3-8B-Instruct | L20/H31 | 0.655 | 0.772 | H30, H20, H23 |
| Gemma-2-2B-it | L12/H1 | 0.804 | 0.877 | H3, H0, H5 |

Interpretation:

- **Mistral** looks like an early/mid semi-specialized stance cluster: H22 is
  not isolated, but the layer still has enough head diversity for a few heads
  to form a clear subjectivity/objectivity axis.
- **Llama 3** looks like a later semi-specialized stance cluster: the best head
  appears in a more redundant late-layer neighborhood, but still separates SUBJ
  about as strongly as Mistral.
- **Gemma 2 2B** has weak single-head Q-space geometry and high head
  redundancy. The signal is not absent, but it appears less localized to a
  distinct head and is more plausibly distributed across similar heads.

Representative head-similarity pair tables are included in
`examples/subj_3models/`.

Llama 3 best-layer CKA/RSA:

![Llama 3 head CKA heatmap](assets/llama3_head_cka_heatmap_layer_20.png)

![Llama 3 head RSA heatmap](assets/llama3_head_rsa_heatmap_layer_20.png)

Gemma 2 2B best-layer CKA/RSA:

![Gemma 2 2B head CKA heatmap](assets/gemma2_2b_head_cka_heatmap_layer_12.png)

![Gemma 2 2B head RSA heatmap](assets/gemma2_2b_head_rsa_heatmap_layer_12.png)

### Mistral vs Llama vs Gemma Heatmaps

Mistral:

![Mistral layer/head heatmap](assets/layer_head_separability_heatmap.png)

Llama 3:

![Llama 3 layer/head heatmap](assets/llama3_it_layer_head_separability_heatmap.png)

Gemma 2 2B:

![Gemma 2 2B layer/head heatmap](assets/gemma2_2b_it_layer_head_separability_heatmap.png)

## Representative Mistral Detail Plots

### Layer x Head Separability

![Layer/head separability heatmap](assets/layer_head_separability_heatmap.png)

### Head Manifolds at Layer 11

![Head manifolds](assets/head_manifolds_layer_11.png)

### 3D Layer Trajectory for the Best Head

![3D layer trajectory](assets/layer_trajectory_3d_head_22_focus_layer_11.png)

### 3D Token-Level Q-Flow

![3D token Q-flow](assets/query_flow_3d_layer_11_head_22_all.png)

### Head Similarity: CKA and RSA

These matrices ask whether strong heads are redundant copies or distinct axes.

![Head CKA heatmap](assets/head_cka_heatmap_layer_11.png)

![Head RSA heatmap](assets/head_rsa_heatmap_layer_11.png)

## Quick Start

### Torch backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python q_space_manifold_monolith.py \
  --backend torch \
  --model-path gpt2 \
  --target-layer 6 \
  --target-head 3 \
  --projection pca \
  --output-dir /tmp/q_space_gpt2_probe
```

### MLX backend

On Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mlx.txt

python q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source subj \
  --dataset-split train \
  --samples-per-class 100 \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --projection pca \
  --detail-best-layer-head \
  --label-permutation-n 100 \
  --high-d-flow-metrics \
  --projection-diagnostics \
  --probe-linear \
  --head-similarity \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --output-dir /tmp/q_space_phase_scan_subj
```

For larger runs, keep the metrics on the full sample but make the plots readable
with plot-only downsampling and optional 3D views:

```bash
python q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source subj \
  --samples-per-class 1000 \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --projection pca \
  --detail-best-layer-head \
  --plot-3d \
  --plot-sample-limit 200 \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --output-dir /tmp/q_space_subj_n1000_plots
```

`--plot-sample-limit` affects only all-sample trajectory and token-flow plots;
CSV metrics, silhouettes, probes, and summaries still use the full captured
dataset.

## Cross-Model Phase Scan

Use `--batch-models` to compare models with the same dataset and metrics.

```bash
python q_space_manifold_monolith.py \
  --dataset-source subj \
  --dataset-split train \
  --samples-per-class 100 \
  --batch-models \
mistral_it=mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit,llama3_it=mlx:mlx-community/Meta-Llama-3-8B-Instruct-4bit,gemma2_2b_it=mlx:mlx-community/gemma-2-2b-it-4bit \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --projection pca \
  --detail-best-layer-head \
  --label-permutation-n 100 \
  --high-d-flow-metrics \
  --projection-diagnostics \
  --probe-linear \
  --head-similarity \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --output-dir /tmp/q_space_phase_scan_subj_3models
```

Batch outputs:

```text
batch_model_summary.csv
batch_top_layer_heads.csv
batch_manifest.json
```

The main comparison field is `best_layer_relative_depth`, which allows models
with different layer counts to be compared on the same normalized axis.

## Pooling Robustness

`pool_last_k=1` measures the final token's Q vector. For questions, this is
often the `?` token; for SST-2/SUBJ declarative sentences, it is often a final
punctuation token. To test whether the effect is robust to this choice:

```bash
python q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source subj \
  --samples-per-class 100 \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --pool-last-k-sweep 1,3,5 \
  --projection pca \
  --detail-best-layer-head \
  --head-similarity \
  --no-plots \
  --output-dir /tmp/q_space_pool_sweep
```

The sweep reuses captured token Q tensors per model and writes:

```text
pool_last_k_sweep_summary.csv
pool_last_k_sweep_manifest.json
```

To test a task framing rather than the naked sentence, use `--text-template`:

```bash
python q_space_manifold_monolith.py \
  --backend mlx \
  --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --dataset-source sst2 \
  --samples-per-class 100 \
  --text-template $'Review: {text}\nSentiment:' \
  --pool-last-k-sweep 1,3,5 \
  --projection pca \
  --detail-best-layer-head \
  --head-similarity \
  --no-plots \
  --output-dir /tmp/q_space_prompted_sst2
```

## Outputs

Each run writes:

```text
q_space_vectors.npz
run_metadata.json
analysis_summary.json
dataset_rows.csv
head_scores.csv
layer_head_scores.csv
layer_head_separability_heatmap.png
token_flow_metrics_layer_L_head_H.csv
token_flow_meta_layer_L_head_H.csv
```

Optional outputs:

```text
label_permutation_summary.csv
linear_probe_summary.csv
projection_diagnostics.csv
highd_token_flow_metrics_layer_L_head_H.csv
head_cka_matrix_layer_L.csv
head_rsa_matrix_layer_L.csv
head_similarity_pairs_layer_L.csv
head_cka_heatmap_layer_L.png
head_rsa_heatmap_layer_L.png
layer_trajectory_3d_head_H_focus_layer_L.png
query_flow_3d_layer_L_head_H_all.png
```

## Near-Term Research Directions

- repeat the Mistral/Llama/Gemma scan on larger SUBJ/SST-2 samples;
- test whether Gemma's weaker single-head signal becomes stronger in 9B or
  appears as a multi-head / multi-layer distributed code;
- compare base vs instruction-tuned checkpoints;
- run prompted SST-2 on base checkpoints;
- inspect whether strong heads are redundant via RSA/CKA;
- implement post-RoPE Q capture as an option;
- add causal ablation of candidate heads and measure downstream degradation.

## Caveats

- The current strongest evidence is geometric and predictive, not causal.
- Linear probes can be over-optimistic when sample counts are small.
- PCA/UMAP are visualizations; silhouette is computed in the original Q-space.
- Flow-field curl/divergence summaries are exploratory 2D projection summaries,
  not physical quantities in the original high-dimensional space.
