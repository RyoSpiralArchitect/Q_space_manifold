# N=1000/Class 3D Base-vs-Instruct Matrix

Date: 2026-05-25

This note records the first larger 3D plotting pass across the current
Mistral, Llama 3, and Gemma 2 2B model matrix.

The run is no longer a toy-scale probe. Each model sees `1000` examples per
class, or `2000` rows per model. With six model configurations, each dataset
produces `12000` model-sample Q observations before layer/head expansion. In
conversation shorthand this was called the "n=6000" run because it is six model
configs at `1000/class`; the concrete per-model sample count in the CSVs is
`2000`.

Raw compact tables are in:

- `examples/n1000_3d_matrix/subj_n1000_3d_batch_top_layer_heads.csv`
- `examples/n1000_3d_matrix/subj_n1000_3d_batch_model_summary.csv`
- `examples/n1000_3d_matrix/sst2_prompted_n1000_3d_pool_last_k_sweep_summary.csv`
- `examples/n1000_3d_matrix/sst2_prompted_n1000_3d_pool_last_k_sweep_manifest.json`

Large full outputs, including per-model 3D plots and `q_space_vectors.npz`, were
left under `/tmp` and are not tracked in the repository.

## Common Settings

- backend: MLX
- model precision: current `mlx-community/*-4bit` checkpoints
- Q capture: pre-RoPE Q projection output
- projection: PCA
- plots: `--plot-3d`
- plot density control: `--plot-sample-limit 200`
- final-token robustness: `--pool-last-k-sweep 1,3,5` for prompted SST-2
- token handling: `--drop-special-tokens`, `--flow-start-token-index 1`
- controls: label permutation, high-dimensional flow metrics, projection
  diagnostics, linear probe, and head similarity
- large-run speed knob: `--linear-probe-permutation-n 0`

`--plot-sample-limit 200` only sparsifies all-sample trajectory and token-flow
plots. Silhouette scores, probes, CSV metrics, and summaries still use the full
captured sample.

## SUBJ Base-vs-Instruct

Dataset:

- source: `SetFit/subj`
- split: `train`
- sample: `1000 subjective + 1000 objective`

Best layer/head by high-dimensional cosine silhouette:

| model | best layer/head | relative depth | silhouette |
| --- | ---: | ---: | ---: |
| Mistral-7B base | L10/H6 | 0.323 | 0.2270 |
| Mistral-7B instruct | L7/H15 | 0.226 | 0.2154 |
| Llama-3-8B base | L11/H6 | 0.355 | 0.1758 |
| Llama-3-8B instruct | L20/H31 | 0.645 | 0.2187 |
| Gemma-2-2B base | L21/H4 | 0.840 | 0.1663 |
| Gemma-2-2B-it | L1/H0 | 0.040 | 0.0345 |

### SUBJ Interpretation

The qualitative base-vs-instruct story survives the larger sample.

**Mistral remains stable but less single-head pinned.** The base model still
peaks around early/mid depth at L10/H6. The instruction-tuned model's top head
moves to L7/H15 at this sample size, but the earlier L11/H22 head remains near
the top. The important pattern is therefore not one magic head, but a stable
early/mid stance-separating band.

**Llama 3 still migrates deeper under instruction tuning.** The base model peaks
near L11/H6, while the instruction-tuned model peaks at L20/H31. This preserves
the earlier observation that Llama 3's strongest SUBJ stance geometry appears
later after instruction tuning.

**Gemma 2 2B is the sharp contrast case.** The base model has a late usable
single-head axis at L21/H4. The instruction-tuned model collapses to a weak
early maximum at L1/H0, with all top scores near `0.02-0.03`. This supports the
more careful reading that Gemma 2 2B-it is not "signal-free"; rather, the signal
is not cleanly localized in one head under this probe.

## Prompted SST-2 Base-vs-Instruct

Dataset:

- source: SST-2
- split: `train`
- sample: `1000 negative + 1000 positive`
- template: `Review: {text}\nSentiment:`

Best layer/head by high-dimensional cosine silhouette:

| model | pool_last_k | best layer/head | relative depth | silhouette |
| --- | ---: | ---: | ---: | ---: |
| Mistral-7B base | 1 | L28/H25 | 0.903 | 0.0745 |
| Mistral-7B base | 3 | L10/H21 | 0.323 | 0.0718 |
| Mistral-7B base | 5 | L10/H21 | 0.323 | 0.0978 |
| Mistral-7B instruct | 1 | L23/H30 | 0.742 | 0.1726 |
| Mistral-7B instruct | 3 | L20/H18 | 0.645 | 0.1082 |
| Mistral-7B instruct | 5 | L10/H21 | 0.323 | 0.0974 |
| Llama-3-8B base | 1 | L20/H24 | 0.645 | 0.1205 |
| Llama-3-8B base | 3 | L20/H24 | 0.645 | 0.0834 |
| Llama-3-8B base | 5 | L8/H15 | 0.258 | 0.0766 |
| Llama-3-8B instruct | 1 | L18/H28 | 0.581 | 0.2246 |
| Llama-3-8B instruct | 3 | L20/H9 | 0.645 | 0.1376 |
| Llama-3-8B instruct | 5 | L8/H15 | 0.258 | 0.1041 |
| Gemma-2-2B base | 1 | L22/H1 | 0.880 | 0.0333 |
| Gemma-2-2B base | 3 | L12/H4 | 0.480 | 0.0439 |
| Gemma-2-2B base | 5 | L12/H4 | 0.480 | 0.0587 |
| Gemma-2-2B-it | 1 | L1/H6 | 0.040 | 0.0127 |
| Gemma-2-2B-it | 3 | L12/H1 | 0.480 | 0.0193 |
| Gemma-2-2B-it | 5 | L12/H3 | 0.480 | 0.0266 |

### SST-2 Interpretation

Prompted SST-2 is not as clean as SUBJ, but it is structured rather than flat.

**Instruction tuning amplifies explicit sentiment stance in Mistral and Llama
3.** With `pool_last_k=1`, Mistral-7B-Instruct rises to L23/H30 at `0.1726`,
while Llama-3-8B-Instruct rises to L18/H28 at `0.2246`. These are substantially
stronger than the matching base checkpoints and much stronger than Gemma 2
2B-it.

**Pooling changes the readout position.** `pool_last_k=1` emphasizes the final
task cue position in `Review: ... Sentiment:`. Pooling over `3` or `5` tokens
mixes the cue with nearby text positions, and the best layer/head can move from
late to mid or early/mid depth. This is not a failure of the probe; it says the
Q-space stance is position-sensitive.

**Llama 3's prompted IT signal remains strongest.** The exact best head changes
from the earlier 100/class run, but the family-level result survives: Llama 3
IT has a strong late/mid prompted polarity head, with L20/H9 reappearing when
`pool_last_k=3`.

**Gemma 2 2B stays weak, especially after instruction tuning.** Gemma base
improves slightly as pooling widens, reaching L12/H4 at `0.0587`.
Gemma 2 2B-it stays near zero across the sweep. Again, this should be phrased
as weak single-head localization, not proof that the model lacks polarity
information.

## Cross-Dataset Pattern

The larger run sharpens the matrix:

```text
SUBJ:
  Mistral stable early/mid band
  Llama 3 deeper after instruction tuning
  Gemma 2 2B base late, Gemma 2 2B-it flat/diffuse

Prompted SST-2:
  Mistral IT and Llama 3 IT light up explicit sentiment-query heads
  pooling changes the measured stance position
  Gemma 2 2B-it remains the weak/localization-negative contrast
```

This makes the probe look less like a projection artifact. A pure artifact
would be more likely to collapse all models and tasks into similar-looking
peaks. Instead, the peak location and strength depend on family, tuning state,
task framing, and token pooling.

## Reproduction Commands

SUBJ:

```bash
./q_space_manifold_monolith.py \
  --dataset-source subj \
  --dataset-split train \
  --samples-per-class 1000 \
  --batch-models mistral_base=mlx:mlx-community/Mistral-7B-v0.3-4bit,mistral_it=mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit,llama3_base=mlx:mlx-community/Meta-Llama-3-8B-4bit,llama3_it=mlx:mlx-community/Meta-Llama-3-8B-Instruct-4bit,gemma2_2b_base=mlx:mlx-community/gemma-2-2b-4bit,gemma2_2b_it=mlx:mlx-community/gemma-2-2b-it-4bit \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --projection pca \
  --detail-best-layer-head \
  --label-permutation-n 200 \
  --high-d-flow-metrics \
  --projection-diagnostics \
  --probe-linear \
  --linear-probe-permutation-n 0 \
  --head-similarity \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --plot-3d \
  --plot-sample-limit 200 \
  --output-dir /tmp/q_space_subj_n1000_base_vs_it_3d
```

Prompted SST-2:

```bash
./q_space_manifold_monolith.py \
  --dataset-source sst2 \
  --dataset-split train \
  --samples-per-class 1000 \
  --text-template $'Review: {text}\nSentiment:' \
  --batch-models mistral_base=mlx:mlx-community/Mistral-7B-v0.3-4bit,mistral_it=mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit,llama3_base=mlx:mlx-community/Meta-Llama-3-8B-4bit,llama3_it=mlx:mlx-community/Meta-Llama-3-8B-Instruct-4bit,gemma2_2b_base=mlx:mlx-community/gemma-2-2b-4bit,gemma2_2b_it=mlx:mlx-community/gemma-2-2b-it-4bit \
  --target-layer-fraction 0.35 \
  --target-head 4 \
  --pool-last-k-sweep 1,3,5 \
  --projection pca \
  --detail-best-layer-head \
  --label-permutation-n 200 \
  --high-d-flow-metrics \
  --projection-diagnostics \
  --probe-linear \
  --linear-probe-permutation-n 0 \
  --head-similarity \
  --drop-special-tokens \
  --flow-start-token-index 1 \
  --plot-3d \
  --plot-sample-limit 200 \
  --output-dir /tmp/q_space_sst2_prompted_n1000_base_vs_it_3d
```

## Next Matrix: Dense Same-Family Check

Before the dense run, the next natural move is to compare the strongest 4bit
heads under post-RoPE Q capture:

```bash
--q-capture-stage post-rope
```

If the same heads or bands remain separable after rotary position phase is
applied, the stance-routing interpretation gets stronger. If the peaks move or
diffuse, then pre-RoPE and post-RoPE should be treated as distinct experimental
surfaces.

After that, repeat this matrix on a larger MacBook Pro with dense checkpoints
from the same families. That gives a clean 12-cell comparison:

```text
3 families x 2 tuning states x 2 task framings
```

The key question is whether the current 4bit findings are architecture-level
and tuning-level effects, or whether some of the observed flattening and peak
movement is partly quantization-sensitive.

The dense run should reuse the same analysis flags, including:

```text
--samples-per-class 1000
--projection pca
--detail-best-layer-head
--label-permutation-n 200
--probe-linear
--linear-probe-permutation-n 0
--head-similarity
--plot-3d
--plot-sample-limit 200
--drop-special-tokens
--flow-start-token-index 1
```

Run both default pre-RoPE and post-RoPE variants when feasible:

```text
--q-capture-stage pre-rope
--q-capture-stage post-rope
```

The model IDs should be resolved on the target machine before download. The
important thing is to keep family, tuning state, dataset, prompt template, and
pooling identical to this run.

## Caveats

- Silhouette is still correlational and is computed in original Q-space, not in
  the 3D projection.
- 3D plots reduce visual overplotting, but they do not change the metric.
- Prompted SST-2 measures a classification stance induced by a task cue, not
  generic review semantics.
- `pool_last_k` changes the token position being measured; it is an
  experimental factor, not just a smoothing trick.
- Dense follow-up is required before claiming the pattern is independent of
  quantization.
