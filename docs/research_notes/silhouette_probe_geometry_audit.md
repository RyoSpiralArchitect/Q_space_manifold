# Silhouette vs Probe Geometry Audit

Date: 2026-06-16

This note records a targeted follow-up to the recurring observation that some
multi-class Q-space tasks have modest cosine silhouette but high linear probe
accuracy. The question is whether those rows are weak because the task signal is
absent, or because the task variable is readable as a non-spherical or
distributed linear code rather than a compact manifold cluster.

The audit uses existing `q_space_vectors.npz` files only. No new model capture
was run.

## Artifacts

Compact CSVs are tracked under:

- `examples/geometry_audit_silhouette_vs_probe/geometry_audit_summary.csv`
- `examples/geometry_audit_silhouette_vs_probe/trec_mistral_base_pre_l17_h26_pca_probe_curve.csv`
- `examples/geometry_audit_silhouette_vs_probe/codexglue_llama3_base_pre_l19_h30_pca_probe_curve.csv`
- `examples/geometry_audit_silhouette_vs_probe/codexglue_mistral_it_pre_l21_h18_pca_probe_curve.csv`
- `examples/geometry_audit_pre_post_rope/pre_post_geometry_audit_comparison.csv`
- `examples/geometry_audit_pre_post_rope/pca_probe_pre_post_comparison.csv`
- `examples/geometry_audit_pre_post_rope/post_rope_geometry_audit_summary.csv`

The source vectors remain outside the repo under `~/q_space_runs/...` because
the vector bundles are large.

## Method

The new helper is:

```bash
./scripts/q_space_geometry_audit.py \
  --run-dir ~/q_space_runs/.../pool_last_k_5/llama3_base \
  --output-dir ~/q_space_runs/geometry_audit_codexglue_llama3_base_pre_l19_h30
```

If `--layer` and `--head` are omitted, the script reads
`analysis_summary.json` and audits the best layer/head by high-dimensional
cosine silhouette.

For each audited row it writes:

- sampled cosine silhouette;
- raw and global-mean-centered nearest-centroid accuracy;
- within-class distance to class centroid;
- pairwise class-centroid distances;
- kNN accuracy on a stratified sample;
- 5-fold linear Ridge probe accuracy;
- linear probe accuracy after PCA compression;
- one-vs-rest descriptive margins.

The global-mean-centered columns are included because Q vectors can share a
large common direction. If class differences ride on a small residual direction,
raw cosine silhouette can understate linear readability.

## Representative Rows

| row | task | layer/head | source sil | sampled sil | centered sil | nearest centroid | kNN@5 | raw linear CV |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TREC Mistral-base pre | coarse question type | L17/H26 | 0.1019 | 0.0948 | 0.0791 | 0.798 | 0.867 | 0.876 |
| CodeXGLUE Llama3-base pre | code language | L19/H30 | 0.2031 | 0.2044 | 0.2169 | 0.927 | 0.964 | 0.982 |
| CodeXGLUE Mistral-IT pre | code language | L21/H18 | 0.0924 | 0.0895 | 0.1127 | 0.900 | 0.923 | 0.974 |

The clearest split is not simply "low silhouette versus high probe". The rows
show three slightly different regimes:

- **TREC Mistral-base**: modest silhouette and modest centroid accuracy, but
  strong kNN and linear probe. This looks like answer-type information spread
  across several directions rather than a clean compact class blob.
- **CodeXGLUE Llama3-base**: silhouette, nearest centroid, kNN, and probe are
  all strong. This row is closer to a compact, locally coherent code-language
  geometry.
- **CodeXGLUE Mistral-IT**: raw silhouette is low, but nearest centroid, kNN,
  and probe are high. This suggests class directions are readable, while the
  raw cosine manifold is compressed by a strong common component.

## PCA Probe Curve

PCA compression makes the difference sharper:

| row | PCA-4 acc | PCA-8 acc | PCA-16 acc | PCA-32 acc | PCA-64 acc | full-128 acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TREC Mistral-base L17/H26 | 0.484 | 0.679 | 0.785 | 0.833 | 0.862 | 0.876 |
| CodeXGLUE Llama3-base L19/H30 | 0.713 | 0.936 | 0.970 | 0.975 | 0.978 | 0.982 |
| CodeXGLUE Mistral-IT L21/H18 | 0.564 | 0.848 | 0.921 | 0.952 | 0.962 | 0.974 |

TREC needs more dimensions to approach the full probe score. CodeXGLUE becomes
highly readable by 8-16 PCA dimensions. This supports the idea that CodeXGLUE
code-language identity is a stronger low-dimensional linear readout, while TREC
answer type is more distributed.

## Common-Mode Interpretation

The raw centroid-distance ratio is below 1.0 in all three audited rows:

| row | mean within distance | mean centroid distance | raw between/within |
| --- | ---: | ---: | ---: |
| TREC Mistral-base | 0.0898 | 0.0704 | 0.784 |
| CodeXGLUE Llama3-base | 0.0467 | 0.0397 | 0.851 |
| CodeXGLUE Mistral-IT | 0.0759 | 0.0316 | 0.417 |

That looks odd if interpreted as ordinary cluster separation. But the global
mean direction is also large:

| row | global mean norm | mean sample norm |
| --- | ---: | ---: |
| TREC Mistral-base | 8.995 | 10.167 |
| CodeXGLUE Llama3-base | 12.561 | 13.401 |
| CodeXGLUE Mistral-IT | 10.026 | 10.996 |

After subtracting the global mean, the centroid-distance ratio becomes greater
than 1.0:

| row | centered within distance | centered centroid distance | centered between/within |
| --- | ---: | ---: | ---: |
| TREC Mistral-base | 0.539 | 1.169 | 2.169 |
| CodeXGLUE Llama3-base | 0.500 | 1.194 | 2.386 |
| CodeXGLUE Mistral-IT | 0.632 | 1.191 | 1.884 |

So the current reading is:

```text
single-head Q-space often has a strong common direction;
task labels may be encoded as relatively small residual directions;
cosine silhouette measures compact raw clusters,
while linear probes can read residual task axes.
```

This explains why modest silhouette and high probe accuracy can coexist without
requiring the signal to be spurious.

## Post-RoPE Follow-Up

The same audit was then run on matched post-RoPE rows:

- TREC Mistral-base L17/H26, `pool_last_k=1`;
- CodeXGLUE Llama3-base L19/H30, `pool_last_k=5`;
- CodeXGLUE Mistral-IT L21/H18, `pool_last_k=5`.

These are matched layer/head rows, not newly chosen post-hoc rows. In all three
cases the same layer/head remains the best post-RoPE row in the corresponding
monolith summary.

| row | pre sampled sil | post sampled sil | pre probe | post probe | pre centered ratio | post centered ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TREC Mistral-base L17/H26 | 0.0948 | 0.0831 | 0.876 | 0.874 | 2.169 | 2.083 |
| CodeXGLUE Llama3-base L19/H30 | 0.2044 | 0.1791 | 0.982 | 0.981 | 2.386 | 2.279 |
| CodeXGLUE Mistral-IT L21/H18 | 0.0895 | 0.0841 | 0.974 | 0.971 | 1.884 | 1.856 |

The post-RoPE result is not a collapse. It is closer to:

```text
raw compactness weakens slightly after RoPE,
but the residual linear readout and centered class-separation ratio remain.
```

This is exactly the kind of outcome that makes the pre/post distinction useful.
If the signal disappeared post-RoPE, the pre-RoPE audit would look more like a
positioning artifact. If nothing changed at all, RoPE would look irrelevant to
this geometry. Instead, the audited rows preserve the task readout while shaving
some raw cosine compactness.

The PCA probe curves tell the same story:

| row | pre PCA-8 | post PCA-8 | pre PCA-16 | post PCA-16 | pre full-128 | post full-128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TREC Mistral-base L17/H26 | 0.679 | 0.674 | 0.785 | 0.778 | 0.876 | 0.874 |
| CodeXGLUE Llama3-base L19/H30 | 0.936 | 0.927 | 0.970 | 0.971 | 0.982 | 0.981 |
| CodeXGLUE Mistral-IT L21/H18 | 0.848 | 0.840 | 0.921 | 0.909 | 0.974 | 0.971 |

For these rows, RoPE does not erase the low-to-mid-dimensional readout. The
largest post-RoPE reductions in the table are still small relative to the
absolute probe accuracy.

## What This Does Not Prove

This is still descriptive geometry. It does not prove that the audited heads
causally drive downstream behavior, nor does it prove that every low-silhouette
row has a useful distributed code. The audit only says that for these
representative rows, the task variable is linearly readable even when raw
cosine silhouette is modest.

## Next Checks

- Add a class-balanced TREC audit excluding `ABBR`.
- Compare Q-space with K-space and V-space to see whether the residual linear
  code is Q-specific.
- Use causal ablation after the geometry is stable across seeds and capture
  stage.
