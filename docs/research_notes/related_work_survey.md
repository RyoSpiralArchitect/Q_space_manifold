# Related-Work Survey

Date: 2026-05-27

This is a lightweight positioning note, not a full literature review. It records
the first external check before tightening claims around Q-space stance
geometry.

## Search Queries

The initial search used these phrases:

```text
query vector probing transformer
attention head subjectivity probe
instruction tuning attention representation
RoPE position attention probe
mechanistic interpretability query key value vectors transformer QK circuit
query activations mechanistic interpretability
sparse query features transformer steering
human-interpretable QK subspaces
```

The quick result: nearby work exists on layer-wise probing, attention-head
specialization and pruning, fine-tuning representation geometry, and RoPE
position encoding. In mechanistic interpretability, the closest neighbors are
QK-circuit analysis, QK-subspace decomposition, and query-activation steering.
This scan did not find a direct prior with exactly the same object:
per-layer/per-head Query activation geometry as a stance/readout probe, with
pre/post-RoPE comparison, token-level Q-flow, and base-vs-instruct family scans.

That does not establish novelty. It only says the current repo should position
itself cautiously as an exploratory probe adjacent to several known literatures.

## Nearby Threads

### Layer-Wise Probing

[BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950)
uses probing to localize linguistic information across layers. Its layer-wise
framing is close in spirit, but it probes encoder representations rather than
query vectors inside attention heads.

### Attention-Head Specialization

[Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting,
the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418) and
[Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)
both support the general idea that attention heads can differ in importance and
role. This repo should therefore avoid over-claiming that head specialization is
new. The more specific angle here is geometric comparison of Q vectors as
candidate retrieval posture.

### Mechanistic Interpretability: QK and OV Circuits

Transformer-circuits-style mechanistic interpretability treats attention heads
as decomposable into QK and OV circuits. A concise statement of this view is in
[QK and OV Circuits](https://learnmechinterp.com/topics/qk-ov-circuits/):
QK controls where a head looks, while OV controls what information it moves.

This is the closest conceptual foundation for the current probe. The repo's
language around "retrieval posture" should be read as an activation-level,
dataset-conditioned view of the Q side of the QK circuit, not as a replacement
for full QK/OV circuit analysis.

[Copy Suppression](https://learnmechinterp.com/topics/copy-suppression/) is
also relevant because it describes a head whose query reads prediction-related
information from the residual stream and uses QK matching to find repeated
tokens. That supports the broad idea that query activations can encode what a
head is currently trying to resolve. The current repo differs by scanning Q
activation geometry across tasks and model families rather than reverse
engineering one head's complete QK/OV algorithm.

### Closest Query/QK Neighbors

[Decomposing Query-Key Feature Interactions Using Contrastive
Covariances](https://arxiv.org/abs/2602.04752) is the closest discovered
near-neighbor. It studies QK space as a bilinear joint embedding between queries
and keys, decomposing it into low-rank human-interpretable components for
semantic and binding features. This overlaps strongly with the idea that
attention's query/key machinery contains interpretable geometry.

The difference is that the Q-space manifold probe currently studies Query
activations alone as a sample geometry:

```text
this repo:  Q activation geometry -> stance/readout separability and token flow
QK paper:   QK bilinear subspaces -> attention-score feature attribution
```

[Steered Generation via Gradient-Based Optimization on Sparse Query
Features](https://arxiv.org/abs/2605.23040) is another close neighbor. It applies
sparse autoencoders to attention query activations and uses those features for
steering generation. This is especially important because it independently
treats query activations as a high-value intervention site.

The difference is again in the object and aim:

```text
this repo:        diagnostic geometry and phase scan
sparse-query SAE: feature decomposition and steering/control
```

Together, these papers weaken any claim that "query activations have not been
studied." They strengthen a more precise claim: stance-sensitive, per-head
Query activation geometry across pre/post-RoPE, task framing, tuning state, and
model family appears underexplored.

### Fine-Tuning and Representation Geometry

[Fine-Tuned Transformers Show Clusters of Similar Representations Across
Layers](https://arxiv.org/abs/2109.08406) uses CKA to study representation
similarity after fine-tuning. [Fine-Tuning Enhances Existing Mechanisms: A Case
Study on Entity Tracking](https://arxiv.org/abs/2402.14811) is also relevant
because it frames fine-tuning as enhancing existing mechanisms rather than
necessarily creating new ones.

This matters for the current base-vs-instruct language: instruction tuning may
preserve, sharpen, relocate, or diffuse an existing Q-space geometry. It is too
early to say it creates a new stance subsystem.

### RoPE and Pre/Post Position Phase

[RoFormer: Enhanced Transformer with Rotary Position
Embedding](https://arxiv.org/abs/2104.09864) is the primary RoPE reference.
Because RoPE rotates query and key states before attention scoring, pre-RoPE and
post-RoPE Q should be treated as different experimental surfaces:

```text
pre-RoPE Q  = content-conditioned query projection before rotary phase
post-RoPE Q = query after positional phase, before score computation
```

The current Mistral-IT pilot says the stance signal survives post-RoPE but
reorganizes. That is a sanity check, not proof that the same result holds across
families or tasks.

## Claim Hygiene

Use these phrasings for now:

- "stance-sensitive Q-space band" rather than "stance subsystem"
- "may depend on family/tuning/task/readout" rather than "shifts by model
  family"
- "weak single-head localization" rather than "no signal"
- "prompted classification stance" rather than "generic sentiment
  representation"
- "pre/post-RoPE pilot sanity check" rather than "artifact ruled out"
- "underexplored probe surface" rather than "first query-vector
  interpretability method"

Potential novelty phrasing:

```text
We explore an underexplored probe surface: per-head Query activation geometry as
a task-conditioned retrieval-posture signal, comparing pre/post-RoPE capture,
model family, tuning state, task framing, token pooling, and token-level flow.
```

## Follow-Up Literature Tasks

- Search specifically for Q/K/V vector probing inside causal decoder attention.
- Search for subjectivity/objectivity probes in decoder-only models, not only
  encoder classifiers.
- Search for RoPE analysis that compares pre-rotation and post-rotation Q/K
  states.
- Read the full QK contrastive-covariance and sparse-query steering papers for
  method-level overlap, especially whether either already reports per-head
  stance or sentiment/subjectivity geometry.
- Add a proper bibliography section if this repo turns into a writeup.
