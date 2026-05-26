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
```

The quick result: nearby work exists on layer-wise probing, attention-head
specialization and pruning, fine-tuning representation geometry, and RoPE
position encoding. This scan did not find a direct prior with exactly the same
object: per-layer/per-head Q-vector stance geometry, pre/post-RoPE comparison,
token-level Q-flow, and base-vs-instruct family scans.

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

## Follow-Up Literature Tasks

- Search specifically for Q/K/V vector probing inside causal decoder attention.
- Search for subjectivity/objectivity probes in decoder-only models, not only
  encoder classifiers.
- Search for RoPE analysis that compares pre-rotation and post-rotation Q/K
  states.
- Add a proper bibliography section if this repo turns into a writeup.
