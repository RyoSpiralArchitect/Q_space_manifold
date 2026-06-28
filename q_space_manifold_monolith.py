#!/usr/bin/env python3
"""Run Q-space manifold, head, layer, and token-flow probes in one script.

  python3 q_space_manifold_monolith.py \
    --backend torch --model-path gpt2 --target-layer 6 --target-head 3

  python3 q_space_manifold_monolith.py \
    --backend mlx --model-path mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --target-layer 6 --target-head 3
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_TEXT_ROWS = [
    {"text": "Why is life so beautiful and amazing?", "label": 0, "class_name": "Positive/Hopeful"},
    {"text": "How can we achieve world peace?", "label": 0, "class_name": "Positive/Hopeful"},
    {"text": "What makes a person truly happy?", "label": 0, "class_name": "Positive/Hopeful"},
    {"text": "Why do people love each other?", "label": 0, "class_name": "Positive/Hopeful"},
    {"text": "How to clear your mind and feel great?", "label": 0, "class_name": "Positive/Hopeful"},
    {"text": "What is the best thing about love?", "label": 0, "class_name": "Positive/Hopeful"},
    {"text": "Why is everything so difficult and sad?", "label": 1, "class_name": "Negative/Pessimistic"},
    {"text": "How do wars start?", "label": 1, "class_name": "Negative/Pessimistic"},
    {"text": "What causes extreme human loneliness?", "label": 1, "class_name": "Negative/Pessimistic"},
    {"text": "Why do people betray each other?", "label": 1, "class_name": "Negative/Pessimistic"},
    {"text": "How to deal with worst case scenarios?", "label": 1, "class_name": "Negative/Pessimistic"},
    {"text": "What is the worst part of failure?", "label": 1, "class_name": "Negative/Pessimistic"},
    {"text": "What is the capital city of France?", "label": 2, "class_name": "Factual/Objective"},
    {"text": "How many planets are in the solar system?", "label": 2, "class_name": "Factual/Objective"},
    {"text": "What is the freezing point of water?", "label": 2, "class_name": "Factual/Objective"},
    {"text": "How does a computer CPU work?", "label": 2, "class_name": "Factual/Objective"},
    {"text": "When was the internet invented?", "label": 2, "class_name": "Factual/Objective"},
    {"text": "Who discovered gravity?", "label": 2, "class_name": "Factual/Objective"},
]

DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

SPECIAL_TOKEN_STRINGS = {
    "<s>",
    "</s>",
    "<unk>",
    "<pad>",
    "<mask>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "[BOS]",
    "[EOS]",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[UNK]",
}

Q_STAGE_PRE_ROPE = "q_projection_output_pre_attention_position_rotation"
Q_STAGE_POST_ROPE = "q_projection_output_post_attention_position_rotation"

Q_CAPTURE_STAGE_ALIASES = {
    "pre-rope": Q_STAGE_PRE_ROPE,
    "pre_rope": Q_STAGE_PRE_ROPE,
    "pre": Q_STAGE_PRE_ROPE,
    "q-proj": Q_STAGE_PRE_ROPE,
    "q_proj": Q_STAGE_PRE_ROPE,
    Q_STAGE_PRE_ROPE: Q_STAGE_PRE_ROPE,
    "post-rope": Q_STAGE_POST_ROPE,
    "post_rope": Q_STAGE_POST_ROPE,
    "post": Q_STAGE_POST_ROPE,
    "post-rope-pre-score": Q_STAGE_POST_ROPE,
    "post_rope_pre_score": Q_STAGE_POST_ROPE,
    Q_STAGE_POST_ROPE: Q_STAGE_POST_ROPE,
}

ACTIVATION_SPACE_ALIASES = {
    "q": "q",
    "query": "q",
    "queries": "q",
    "query-space": "q",
    "query_space": "q",
    "k": "k",
    "key": "k",
    "keys": "k",
    "key-space": "k",
    "key_space": "k",
    "v": "v",
    "value": "v",
    "values": "v",
    "value-space": "v",
    "value_space": "v",
}

ACTIVATION_SPACE_LABELS = {
    "q": "Q",
    "k": "K",
    "v": "V",
}


@dataclass(frozen=True)
class TextDataset:
    texts: list[str]
    labels: list[int]
    class_names: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CaptureBundle:
    final_q_all: Any
    token_q_records: list[Any]
    token_records: list[list[str]]
    model_info: dict[str, Any]


@dataclass(frozen=True)
class TorchProjectionSpec:
    path_template: str
    kind: str


def cleanup_runtime_caches(backend: str | None = None) -> None:
    """Release large model/projection temporaries between long sweep stages."""
    gc.collect()
    if backend in (None, "mlx"):
        try:
            import mlx.core as mx  # type: ignore
        except Exception:
            pass
        else:
            try:
                mx.synchronize()
            except Exception:
                pass
            try:
                mx.clear_cache()
            except Exception:
                pass
            metal = getattr(mx, "metal", None)
            if metal is not None:
                try:
                    metal.clear_cache()
                except Exception:
                    pass
    if backend in (None, "torch"):
        try:
            import torch  # type: ignore
        except Exception:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass


@dataclass(frozen=True)
class ModelRunSpec:
    alias: str
    backend: str
    model_path: str
    target_layer: int | None = None
    target_head: int | None = None
    target_layer_fraction: float | None = None
    detail_layer_heads: str | None = None
    activation_space: str | None = None


def load_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit("numpy is required: pip install numpy") from exc
    return np


def load_pyplot(show: bool):
    try:
        import matplotlib  # type: ignore

        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plots: pip install matplotlib") from exc
    return plt


def progress(items: Iterable[Any], desc: str) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        return items
    return tqdm(items, desc=desc)


def rows_to_text_dataset(
    rows: Sequence[Any],
    *,
    declared_class_names: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TextDataset:
    raw_texts: list[str] = []
    raw_labels: list[int] = []
    raw_label_to_name: dict[int, str] = {}
    for index, row in enumerate(rows):
        if isinstance(row, str):
            text = row
            raw_label = 0
            class_name = "Class 0"
        elif isinstance(row, dict):
            text = str(row.get("text", ""))
            raw_label = int(row.get("label", 0))
            class_name = str(row.get("class_name", f"Class {raw_label}"))
        else:
            raise SystemExit(f"dataset row {index} must be a string or object")
        if not text:
            raise SystemExit(f"dataset row {index} is missing text")
        raw_texts.append(text)
        raw_labels.append(raw_label)
        raw_label_to_name.setdefault(raw_label, class_name)

    raw_label_values = sorted(set(raw_labels))
    raw_to_label = {raw_label: idx for idx, raw_label in enumerate(raw_label_values)}
    labels = [raw_to_label[raw_label] for raw_label in raw_labels]
    if declared_class_names and all(0 <= raw_label < len(declared_class_names) for raw_label in raw_label_values):
        class_names = [str(declared_class_names[raw_label]) for raw_label in raw_label_values]
    else:
        class_names = [raw_label_to_name[raw_label] for raw_label in raw_label_values]
    dataset_metadata = dict(metadata or {})
    dataset_metadata["sample_count"] = len(raw_texts)
    dataset_metadata["class_counts"] = {
        class_names[label]: labels.count(label)
        for label in sorted(set(labels))
    }
    dataset_metadata["raw_label_to_label"] = {
        str(raw_label): raw_to_label[raw_label]
        for raw_label in raw_label_values
    }
    return TextDataset(texts=raw_texts, labels=labels, class_names=class_names, metadata=dataset_metadata)


def apply_text_template(dataset: TextDataset, template: str) -> TextDataset:
    if not template:
        return dataset
    rendered_texts = []
    for index, (text, label) in enumerate(zip(dataset.texts, dataset.labels)):
        class_name = dataset.class_names[label]
        try:
            rendered = template.format(
                text=text,
                label=label,
                class_name=class_name,
                index=index,
            )
        except KeyError as exc:
            raise SystemExit(
                f"--text-template unknown field {exc.args[0]!r}; "
                "available fields are {text}, {label}, {class_name}, and {index}"
            ) from exc
        if not rendered.strip():
            raise SystemExit(f"--text-template produced empty text for sample {index}")
        rendered_texts.append(rendered)
    metadata = dict(dataset.metadata)
    metadata["text_template"] = template
    metadata["text_template_fields"] = ["text", "label", "class_name", "index"]
    return TextDataset(
        texts=rendered_texts,
        labels=list(dataset.labels),
        class_names=list(dataset.class_names),
        metadata=metadata,
    )


def balanced_or_limited_rows(
    rows: Sequence[dict[str, Any]],
    *,
    samples_per_class: int,
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    np = load_numpy()
    rows_list = [dict(row) for row in rows]
    rng = np.random.default_rng(seed)
    if samples_per_class > 0:
        by_label: dict[int, list[dict[str, Any]]] = {}
        for row in rows_list:
            by_label.setdefault(int(row["label"]), []).append(row)
        sampled = []
        for label in sorted(by_label):
            group = list(by_label[label])
            order = rng.permutation(len(group))
            take = min(samples_per_class, len(group))
            sampled.extend(group[int(index)] for index in order[:take])
        rows_list = sampled
    if max_samples > 0 and len(rows_list) > max_samples:
        order = rng.permutation(len(rows_list))
        rows_list = [rows_list[int(index)] for index in order[:max_samples]]
    return rows_list


def load_json_dataset(path: Path) -> TextDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("texts") or payload.get("rows")
        declared_class_names = payload.get("class_names")
    else:
        rows = payload
        declared_class_names = None
    if not isinstance(rows, list):
        raise SystemExit("dataset JSON must be a list or an object with a 'texts' list")
    return rows_to_text_dataset(
        rows,
        declared_class_names=declared_class_names,
        metadata={"dataset_source": "json", "dataset_json": str(path)},
    )


def hf_feature_class_names(dataset: Any, label_column: str) -> list[str] | None:
    feature = getattr(dataset, "features", {}).get(label_column)
    names = getattr(feature, "names", None)
    if names:
        return [str(name) for name in names]
    return None


def load_hf_rows(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    label_column: str,
    fallback_class_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str] | None, dict[str, Any]]:
    try:
        from datasets import load_dataset as hf_load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit("Hugging Face datasets support requires: pip install datasets") from exc

    loaded = hf_load_dataset(dataset_name, dataset_config, split=split) if dataset_config else hf_load_dataset(dataset_name, split=split)
    if text_column not in loaded.column_names:
        candidates = [name for name in ["text", "sentence", "review", "content"] if name in loaded.column_names]
        if not candidates:
            raise SystemExit(f"text column {text_column!r} not found in dataset columns: {loaded.column_names}")
        text_column = candidates[0]
    if label_column not in loaded.column_names:
        candidates = [name for name in ["label", "labels", "class"] if name in loaded.column_names]
        if not candidates:
            raise SystemExit(f"label column {label_column!r} not found in dataset columns: {loaded.column_names}")
        label_column = candidates[0]

    declared_class_names = hf_feature_class_names(loaded, label_column) or (
        [str(name) for name in fallback_class_names] if fallback_class_names else None
    )
    rows = []
    for item in loaded:
        text = str(item[text_column]).strip()
        if not text:
            continue
        label = int(item[label_column])
        class_name = (
            declared_class_names[label]
            if declared_class_names and 0 <= label < len(declared_class_names)
            else f"Class {label}"
        )
        rows.append({"text": text, "label": label, "class_name": class_name})
    rows = balanced_or_limited_rows(
        rows,
        samples_per_class=args.samples_per_class,
        max_samples=args.max_samples,
        seed=args.dataset_seed if args.dataset_seed is not None else args.random_state,
    )
    metadata = {
        "dataset_source": args.dataset_source,
        "hf_dataset_name": dataset_name,
        "hf_dataset_config": dataset_config,
        "dataset_split": split,
        "text_column": text_column,
        "label_column": label_column,
        "samples_per_class": args.samples_per_class,
        "max_samples": args.max_samples,
        "dataset_seed": args.dataset_seed if args.dataset_seed is not None else args.random_state,
    }
    return rows, declared_class_names, metadata


def load_dataset_from_args(args: argparse.Namespace) -> TextDataset:
    source = args.dataset_source
    if args.dataset_json is not None and source == "default":
        source = "json"
    if source == "default":
        rows = balanced_or_limited_rows(
            [dict(row) for row in DEFAULT_TEXT_ROWS],
            samples_per_class=args.samples_per_class,
            max_samples=args.max_samples,
            seed=args.dataset_seed if args.dataset_seed is not None else args.random_state,
        )
        return rows_to_text_dataset(
            rows,
            metadata={
                "dataset_source": "default",
                "samples_per_class": args.samples_per_class,
                "max_samples": args.max_samples,
                "dataset_seed": args.dataset_seed if args.dataset_seed is not None else args.random_state,
            },
        )
    if source == "json":
        if args.dataset_json is None:
            raise SystemExit("--dataset-source json requires --dataset-json")
        return load_json_dataset(args.dataset_json)
    if source == "sst2":
        rows, class_names, metadata = load_hf_rows(
            args,
            dataset_name="glue",
            dataset_config="sst2",
            split=args.dataset_split,
            text_column=args.text_column or "sentence",
            label_column=args.label_column or "label",
            fallback_class_names=["negative", "positive"],
        )
        return rows_to_text_dataset(rows, declared_class_names=class_names, metadata=metadata)
    if source == "subj":
        rows, class_names, metadata = load_hf_rows(
            args,
            dataset_name=args.hf_dataset_name or "SetFit/subj",
            dataset_config=args.hf_dataset_config,
            split=args.dataset_split,
            text_column=args.text_column or "text",
            label_column=args.label_column or "label",
            fallback_class_names=["subjective", "objective"],
        )
        return rows_to_text_dataset(rows, declared_class_names=class_names, metadata=metadata)
    if source == "hf":
        if not args.hf_dataset_name:
            raise SystemExit("--dataset-source hf requires --hf-dataset-name")
        rows, class_names, metadata = load_hf_rows(
            args,
            dataset_name=args.hf_dataset_name,
            dataset_config=args.hf_dataset_config,
            split=args.dataset_split,
            text_column=args.text_column or "text",
            label_column=args.label_column or "label",
        )
        return rows_to_text_dataset(rows, declared_class_names=class_names, metadata=metadata)
    raise SystemExit(f"unknown dataset source: {source}")


def load_dataset(path: Path | None) -> TextDataset:
    if path is None:
        return rows_to_text_dataset(list(DEFAULT_TEXT_ROWS), metadata={"dataset_source": "default"})
    return load_json_dataset(path)


def portable_output_path(path_value: Any) -> str:
    if path_value is None:
        return ""
    text = str(path_value)
    if not text or text.startswith("~/"):
        return text
    path = Path(text)
    if not path.is_absolute():
        return text
    try:
        return f"~/{path.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return text


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if type(obj).__module__.startswith("numpy") and hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): sanitize_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if type(obj).__module__.startswith("numpy") and hasattr(obj, "tolist"):
        return sanitize_json(obj.tolist())
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def normalize_rows(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        return rows.to_dict(orient="records")
    return [dict(row) for row in rows]


def normalize_q_capture_stage(stage: str) -> str:
    normalized = str(stage or "pre-rope").strip().lower()
    if normalized not in Q_CAPTURE_STAGE_ALIASES:
        valid = ", ".join(sorted({key for key in Q_CAPTURE_STAGE_ALIASES if not key.startswith("q_projection_")}))
        raise SystemExit(f"unknown --q-capture-stage {stage!r}; expected one of: {valid}")
    return Q_CAPTURE_STAGE_ALIASES[normalized]


def normalize_activation_space(space: str) -> str:
    normalized = str(space or "q").strip().lower()
    if normalized not in ACTIVATION_SPACE_ALIASES:
        valid = ", ".join(sorted(ACTIVATION_SPACE_ALIASES))
        raise SystemExit(f"unknown --activation-space {space!r}; expected one of: {valid}")
    return ACTIVATION_SPACE_ALIASES[normalized]


def activation_space_label(space: str) -> str:
    return ACTIVATION_SPACE_LABELS.get(normalize_activation_space(space), str(space).upper())


def activation_capture_stage(stage: str, activation_space: str = "q") -> str:
    """Space-neutral capture-stage label for Q/K/V projection artifacts."""
    stage = normalize_q_capture_stage(stage)
    space = normalize_activation_space(activation_space)
    if stage == Q_STAGE_PRE_ROPE:
        return "projection_output_pre_attention_position_rotation"
    if stage == Q_STAGE_POST_ROPE:
        return "query_after_attention_position_rotation_pre_score"
    return f"{space}_activation_capture_stage:{stage}"


def q_capture_position_note(
    stage: str,
    *,
    backend: str,
    projection_kind: str,
    activation_space: str = "q",
) -> str:
    activation_label = activation_space_label(activation_space)
    if stage == Q_STAGE_PRE_ROPE:
        if activation_space == "v":
            return (
                f"This is the pre-attention {activation_label} projection output. "
                "For RoPE attention, value vectors are not rotary-position-rotated; "
                "use this as the V-space companion to pre-RoPE Q/K projection captures."
            )
        if backend == "torch":
            return (
                f"For RoPE models this is pre-RoPE {activation_label}. For GPT-2-style absolute-position "
                f"models this is the {activation_label} projection after positional information has already "
                "entered the residual stream, before attention scoring."
            )
        return (
            f"For RoPE models this is pre-RoPE {activation_label} captured at the "
            f"{activation_space}_projection output, before rotary position embedding is applied inside attention."
        )
    if stage == Q_STAGE_POST_ROPE:
        return (
            "For RoPE models this is the query tensor captured from the model's actual "
            "RoPE call after rotary position embedding, before attention scaling/scoring. "
            "The current implementation supports MLX full-sequence capture with cache offset 0."
        )
    return f"Q capture stage {stage!r} for backend={backend}, projection_kind={projection_kind}."


def effective_linear_probe_permutation_n(args: argparse.Namespace) -> int:
    if args.linear_probe_permutation_n is None:
        return int(args.label_permutation_n)
    return int(args.linear_probe_permutation_n)


def score_sort_key(row: dict[str, Any]) -> float:
    score = float(row["silhouette_cosine"])
    return score if not math.isnan(score) else -999.0


def parse_layer_head_specs(spec: str | None) -> list[tuple[int, int]]:
    if not spec:
        return []
    pairs: list[tuple[int, int]] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        normalized = part.lower().replace("layer", "").replace("head", "")
        normalized = normalized.replace("l", "").replace("h", "")
        if ":" not in normalized:
            raise SystemExit(f"invalid --detail-layer-heads item {part!r}; expected LAYER:HEAD")
        layer_text, head_text = normalized.split(":", 1)
        try:
            pairs.append((int(layer_text), int(head_text)))
        except ValueError as exc:
            raise SystemExit(f"invalid --detail-layer-heads item {part!r}; expected integer LAYER:HEAD") from exc
    return pairs


def add_unique_pair(
    pairs: list[dict[str, Any]],
    *,
    layer: int,
    head: int,
    reason: str,
) -> None:
    if any(item["layer"] == layer and item["head"] == head for item in pairs):
        return
    pairs.append({"layer": layer, "head": head, "reason": reason})


def slugify(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    slug = "".join(safe).strip("_")
    return slug or "model"


def parse_model_run_spec(raw: str, default_backend: str) -> ModelRunSpec:
    item = raw.strip()
    if not item:
        raise SystemExit("empty model spec in --batch-models")
    alias = ""
    body = item
    if "=" in item:
        alias, body = item.split("=", 1)
        alias = alias.strip()
        body = body.strip()
    backend = default_backend
    model_path = body
    if ":" in body:
        prefix, rest = body.split(":", 1)
        if prefix in {"torch", "mlx"}:
            backend = prefix
            model_path = rest
    if backend not in {"torch", "mlx"}:
        raise SystemExit(f"invalid backend {backend!r} in model spec {raw!r}")
    if not model_path:
        raise SystemExit(f"missing model path in model spec {raw!r}")
    if not alias:
        alias = slugify(model_path.split("/")[-1])
    return ModelRunSpec(alias=slugify(alias), backend=backend, model_path=model_path)


def load_model_run_specs(args: argparse.Namespace) -> list[ModelRunSpec]:
    specs: list[ModelRunSpec] = []
    if args.model_list_json is not None:
        payload = json.loads(args.model_list_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("--model-list-json must contain a list")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise SystemExit(f"model-list-json item {index} must be an object")
            model_path = str(item.get("model_path") or item.get("path") or "")
            if not model_path:
                raise SystemExit(f"model-list-json item {index} is missing model_path")
            backend = str(item.get("backend", args.backend))
            if backend not in {"torch", "mlx"}:
                raise SystemExit(f"model-list-json item {index} has invalid backend {backend!r}")
            alias = slugify(str(item.get("alias") or model_path.split("/")[-1]))
            specs.append(
                ModelRunSpec(
                    alias=alias,
                    backend=backend,
                    model_path=model_path,
                    target_layer=int(item["target_layer"]) if item.get("target_layer") is not None else None,
                    target_head=int(item["target_head"]) if item.get("target_head") is not None else None,
                    target_layer_fraction=float(item["target_layer_fraction"])
                    if item.get("target_layer_fraction") is not None
                    else None,
                    detail_layer_heads=str(item["detail_layer_heads"])
                    if item.get("detail_layer_heads") is not None
                    else None,
                    activation_space=normalize_activation_space(str(item["activation_space"]))
                    if item.get("activation_space") is not None
                    else None,
                )
            )
    if args.batch_models:
        specs.extend(parse_model_run_spec(part, args.backend) for part in args.batch_models.split(",") if part.strip())
    return specs


def args_for_model_spec(args: argparse.Namespace, spec: ModelRunSpec, output_dir: Path) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.backend = spec.backend
    run_args.model_path = spec.model_path
    run_args.output_dir = output_dir / spec.alias
    run_args.model_alias = spec.alias
    if spec.target_layer is not None:
        run_args.target_layer = spec.target_layer
    if spec.target_head is not None:
        run_args.target_head = spec.target_head
    if spec.target_layer_fraction is not None:
        run_args.target_layer_fraction = spec.target_layer_fraction
    if spec.detail_layer_heads is not None:
        run_args.detail_layer_heads = spec.detail_layer_heads
    if spec.activation_space is not None:
        run_args.activation_space = spec.activation_space
    return run_args


def get_attr_path(root: Any, dotted: str) -> Any:
    current = root
    for part in dotted.split("."):
        if isinstance(current, (list, tuple)):
            current = current[int(part)]
        elif part.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def try_attr_path(root: Any, dotted: str) -> Any | None:
    try:
        return get_attr_path(root, dotted)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def first_int_attr(obj: Any, names: Sequence[str]) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return int(value)
    return None


def infer_torch_shape(model: Any) -> tuple[int, int, int, int]:
    config = model.config
    n_layers = first_int_attr(config, ["n_layer", "num_hidden_layers", "num_layers"])
    num_heads = first_int_attr(config, ["n_head", "num_attention_heads", "num_heads", "n_heads"])
    hidden_dim = first_int_attr(config, ["n_embd", "hidden_size", "d_model"])
    head_dim = first_int_attr(config, ["head_dim", "attention_head_size"])
    if n_layers is None or num_heads is None or hidden_dim is None:
        raise SystemExit("could not infer layer/head/hidden dimensions from transformers config")
    if head_dim is None:
        head_dim = hidden_dim // num_heads
    return n_layers, num_heads, hidden_dim, head_dim


def torch_projection_candidates(activation_space: str) -> list[TorchProjectionSpec]:
    space = normalize_activation_space(activation_space)
    if space == "q":
        separate_names = ["q_proj", "query_proj", "wq"]
    elif space == "k":
        separate_names = ["k_proj", "key_proj", "wk"]
    else:
        separate_names = ["v_proj", "value_proj", "wv"]
    candidates = [TorchProjectionSpec("transformer.h.{layer}.attn.c_attn", "fused_qkv")]
    for name in separate_names:
        candidates.extend(
            [
                TorchProjectionSpec(f"model.layers.{{layer}}.self_attn.{name}", f"{space}_proj"),
                TorchProjectionSpec(f"model.decoder.layers.{{layer}}.self_attn.{name}", f"{space}_proj"),
            ]
        )
    candidates.extend(
        [
            TorchProjectionSpec("gpt_neox.layers.{layer}.attention.query_key_value", "fused_qkv"),
            TorchProjectionSpec("transformer.blocks.{layer}.attn.Wqkv", "fused_qkv"),
        ]
    )
    return candidates


def detect_torch_projection(model: Any, n_layers: int, activation_space: str) -> TorchProjectionSpec:
    candidates = [
        *torch_projection_candidates(activation_space),
    ]
    for spec in candidates:
        first = try_attr_path(model, spec.path_template.format(layer=0))
        last = try_attr_path(model, spec.path_template.format(layer=n_layers - 1))
        if first is not None and last is not None:
            return spec
    activation_label = activation_space_label(activation_space)
    raise SystemExit(
        f"could not find a supported torch {activation_label} projection. Supported shapes include "
        "GPT-2 c_attn and Llama/Qwen-style self_attn q/k/v projections."
    )


def collect_with_torch(args: argparse.Namespace, dataset: TextDataset) -> CaptureBundle:
    np = load_numpy()
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit("torch backend requires: pip install torch transformers") from exc
    activation_space = normalize_activation_space(args.activation_space)
    q_capture_stage = normalize_q_capture_stage(args.q_capture_stage)
    if q_capture_stage == Q_STAGE_POST_ROPE:
        raise SystemExit(
            "--q-capture-stage post-rope is currently implemented for the MLX backend only. "
            "Use --backend mlx for post-RoPE Q capture."
        )

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.torch_dtype != "auto":
        dtype = getattr(torch, args.torch_dtype, None)
        if dtype is None:
            raise SystemExit(f"unknown torch dtype: {args.torch_dtype}")
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs).to(device)
    model.eval()

    n_layers, num_heads, hidden_dim, head_dim = infer_torch_shape(model)
    spec = detect_torch_projection(model, n_layers, activation_space)
    current_q_cache: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def q_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            projection_output = output[0] if isinstance(output, (tuple, list)) else output
            if spec.kind == "fused_qkv":
                q_width = num_heads * head_dim
                offset = {"q": 0, "k": q_width, "v": 2 * q_width}[activation_space]
                if projection_output.shape[-1] < offset + q_width:
                    raise RuntimeError(
                        f"layer {layer_idx} QKV output is too small for {activation_space_label(activation_space)} "
                        f"width {q_width}: {tuple(projection_output.shape)}"
                    )
                captured = projection_output[..., offset: offset + q_width]
                captured_heads = num_heads
            else:
                captured = projection_output
                if captured.shape[-1] % head_dim != 0:
                    raise RuntimeError(
                        f"layer {layer_idx} {activation_space_label(activation_space)} shape "
                        f"{tuple(captured.shape)} is not divisible by head_dim={head_dim}"
                    )
                captured_heads = int(captured.shape[-1] // head_dim)
            captured = captured.reshape(captured.shape[0], captured.shape[1], captured_heads, head_dim)
            captured = captured.permute(0, 2, 1, 3).contiguous()
            current_q_cache[layer_idx] = captured.detach().float().cpu().numpy()[0]

        return q_hook

    for layer_idx in range(n_layers):
        module = get_attr_path(model, spec.path_template.format(layer=layer_idx))
        handles.append(module.register_forward_hook(make_hook(layer_idx)))

    final_q_records: list[Any] = []
    token_q_records: list[Any] = []
    token_records: list[list[str]] = []

    try:
        with torch.no_grad():
            for text in progress(dataset.texts, f"extracting {activation_space_label(activation_space)}/tokens"):
                current_q_cache.clear()
                encoded = tokenizer(text)
                input_ids = truncate_token_ids(encoded["input_ids"], args)
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                inputs = {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
                }
                model(**inputs)
                missing = [idx for idx in range(n_layers) if idx not in current_q_cache]
                if missing:
                    raise RuntimeError(f"missing {activation_space_label(activation_space)} captures for layers: {missing}")
                q_by_layer = np.stack([current_q_cache[idx] for idx in range(n_layers)], axis=0)
                k = min(args.pool_last_k, q_by_layer.shape[2])
                final_q = q_by_layer[:, :, -k:, :].mean(axis=2)
                compact_q, compact_tokens = compact_token_q_record(np, q_by_layer, tokens, args)
                final_q_records.append(final_q)
                token_q_records.append(compact_q)
                token_records.append(compact_tokens)
    finally:
        for handle in handles:
            handle.remove()

    final_q_all = np.stack(final_q_records, axis=0)
    final_q_records.clear()
    current_q_cache.clear()
    result = CaptureBundle(
        final_q_all=final_q_all,
        token_q_records=token_q_records,
        token_records=token_records,
        model_info={
            "backend": "torch",
            "model_path": args.model_path,
            "device": device,
            "projection_path": spec.path_template,
            "projection_kind": spec.kind,
            "activation_space": activation_space,
            "activation_space_label": activation_space_label(activation_space),
            "q_capture_stage": q_capture_stage,
            "activation_capture_stage": activation_capture_stage(q_capture_stage, activation_space),
            "q_capture_position_note": q_capture_position_note(
                q_capture_stage,
                backend="torch",
                projection_kind=spec.kind,
                activation_space=activation_space,
            ),
            "layers": n_layers,
            "heads": int(final_q_all.shape[2]),
            "query_heads": num_heads,
            "hidden_dim": hidden_dim,
            "head_dim": head_dim,
            "pool_last_k": args.pool_last_k,
        },
    )
    del model, tokenizer
    cleanup_runtime_caches("torch")
    return result


def find_mlx_layers(model: Any) -> list[Any]:
    candidates = [
        getattr(model, "layers", None),
        try_attr_path(model, "model.layers"),
        try_attr_path(model, "model.model.layers"),
        try_attr_path(model, "transformer.layers"),
    ]
    for layers in candidates:
        if isinstance(layers, list) and layers:
            return layers
        if isinstance(layers, tuple) and layers:
            return list(layers)
    raise SystemExit("could not find MLX transformer layers on the loaded model")


def infer_mlx_shape(model: Any, layers: Sequence[Any]) -> tuple[int, int, int, int]:
    args_obj = (
        getattr(model, "args", None)
        or getattr(getattr(model, "model", None), "args", None)
        or getattr(getattr(getattr(model, "model", None), "model", None), "args", None)
    )
    n_layers = first_int_attr(args_obj, ["num_hidden_layers", "n_layers"]) if args_obj is not None else None
    num_heads = first_int_attr(args_obj, ["num_attention_heads", "n_heads"]) if args_obj is not None else None
    hidden_dim = first_int_attr(args_obj, ["hidden_size", "dim", "d_model"]) if args_obj is not None else None
    head_dim = first_int_attr(args_obj, ["head_dim"]) if args_obj is not None else None

    if n_layers is None:
        n_layers = len(layers)
    if num_heads is None or hidden_dim is None or head_dim is None:
        first_attn = find_mlx_attention(layers[0])[0]
        num_heads = num_heads or first_int_attr(first_attn, ["n_heads", "num_heads", "num_attention_heads"])
        head_dim = head_dim or first_int_attr(first_attn, ["head_dim"])
    if num_heads is None or hidden_dim is None:
        raise SystemExit("could not infer MLX head/hidden dimensions")
    if head_dim is None:
        head_dim = hidden_dim // num_heads
    return int(n_layers), int(num_heads), int(hidden_dim), int(head_dim)


def find_mlx_attention(layer: Any) -> tuple[Any, str]:
    for name in ["self_attn", "attention", "attn"]:
        value = getattr(layer, name, None)
        if value is not None:
            return value, name
    raise SystemExit("could not find attention module on MLX layer")


def mlx_projection_names(activation_space: str) -> list[str]:
    space = normalize_activation_space(activation_space)
    if space == "q":
        return ["q_proj", "wq", "query_proj"]
    if space == "k":
        return ["k_proj", "wk", "key_proj"]
    return ["v_proj", "wv", "value_proj"]


def find_mlx_projection(layer: Any, activation_space: str) -> tuple[Any, str, str]:
    attention, attention_name = find_mlx_attention(layer)
    for projection_name in mlx_projection_names(activation_space):
        if hasattr(attention, projection_name):
            return attention, projection_name, f"{attention_name}.{projection_name}"
    raise SystemExit(
        f"could not find an MLX {activation_space_label(activation_space)} projection "
        f"such as self_attn.{activation_space}_proj"
    )


def find_mlx_rope(layer: Any) -> tuple[Any, str, str]:
    attention, attention_name = find_mlx_attention(layer)
    for rope_name in ["rope", "rotary_emb", "rotary_embedding"]:
        if hasattr(attention, rope_name):
            return attention, rope_name, f"{attention_name}.{rope_name}"
    raise SystemExit(
        "could not find an MLX RoPE module such as self_attn.rope; "
        "--q-capture-stage post-rope requires a RoPE-based attention module"
    )


class MlxQCaptureWrapper:
    def __init__(self, inner: Any, layer_idx: int, cache: dict[int, Any]) -> None:
        self._inner = inner
        self._layer_idx = layer_idx
        self._cache = cache

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._inner(*args, **kwargs)
        self._cache[self._layer_idx] = output
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class MlxRopeQCaptureWrapper:
    def __init__(self, inner: Any, layer_idx: int, cache: dict[int, Any], expected_heads: int) -> None:
        self._inner = inner
        self._layer_idx = layer_idx
        self._cache = cache
        self._expected_heads = expected_heads

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._inner(*args, **kwargs)
        shape = getattr(output, "shape", ())
        if self._layer_idx not in self._cache and len(shape) == 4 and int(shape[1]) == self._expected_heads:
            self._cache[self._layer_idx] = output
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def encode_mlx(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text) if hasattr(tokenizer, "encode") else tokenizer(text)
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def mlx_tokens(tokenizer: Any, ids: Sequence[int]) -> list[str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(tok) for tok in tokenizer.convert_ids_to_tokens(list(ids))]
    if hasattr(tokenizer, "decode"):
        return [str(tokenizer.decode([token_id])) for token_id in ids]
    return [str(token_id) for token_id in ids]


def truncate_token_ids(ids: Sequence[int], args: argparse.Namespace) -> list[int]:
    token_ids = [int(token_id) for token_id in ids]
    limit = int(getattr(args, "max_token_length", 0) or 0)
    if limit > 0 and len(token_ids) > limit:
        if args.token_truncation_side == "tail":
            token_ids = token_ids[-limit:]
        else:
            token_ids = token_ids[:limit]
    if not token_ids:
        raise RuntimeError("tokenization produced an empty input")
    return token_ids


def stored_token_indices(seq_len: int, args: argparse.Namespace) -> list[int]:
    limit = int(getattr(args, "max_stored_tokens", 0) or 0)
    if limit <= 0 or seq_len <= limit:
        return list(range(seq_len))
    if args.stored_token_selection == "tail":
        return list(range(seq_len - limit, seq_len))
    return list(range(limit))


def token_q_storage_dtype(args: argparse.Namespace) -> str:
    dtype = str(getattr(args, "token_q_storage_dtype", "float32"))
    if dtype not in {"float16", "float32"}:
        raise RuntimeError(f"unsupported token Q storage dtype: {dtype}")
    return dtype


def compact_token_q_record(np: Any, q_by_layer: Any, tokens: Sequence[str], args: argparse.Namespace) -> tuple[Any, list[str]]:
    indices = stored_token_indices(int(q_by_layer.shape[2]), args)
    compact = q_by_layer[:, :, indices, :]
    dtype = token_q_storage_dtype(args)
    if dtype != str(compact.dtype):
        compact = compact.astype(dtype)
    return compact, [str(tokens[idx]) for idx in indices]


def collect_with_mlx(args: argparse.Namespace, dataset: TextDataset) -> CaptureBundle:
    np = load_numpy()
    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm import load  # type: ignore
    except ImportError as exc:
        raise SystemExit("mlx backend requires: pip install mlx mlx-lm") from exc
    activation_space = normalize_activation_space(args.activation_space)
    q_capture_stage = normalize_q_capture_stage(args.q_capture_stage)
    if q_capture_stage == Q_STAGE_POST_ROPE and activation_space != "q":
        raise SystemExit(
            "--q-capture-stage post-rope currently captures RoPE-applied Q only. "
            "Use --q-capture-stage pre-rope with --activation-space k/v for K/V projection-space comparisons."
        )

    load_kwargs: dict[str, Any] = {}
    if args.trust_remote_code:
        load_kwargs["tokenizer_config"] = {"trust_remote_code": True}
    model, tokenizer = load(args.model_path, **load_kwargs)
    layers = find_mlx_layers(model)
    n_layers, num_heads, hidden_dim, head_dim = infer_mlx_shape(model, layers)
    if n_layers != len(layers):
        n_layers = min(n_layers, len(layers))
        layers = list(layers[:n_layers])

    current_q_cache: dict[int, Any] = {}
    originals: list[tuple[Any, str, Any]] = []
    projection_path = ""
    rope_path = ""
    for layer_idx, layer in enumerate(layers):
        parent, attr, projection_path = find_mlx_projection(layer, activation_space)
        if q_capture_stage == Q_STAGE_POST_ROPE:
            rope_parent, rope_attr, rope_path = find_mlx_rope(layer)
            original = getattr(rope_parent, rope_attr)
            originals.append((rope_parent, rope_attr, original))
            setattr(
                rope_parent,
                rope_attr,
                MlxRopeQCaptureWrapper(original, layer_idx, current_q_cache, num_heads),
            )
        else:
            original = getattr(parent, attr)
            originals.append((parent, attr, original))
            setattr(parent, attr, MlxQCaptureWrapper(original, layer_idx, current_q_cache))

    final_q_records: list[Any] = []
    token_q_records: list[Any] = []
    token_records: list[list[str]] = []

    try:
        for text in progress(dataset.texts, f"extracting MLX {activation_space_label(activation_space)}/tokens"):
            current_q_cache.clear()
            ids = truncate_token_ids(encode_mlx(tokenizer, text), args)
            tokens = mlx_tokens(tokenizer, ids)
            inputs = mx.array([ids])
            output = model(inputs)
            mx.eval(output)
            missing = [idx for idx in range(n_layers) if idx not in current_q_cache]
            if missing:
                raise RuntimeError(f"missing MLX {activation_space_label(activation_space)} captures for layers: {missing}")
            q_by_layer = []
            for layer_idx in range(n_layers):
                q_raw = current_q_cache[layer_idx]
                mx.eval(q_raw)
                if q_capture_stage == Q_STAGE_POST_ROPE:
                    if len(q_raw.shape) != 4 or q_raw.shape[1] != num_heads or q_raw.shape[-1] != head_dim:
                        raise RuntimeError(
                            f"MLX layer {layer_idx} post-RoPE Q shape {q_raw.shape} does not match "
                            f"[batch, heads={num_heads}, seq, head_dim={head_dim}]"
                        )
                    q_np = np.array(q_raw).astype("float32")[0]
                else:
                    if q_raw.shape[-1] % head_dim != 0:
                        raise RuntimeError(
                            f"MLX layer {layer_idx} {activation_space_label(activation_space)} shape "
                            f"{q_raw.shape} is not divisible by head_dim={head_dim}"
                        )
                    captured_heads = int(q_raw.shape[-1] // head_dim)
                    q_mx = q_raw.reshape(q_raw.shape[0], q_raw.shape[1], captured_heads, head_dim)
                    q_mx = q_mx.transpose(0, 2, 1, 3)
                    mx.eval(q_mx)
                    q_np = np.array(q_mx).astype("float32")[0]
                q_by_layer.append(q_np)
            q_by_layer_np = np.stack(q_by_layer, axis=0)
            k = min(args.pool_last_k, q_by_layer_np.shape[2])
            final_q = q_by_layer_np[:, :, -k:, :].mean(axis=2)
            compact_q, compact_tokens = compact_token_q_record(np, q_by_layer_np, tokens, args)
            final_q_records.append(final_q)
            token_q_records.append(compact_q)
            token_records.append(compact_tokens)
    finally:
        for parent, attr, original in originals:
            setattr(parent, attr, original)

    final_q_all = np.stack(final_q_records, axis=0)
    final_q_records.clear()
    current_q_cache.clear()
    result = CaptureBundle(
        final_q_all=final_q_all,
        token_q_records=token_q_records,
        token_records=token_records,
        model_info={
            "backend": "mlx",
            "model_path": args.model_path,
            "projection_path": projection_path,
            "projection_kind": f"{activation_space}_proj",
            "activation_space": activation_space,
            "activation_space_label": activation_space_label(activation_space),
            "rope_path": rope_path if q_capture_stage == Q_STAGE_POST_ROPE else "",
            "q_capture_stage": q_capture_stage,
            "activation_capture_stage": activation_capture_stage(q_capture_stage, activation_space),
            "q_capture_position_note": q_capture_position_note(
                q_capture_stage,
                backend="mlx",
                projection_kind=f"{activation_space}_proj",
                activation_space=activation_space,
            ),
            "layers": n_layers,
            "heads": int(final_q_all.shape[2]),
            "query_heads": num_heads,
            "hidden_dim": hidden_dim,
            "head_dim": head_dim,
            "pool_last_k": args.pool_last_k,
        },
    )
    del model, tokenizer
    cleanup_runtime_caches("mlx")
    return result


def pca_2d(np: Any, x: Any) -> Any:
    return pca_nd(np, x, n_components=2)


def pca_nd(np: Any, x: Any, *, n_components: int) -> Any:
    x = np.asarray(x, dtype="float32")
    n_components = max(1, int(n_components))
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, n_components), dtype="float32")
    x_centered = x - x.mean(axis=0, keepdims=True)
    k = min(n_components, x_centered.shape[0], x_centered.shape[1])
    if k <= 0:
        return np.zeros((x.shape[0], n_components), dtype="float32")
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    z = x_centered @ vt[:k].T
    if k < n_components:
        z = np.pad(z, ((0, 0), (0, n_components - k)))
    return z.astype("float32")


def compute_nd(
    np: Any,
    x: Any,
    *,
    n_components: int,
    n_neighbors: int = 5,
    min_dist: float = 0.3,
    metric: str = "cosine",
    random_state: int = 42,
    projection: str = "umap",
) -> Any:
    x = np.asarray(x, dtype="float32")
    n_components = max(1, int(n_components))
    if x.shape[0] <= 3 or projection == "pca":
        return pca_nd(np, x, n_components=n_components)
    try:
        import umap  # type: ignore
    except ImportError:
        print("umap-learn is not installed; falling back to PCA.", file=sys.stderr)
        return pca_nd(np, x, n_components=n_components)
    nn = max(2, min(n_neighbors, x.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=nn,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value 1 overridden to 1 by setting random_state.*",
            category=UserWarning,
        )
        return reducer.fit_transform(x)


def compute_2d(
    np: Any,
    x: Any,
    *,
    n_neighbors: int = 5,
    min_dist: float = 0.3,
    metric: str = "cosine",
    random_state: int = 42,
    projection: str = "umap",
) -> Any:
    return compute_nd(
        np,
        x,
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        projection=projection,
    )


def cosine_distance_matrix(np: Any, x: Any) -> Any:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    unit = x / safe
    dist = 1.0 - np.clip(unit @ unit.T, -1.0, 1.0)
    return np.maximum(dist, 0.0)


def euclidean_distance_matrix(np: Any, x: Any) -> Any:
    x = np.asarray(x, dtype="float32")
    diffs = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def silhouette_from_distance(np: Any, dist: Any, labels: Sequence[int]) -> float:
    labels_np = np.asarray(labels)
    if len(set(int(label) for label in labels_np)) < 2 or len(labels_np) <= len(set(labels_np)):
        return float("nan")
    try:
        scores = []
        for index, label in enumerate(labels_np):
            same = labels_np == label
            other = labels_np != label
            same[index] = False
            if not np.any(same) or not np.any(other):
                continue
            a = float(dist[index, same].mean())
            b = min(float(dist[index, labels_np == other_label].mean()) for other_label in set(labels_np[other]))
            denom = max(a, b)
            if denom > 1e-12:
                scores.append((b - a) / denom)
        return float(np.mean(scores)) if scores else float("nan")
    except Exception:
        return float("nan")


def safe_silhouette(np: Any, x: Any, labels: Sequence[int]) -> float:
    return silhouette_from_distance(np, cosine_distance_matrix(np, x), labels)


def pca_explained_variance_ratio(np: Any, x: Any, n_components: int = 2) -> list[float]:
    x = np.asarray(x, dtype="float32")
    if x.ndim != 2 or min(x.shape) <= 1:
        return []
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(x_centered, full_matrices=False)
    variances = (singular_values**2) / max(1, x.shape[0] - 1)
    total = float(variances.sum())
    if total <= 1e-12:
        return [0.0 for _ in range(min(n_components, len(variances)))]
    return [float(value / total) for value in variances[:n_components]]


def knn_indices_from_distance(np: Any, dist: Any, k: int) -> Any:
    dist = np.asarray(dist)
    n = dist.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=int)
    kk = max(1, min(k, n - 1))
    order = np.argsort(dist, axis=1)
    return order[:, 1 : kk + 1]


def knn_recall(np: Any, reference_dist: Any, projected_dist: Any, k: int) -> float:
    reference = knn_indices_from_distance(np, reference_dist, k)
    projected = knn_indices_from_distance(np, projected_dist, k)
    if reference.shape[1] == 0:
        return float("nan")
    recalls = []
    for ref_row, proj_row in zip(reference, projected):
        ref_set = set(int(item) for item in ref_row)
        proj_set = set(int(item) for item in proj_row)
        recalls.append(len(ref_set & proj_set) / len(ref_set))
    return float(np.mean(recalls)) if recalls else float("nan")


def class_centroid_rows(np: Any, x: Any, labels: Any, class_names: Sequence[str]) -> list[dict[str, Any]]:
    x = np.asarray(x, dtype="float32")
    labels_np = np.asarray(labels)
    rows = []
    for label in sorted(set(int(value) for value in labels_np)):
        idx = labels_np == label
        cluster = x[idx]
        if len(cluster) == 0:
            continue
        centroid = cluster.mean(axis=0)
        dispersion = float(np.mean(cosine_distance_matrix(np, cluster))) if len(cluster) > 1 else 0.0
        rows.append(
            {
                "label": label,
                "class_name": class_names[label],
                "count": int(idx.sum()),
                "centroid": centroid,
                "within_class_mean_cosine_distance": dispersion,
            }
        )
    return rows


def centroid_distance_summary(np: Any, x: Any, labels: Any, class_names: Sequence[str]) -> dict[str, Any]:
    centroid_rows = class_centroid_rows(np, x, labels, class_names)
    distances = []
    for left_idx, left in enumerate(centroid_rows):
        for right in centroid_rows[left_idx + 1 :]:
            dist = cosine_distance_matrix(np, np.stack([left["centroid"], right["centroid"]]))[0, 1]
            distances.append(
                {
                    "pair": f"{left['class_name']} vs {right['class_name']}",
                    "cosine_distance": float(dist),
                }
            )
    return {
        "class_dispersion": [
            {
                key: value
                for key, value in row.items()
                if key != "centroid"
            }
            for row in centroid_rows
        ],
        "centroid_pair_distances": distances,
        "mean_centroid_pair_cosine_distance": float(np.mean([d["cosine_distance"] for d in distances]))
        if distances
        else float("nan"),
    }


def class_color(class_idx: int) -> str:
    return DEFAULT_COLORS[class_idx % len(DEFAULT_COLORS)]


def projection_axis_label(args: argparse.Namespace, dimension: int) -> str:
    name = "UMAP" if args.projection == "umap" else "PCA"
    return f"{name} Dimension {dimension}"


def plot_sample_indices(np: Any, sample_count: int, args: argparse.Namespace) -> list[int]:
    limit = int(getattr(args, "plot_sample_limit", 0) or 0)
    if limit <= 0 or limit >= sample_count:
        return list(range(sample_count))
    rng = np.random.default_rng(int(getattr(args, "random_state", 42)))
    return sorted(int(index) for index in rng.choice(sample_count, size=limit, replace=False))


def project_highd_paths(
    np: Any,
    highd_paths: Sequence[Any],
    args: argparse.Namespace,
    *,
    n_components: int,
) -> tuple[Any, list[Any]]:
    token_counts = [len(path) for path in highd_paths]
    nonempty = [np.asarray(path, dtype="float32") for path in highd_paths if len(path) > 0]
    if not nonempty:
        return np.zeros((0, n_components), dtype="float32"), [
            np.zeros((0, n_components), dtype="float32") for _ in highd_paths
        ]
    all_q = np.concatenate(nonempty, axis=0)
    all_emb = compute_nd(
        np,
        all_q,
        n_components=n_components,
        n_neighbors=10,
        min_dist=0.2,
        metric=args.metric,
        random_state=args.random_state,
        projection=args.projection,
    )
    paths = []
    offset = 0
    for token_count in token_counts:
        paths.append(all_emb[offset : offset + token_count])
        offset += token_count
    return all_emb, paths


def save_figure(fig: Any, path: Path, *, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        fig.show()


def q_capture_label(args: argparse.Namespace) -> str:
    stage = getattr(args, "q_capture_stage", "")
    activation_label = activation_space_label(getattr(args, "activation_space", "q"))
    if stage == Q_STAGE_PRE_ROPE:
        return f"pre-RoPE {activation_label} projection output"
    if stage == Q_STAGE_POST_ROPE:
        return "post-RoPE Q, pre-score"
    return str(stage or f"{activation_label} projection output")


def q_capture_subtitle(args: argparse.Namespace) -> str:
    return f"Q capture: {q_capture_label(args)}; pool_last_k={args.pool_last_k}"


def plot_head_manifolds(
    np: Any,
    plt: Any,
    final_q_all: Any,
    labels: Any,
    class_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    layer_idx = args.target_layer
    num_heads = final_q_all.shape[2]
    rows = []
    for head_idx in range(num_heads):
        x = final_q_all[:, layer_idx, head_idx, :]
        rows.append(
            {
                "layer": layer_idx,
                "head": head_idx,
                "silhouette_cosine": safe_silhouette(np, x, labels),
            }
        )
    rows.sort(key=score_sort_key, reverse=True)

    cols = min(4, num_heads)
    rows_n = int(math.ceil(num_heads / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.5 * rows_n), squeeze=False)
    axes_flat = axes.reshape(-1)
    score_by_head = {int(row["head"]): float(row["silhouette_cosine"]) for row in rows}
    for head_idx in range(num_heads):
        ax = axes_flat[head_idx]
        x = final_q_all[:, layer_idx, head_idx, :]
        emb = compute_2d(
            np,
            x,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
            projection=args.projection,
        )
        for class_idx, class_name in enumerate(class_names):
            idx = labels == class_idx
            ax.scatter(
                emb[idx, 0],
                emb[idx, 1],
                c=class_color(class_idx),
                label=class_name if head_idx == 0 else None,
                s=60,
                alpha=0.85,
            )
        ax.set_title(f"Layer {layer_idx} / Head {head_idx}\nsil={score_by_head[head_idx]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.25)
    for index in range(num_heads, len(axes_flat)):
        axes_flat[index].axis("off")
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=max(1, min(3, len(class_names))),
    )
    fig.suptitle(
        f"Q-Space Manifolds across Heads - Layer {layer_idx}\n{q_capture_subtitle(args)}",
        y=1.10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir / f"head_manifolds_layer_{layer_idx}.png", show=args.show)
    plt.close(fig)
    return rows


def layer_separation_rows(np: Any, final_q_all: Any, labels: Any, head_idx: int) -> list[dict[str, Any]]:
    n_layers = final_q_all.shape[1]
    return [
        {
            "layer": layer_idx,
            "head": head_idx,
            "silhouette_cosine": safe_silhouette(np, final_q_all[:, layer_idx, head_idx, :], labels),
        }
        for layer_idx in range(n_layers)
    ]


def layer_head_score_rows(np: Any, final_q_all: Any, labels: Any) -> list[dict[str, Any]]:
    n_layers = final_q_all.shape[1]
    num_heads = final_q_all.shape[2]
    rows = []
    for layer_idx in range(n_layers):
        for head_idx in range(num_heads):
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "silhouette_cosine": safe_silhouette(
                        np,
                        final_q_all[:, layer_idx, head_idx, :],
                        labels,
                    ),
                }
            )
    rows.sort(key=score_sort_key, reverse=True)
    return rows


def centered_rows(np: Any, x: Any) -> Any:
    x = np.asarray(x, dtype="float32")
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(np: Any, x: Any, y: Any) -> float:
    x_centered = centered_rows(np, x)
    y_centered = centered_rows(np, y)
    xy = x_centered.T @ y_centered
    xx = x_centered.T @ x_centered
    yy = y_centered.T @ y_centered
    numerator = float(np.sum(xy * xy))
    denominator = math.sqrt(float(np.sum(xx * xx)) * float(np.sum(yy * yy)))
    if denominator <= 1e-12:
        return float("nan")
    return numerator / denominator


def upper_triangle_values(np: Any, matrix: Any) -> Any:
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.shape[0] < 2:
        return np.asarray([], dtype="float32")
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def pearson_corr(np: Any, x: Any, y: Any) -> float:
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y, dtype="float32")
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def head_similarity_matrices(np: Any, final_q_all: Any, *, layer_idx: int) -> dict[str, Any]:
    x_layer = np.asarray(final_q_all[:, layer_idx, :, :], dtype="float32")
    num_heads = x_layer.shape[1]
    cka = np.eye(num_heads, dtype="float32")
    rsa = np.eye(num_heads, dtype="float32")
    distance_vectors = [
        upper_triangle_values(np, cosine_distance_matrix(np, x_layer[:, head_idx, :]))
        for head_idx in range(num_heads)
    ]
    for left in range(num_heads):
        for right in range(left + 1, num_heads):
            cka_value = linear_cka(np, x_layer[:, left, :], x_layer[:, right, :])
            rsa_value = pearson_corr(np, distance_vectors[left], distance_vectors[right])
            cka[left, right] = cka[right, left] = cka_value
            rsa[left, right] = rsa[right, left] = rsa_value
    return {"cka": cka, "rsa": rsa}


def matrix_rows(np: Any, matrix: Any) -> list[dict[str, Any]]:
    matrix = np.asarray(matrix)
    rows = []
    for row_idx in range(matrix.shape[0]):
        row = {"head": row_idx}
        for col_idx in range(matrix.shape[1]):
            row[f"head_{col_idx}"] = float(matrix[row_idx, col_idx])
        rows.append(row)
    return rows


def head_similarity_pair_rows(np: Any, cka: Any, rsa: Any, *, layer_idx: int) -> list[dict[str, Any]]:
    rows = []
    num_heads = cka.shape[0]
    for left in range(num_heads):
        for right in range(left + 1, num_heads):
            rows.append(
                {
                    "layer": layer_idx,
                    "head_left": left,
                    "head_right": right,
                    "linear_cka": float(cka[left, right]),
                    "rsa_cosine_distance_corr": float(rsa[left, right]),
                }
            )
    rows.sort(
        key=lambda row: row["linear_cka"] if math.isfinite(row["linear_cka"]) else -999.0,
        reverse=True,
    )
    return rows


def plot_head_similarity_heatmap(
    np: Any,
    plt: Any,
    matrix: Any,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    layer_idx: int,
    metric_name: str,
    file_stem: str,
) -> None:
    matrix = np.asarray(matrix, dtype="float32")
    fig, ax = plt.subplots(figsize=(8, 7))
    if "RSA" in metric_name:
        image = ax.imshow(matrix, aspect="equal", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    else:
        image = ax.imshow(matrix, aspect="equal", cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label=metric_name)
    ax.set_title(f"Head Similarity - {metric_name}\nLayer {layer_idx}; {q_capture_subtitle(args)}")
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.tight_layout()
    save_figure(fig, output_dir / f"{file_stem}_layer_{layer_idx}.png", show=args.show)
    plt.close(fig)


def write_head_similarity_outputs(
    np: Any,
    plt: Any | None,
    final_q_all: Any,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    layer_idx: int,
) -> dict[str, Any]:
    matrices = head_similarity_matrices(np, final_q_all, layer_idx=layer_idx)
    cka = matrices["cka"]
    rsa = matrices["rsa"]
    pair_rows = head_similarity_pair_rows(np, cka, rsa, layer_idx=layer_idx)
    write_csv_rows(output_dir / f"head_cka_matrix_layer_{layer_idx}.csv", matrix_rows(np, cka))
    write_csv_rows(output_dir / f"head_rsa_matrix_layer_{layer_idx}.csv", matrix_rows(np, rsa))
    write_csv_rows(output_dir / f"head_similarity_pairs_layer_{layer_idx}.csv", pair_rows)
    if plt is not None:
        plot_head_similarity_heatmap(
            np,
            plt,
            cka,
            output_dir,
            args,
            layer_idx=layer_idx,
            metric_name="Linear CKA",
            file_stem="head_cka_heatmap",
        )
        plot_head_similarity_heatmap(
            np,
            plt,
            rsa,
            output_dir,
            args,
            layer_idx=layer_idx,
            metric_name="RSA cosine-distance correlation",
            file_stem="head_rsa_heatmap",
        )
    return {
        "layer": layer_idx,
        "top_pairs_by_cka": pair_rows[: min(10, len(pair_rows))],
        "mean_offdiag_cka": float(np.mean(upper_triangle_values(np, cka))) if cka.shape[0] > 1 else float("nan"),
        "mean_offdiag_rsa": float(np.mean(upper_triangle_values(np, rsa))) if rsa.shape[0] > 1 else float("nan"),
    }


def resolve_head_similarity_layers(
    args: argparse.Namespace,
    *,
    n_layers: int,
    detail_layer_heads: Sequence[dict[str, Any]],
    best_layer_head: dict[str, Any] | None,
) -> list[int]:
    spec = str(args.head_similarity_layers or "detail")
    if spec.strip().lower() == "all":
        return list(range(n_layers))
    layers: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part == "target":
            layers.add(int(args.target_layer))
        elif part == "detail":
            layers.update(int(pair["layer"]) for pair in detail_layer_heads)
        elif part == "best":
            if best_layer_head is not None:
                layers.add(int(best_layer_head["layer"]))
        else:
            try:
                layer_idx = int(part.replace("layer", "").replace("l", ""))
            except ValueError as exc:
                raise SystemExit(
                    f"invalid --head-similarity-layers item {raw_part!r}; "
                    "use target, detail, best, all, or integer layer indices"
                ) from exc
            if not (0 <= layer_idx < n_layers):
                raise SystemExit(f"--head-similarity-layers layer must be in [0, {n_layers - 1}], got {layer_idx}")
            layers.add(layer_idx)
    return sorted(layers)


def label_permutation_rows(
    np: Any,
    final_q_all: Any,
    labels: Any,
    probes: Sequence[dict[str, Any]],
    *,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n_permutations <= 0:
        return []
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    rows = []
    for probe in probes:
        layer_idx = int(probe["layer"])
        head_idx = int(probe["head"])
        x = final_q_all[:, layer_idx, head_idx, :]
        summary = silhouette_permutation_summary(
            np,
            x,
            labels_array,
            n_permutations=n_permutations,
            rng=rng,
        )
        row = {
            "layer": layer_idx,
            "head": head_idx,
            "reason": probe.get("reason", ""),
            **summary,
            "n_permutations": n_permutations,
            "seed": seed,
        }
        if "rank" in probe:
            row["rank"] = probe["rank"]
        rows.append(row)
    return rows


def silhouette_permutation_summary(
    np: Any,
    x: Any,
    labels_array: Any,
    *,
    n_permutations: int,
    rng: Any,
) -> dict[str, Any]:
    dist = cosine_distance_matrix(np, x)
    actual = silhouette_from_distance(np, dist, labels_array)
    null_scores = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels_array)
        null_scores.append(silhouette_from_distance(np, dist, shuffled))
    finite_null = [float(score) for score in null_scores if math.isfinite(float(score))]
    if finite_null and math.isfinite(float(actual)):
        null_mean = float(np.mean(finite_null))
        null_std = float(np.std(finite_null))
        null_max = float(np.max(finite_null))
        p_ge_actual = (1 + sum(score >= float(actual) for score in finite_null)) / (len(finite_null) + 1)
        z_score = (float(actual) - null_mean) / null_std if null_std > 1e-12 else float("nan")
    else:
        null_mean = float("nan")
        null_std = float("nan")
        null_max = float("nan")
        p_ge_actual = float("nan")
        z_score = float("nan")
    return {
        "actual_silhouette_cosine": actual,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_z_score": z_score,
        "null_max": null_max,
        "p_ge_actual": p_ge_actual,
        "finite_null_count": len(finite_null),
    }


def top_layer_head_permutation_rows(
    np: Any,
    final_q_all: Any,
    labels: Any,
    layer_head_scores: Sequence[dict[str, Any]],
    *,
    rank_limit: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if rank_limit <= 0 or n_permutations <= 0:
        return []
    probes = []
    for rank, row in enumerate(layer_head_scores[:rank_limit], start=1):
        probes.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "rank": rank,
                "reason": f"top_layer_head_rank_{rank}",
            }
        )
    return label_permutation_rows(
        np,
        final_q_all,
        labels,
        probes,
        n_permutations=n_permutations,
        seed=seed,
    )


def permutation_rows_by_layer_head(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    lookup = {}
    for row in rows:
        lookup[(int(row["layer"]), int(row["head"]))] = row
    return lookup


def silhouette_null_columns(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "silhouette_null_mean": "",
            "silhouette_null_std": "",
            "silhouette_null_z_score": "",
            "silhouette_null_max": "",
            "silhouette_p_ge_actual": "",
            "silhouette_null_n_permutations": "",
            "silhouette_finite_null_count": "",
        }
    return {
        "silhouette_null_mean": row.get("null_mean", ""),
        "silhouette_null_std": row.get("null_std", ""),
        "silhouette_null_z_score": row.get("null_z_score", ""),
        "silhouette_null_max": row.get("null_max", ""),
        "silhouette_p_ge_actual": row.get("p_ge_actual", ""),
        "silhouette_null_n_permutations": row.get("n_permutations", ""),
        "silhouette_finite_null_count": row.get("finite_null_count", ""),
    }


def plot_layer_head_heatmap(
    np: Any,
    plt: Any,
    rows: Sequence[dict[str, Any]],
    n_layers: int,
    num_heads: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    grid = np.full((n_layers, num_heads), np.nan, dtype="float32")
    for row in rows:
        grid[int(row["layer"]), int(row["head"])] = float(row["silhouette_cosine"])

    fig_width = max(8, num_heads * 0.55)
    fig_height = max(5, n_layers * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    finite = grid[np.isfinite(grid)]
    if finite.size:
        limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
        image = ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(grid, aspect="auto", cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="Silhouette score / cosine")
    ax.scatter([args.target_head], [args.target_layer], marker="s", s=90, facecolors="none", edgecolors="black", linewidths=1.5)
    ax.set_title(f"Layer x Head Q-Space Separability\n{q_capture_subtitle(args)}")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(n_layers))
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "layer_head_separability_heatmap.png", show=args.show)
    plt.close(fig)


def plot_layer_separation_curve(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    head_idx: int,
    focus_layer_idx: int,
) -> None:
    layers = [row["layer"] for row in rows]
    scores = [row["silhouette_cosine"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, scores, marker="o")
    ax.axvline(focus_layer_idx, linestyle="--", alpha=0.5)
    ax.set_title(f"Layer-wise Q Separability - Head {head_idx}\n{q_capture_subtitle(args)}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Silhouette score / cosine")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / f"layer_separation_head_{head_idx}_focus_layer_{focus_layer_idx}.png",
        show=args.show,
    )
    plt.close(fig)


def plot_layer_trajectory(
    np: Any,
    plt: Any,
    final_q_all: Any,
    labels: Any,
    class_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    head_idx: int,
    focus_layer_idx: int,
) -> None:
    sample_count, n_layers, _, head_dim = final_q_all.shape
    x = final_q_all[:, :, head_idx, :].reshape(sample_count * n_layers, head_dim)
    emb = compute_2d(
        np,
        x,
        n_neighbors=12,
        min_dist=0.25,
        metric=args.metric,
        random_state=args.random_state,
        projection=args.projection,
    ).reshape(sample_count, n_layers, 2)

    fig, ax = plt.subplots(figsize=(9, 7))
    if args.show_individual_trajectories:
        for sample_idx in plot_sample_indices(np, sample_count, args):
            ax.plot(emb[sample_idx, :, 0], emb[sample_idx, :, 1], alpha=0.15, linewidth=1)
    for class_idx, class_name in enumerate(class_names):
        idx = labels == class_idx
        mean_path = emb[idx].mean(axis=0)
        ax.plot(
            mean_path[:, 0],
            mean_path[:, 1],
            marker="o",
            linewidth=3,
            color=class_color(class_idx),
            label=class_name,
        )
        for layer_idx in range(n_layers):
            if layer_idx in {0, focus_layer_idx, n_layers - 1}:
                ax.text(mean_path[layer_idx, 0], mean_path[layer_idx, 1], f"L{layer_idx}", fontsize=10)
    ax.set_title(f"Layer Trajectory of Q-Space - Head {head_idx}\n{q_capture_subtitle(args)}")
    ax.set_xlabel(projection_axis_label(args, 1))
    ax.set_ylabel(projection_axis_label(args, 2))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / f"layer_trajectory_head_{head_idx}_focus_layer_{focus_layer_idx}.png",
        show=args.show,
    )
    plt.close(fig)


def plot_layer_trajectory_3d(
    np: Any,
    plt: Any,
    final_q_all: Any,
    labels: Any,
    class_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    head_idx: int,
    focus_layer_idx: int,
) -> None:
    sample_count, n_layers, _, head_dim = final_q_all.shape
    x = final_q_all[:, :, head_idx, :].reshape(sample_count * n_layers, head_dim)
    emb = compute_nd(
        np,
        x,
        n_components=3,
        n_neighbors=12,
        min_dist=0.25,
        metric=args.metric,
        random_state=args.random_state,
        projection=args.projection,
    ).reshape(sample_count, n_layers, 3)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if args.show_individual_trajectories:
        for sample_idx in plot_sample_indices(np, sample_count, args):
            ax.plot(
                emb[sample_idx, :, 0],
                emb[sample_idx, :, 1],
                emb[sample_idx, :, 2],
                alpha=0.12,
                linewidth=1,
            )
    for class_idx, class_name in enumerate(class_names):
        idx = labels == class_idx
        mean_path = emb[idx].mean(axis=0)
        ax.plot(
            mean_path[:, 0],
            mean_path[:, 1],
            mean_path[:, 2],
            marker="o",
            linewidth=3,
            color=class_color(class_idx),
            label=class_name,
        )
        for layer_idx in range(n_layers):
            if layer_idx in {0, focus_layer_idx, n_layers - 1}:
                ax.text(
                    mean_path[layer_idx, 0],
                    mean_path[layer_idx, 1],
                    mean_path[layer_idx, 2],
                    f"L{layer_idx}",
                    fontsize=10,
                )
    ax.view_init(elev=args.plot_3d_elev, azim=args.plot_3d_azim)
    ax.set_title(f"3D Layer Trajectory of Q-Space - Head {head_idx}\n{q_capture_subtitle(args)}")
    ax.set_xlabel(projection_axis_label(args, 1))
    ax.set_ylabel(projection_axis_label(args, 2))
    ax.set_zlabel(projection_axis_label(args, 3))
    ax.legend()
    fig.tight_layout()
    save_figure(
        fig,
        output_dir / f"layer_trajectory_3d_head_{head_idx}_focus_layer_{focus_layer_idx}.png",
        show=args.show,
    )
    plt.close(fig)


def is_special_token(token: str) -> bool:
    stripped = token.strip()
    if stripped in SPECIAL_TOKEN_STRINGS:
        return True
    # Common tokenizer wrappers use angle-bracket sentinels for special tokens.
    return stripped.startswith("<|") and stripped.endswith("|>")


def flow_token_indices(tokens: Sequence[str], args: argparse.Namespace) -> list[int]:
    indices = list(range(min(args.flow_start_token_index, len(tokens)), len(tokens)))
    if args.drop_special_tokens:
        indices = [idx for idx in indices if not is_special_token(str(tokens[idx]))]
    return indices


def project_token_qs(
    np: Any,
    token_q_records: Sequence[Any],
    token_records: Sequence[Sequence[str]],
    labels: Any,
    class_names: Sequence[str],
    texts: Sequence[str],
    args: argparse.Namespace,
    *,
    layer_idx: int,
    head_idx: int,
) -> tuple[Any, list[Any], list[Any], list[list[dict[str, Any]]], list[dict[str, Any]]]:
    all_q = []
    meta_rows = []
    highd_paths = []
    path_token_records: list[list[dict[str, Any]]] = []
    token_counts: list[int] = []
    for sample_idx, q_by_layer in enumerate(token_q_records):
        q_seq = np.asarray(q_by_layer[layer_idx, head_idx, :, :])
        tokens = token_records[sample_idx]
        selected_indices = [
            idx for idx in flow_token_indices(tokens, args) if idx < q_seq.shape[0]
        ]
        token_counts.append(len(selected_indices))
        sample_token_records = []
        for flow_position, token_idx in enumerate(selected_indices):
            all_q.append(q_seq[token_idx])
            label = int(labels[sample_idx])
            record = {
                "sample_idx": sample_idx,
                "flow_position": flow_position,
                "token_idx": token_idx,
                "token": token_records[sample_idx][token_idx],
                "label": label,
                "class_name": class_names[label],
                "text": texts[sample_idx],
            }
            sample_token_records.append(record)
            meta_rows.append(record)
        highd_paths.append(q_seq[selected_indices])
        path_token_records.append(sample_token_records)
    if not all_q:
        raise SystemExit("token-flow filter removed every token; relax --flow-start-token-index or --drop-special-tokens")
    all_q_array = np.stack(all_q, axis=0)
    all_emb = compute_2d(
        np,
        all_q_array,
        n_neighbors=10,
        min_dist=0.2,
        metric=args.metric,
        random_state=args.random_state,
        projection=args.projection,
    )
    paths = []
    offset = 0
    for token_count in token_counts:
        paths.append(all_emb[offset : offset + token_count])
        offset += token_count
    return all_emb, paths, highd_paths, path_token_records, meta_rows


def flow_metrics(np: Any, points: Any) -> dict[str, float]:
    points = np.asarray(points, dtype="float32")
    if len(points) < 2:
        return {
            "path_length": float("nan"),
            "chord_length": float("nan"),
            "straightness": float("nan"),
            "mean_turn_deg": float("nan"),
            "signed_area": float("nan"),
            "flow_token_count": int(len(points)),
        }
    diffs = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(step_lengths.sum())
    chord_length = float(np.linalg.norm(points[-1] - points[0]))
    straightness = chord_length / (path_length + 1e-12)
    turn_angles = []
    for a, b in zip(diffs[:-1], diffs[1:]):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-12:
            cosine = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
            turn_angles.append(float(np.degrees(np.arccos(cosine))))
    mean_turn_deg = float(np.mean(turn_angles)) if turn_angles else float("nan")
    centered = points - points.mean(axis=0, keepdims=True)
    signed_area = 0.5 * float(
        np.sum(centered[:-1, 0] * centered[1:, 1] - centered[:-1, 1] * centered[1:, 0])
    )
    return {
        "path_length": path_length,
        "chord_length": chord_length,
        "straightness": straightness,
        "mean_turn_deg": mean_turn_deg,
        "signed_area": signed_area,
        "flow_token_count": int(len(points)),
    }


def build_flow_rows(
    np: Any,
    paths: Sequence[Any],
    labels: Any,
    class_names: Sequence[str],
    texts: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for sample_idx, path in enumerate(paths):
        label = int(labels[sample_idx])
        rows.append(
            {
                "sample_idx": sample_idx,
                "label": label,
                "class_name": class_names[label],
                "text": texts[sample_idx],
                **flow_metrics(np, path),
            }
        )
    return rows


def consecutive_cosine_distances(np: Any, points: Any) -> Any:
    points = np.asarray(points, dtype="float32")
    if len(points) < 2:
        return np.asarray([], dtype="float32")
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    unit = points / safe
    cosines = np.sum(unit[:-1] * unit[1:], axis=1)
    return 1.0 - np.clip(cosines, -1.0, 1.0)


def highd_flow_metrics(np: Any, points: Any) -> dict[str, float]:
    points = np.asarray(points, dtype="float32")
    token_count = int(len(points))
    if token_count < 2:
        return {
            "highd_flow_token_count": token_count,
            "highd_path_length_cosine": float("nan"),
            "highd_chord_cosine": float("nan"),
            "highd_straightness_cosine": float("nan"),
            "highd_mean_step_cosine": float("nan"),
            "highd_max_step_cosine": float("nan"),
            "highd_first_step_cosine": float("nan"),
            "highd_last_step_cosine": float("nan"),
            "highd_path_length_euclidean": float("nan"),
            "highd_chord_euclidean": float("nan"),
            "highd_straightness_euclidean": float("nan"),
            "highd_mean_turn_deg": float("nan"),
            "highd_max_turn_deg": float("nan"),
            "highd_start_q_norm": float("nan"),
            "highd_end_q_norm": float("nan"),
            "highd_mean_q_norm": float("nan"),
        }

    step_cosine = consecutive_cosine_distances(np, points)
    path_length_cosine = float(step_cosine.sum())
    chord_cosine = float(cosine_distance_matrix(np, np.stack([points[0], points[-1]]))[0, 1])
    diffs = np.diff(points, axis=0)
    step_euclidean = np.linalg.norm(diffs, axis=1)
    path_length_euclidean = float(step_euclidean.sum())
    chord_euclidean = float(np.linalg.norm(points[-1] - points[0]))
    turn_angles = []
    for left, right in zip(diffs[:-1], diffs[1:]):
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        if denom > 1e-12:
            cosine = np.clip(np.dot(left, right) / denom, -1.0, 1.0)
            turn_angles.append(float(np.degrees(np.arccos(cosine))))
    norms = np.linalg.norm(points, axis=1)
    return {
        "highd_flow_token_count": token_count,
        "highd_path_length_cosine": path_length_cosine,
        "highd_chord_cosine": chord_cosine,
        "highd_straightness_cosine": chord_cosine / (path_length_cosine + 1e-12),
        "highd_mean_step_cosine": float(np.mean(step_cosine)),
        "highd_max_step_cosine": float(np.max(step_cosine)),
        "highd_first_step_cosine": float(step_cosine[0]),
        "highd_last_step_cosine": float(step_cosine[-1]),
        "highd_path_length_euclidean": path_length_euclidean,
        "highd_chord_euclidean": chord_euclidean,
        "highd_straightness_euclidean": chord_euclidean / (path_length_euclidean + 1e-12),
        "highd_mean_turn_deg": float(np.mean(turn_angles)) if turn_angles else float("nan"),
        "highd_max_turn_deg": float(np.max(turn_angles)) if turn_angles else float("nan"),
        "highd_start_q_norm": float(norms[0]),
        "highd_end_q_norm": float(norms[-1]),
        "highd_mean_q_norm": float(np.mean(norms)),
    }


def build_highd_flow_rows(
    np: Any,
    highd_paths: Sequence[Any],
    labels: Any,
    class_names: Sequence[str],
    texts: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for sample_idx, path in enumerate(highd_paths):
        label = int(labels[sample_idx])
        rows.append(
            {
                "sample_idx": sample_idx,
                "label": label,
                "class_name": class_names[label],
                "text": texts[sample_idx],
                **highd_flow_metrics(np, path),
            }
        )
    return rows


def summarize_metric_rows(
    np: Any,
    rows: Sequence[dict[str, Any]],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if metric_name in row and math.isfinite(float(row[metric_name]))
        ]
        if not values:
            summary[metric_name] = {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
            continue
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def highd_flow_summary(np: Any, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "highd_path_length_cosine",
        "highd_chord_cosine",
        "highd_straightness_cosine",
        "highd_mean_step_cosine",
        "highd_mean_turn_deg",
        "highd_path_length_euclidean",
        "highd_chord_euclidean",
        "highd_straightness_euclidean",
    ]
    by_class = {}
    for class_name in sorted({str(row["class_name"]) for row in rows}):
        class_rows = [row for row in rows if str(row["class_name"]) == class_name]
        by_class[class_name] = summarize_metric_rows(np, class_rows, metrics)
    return {
        "metric_status": "original_high_dimensional_q_space_summary",
        "overall": summarize_metric_rows(np, rows, metrics),
        "by_class": by_class,
    }


def projection_diagnostic_row(
    np: Any,
    x_highd: Any,
    emb_2d: Any,
    labels: Any,
    class_names: Sequence[str],
    args: argparse.Namespace,
    *,
    layer_idx: int,
    head_idx: int,
    reason: str,
) -> dict[str, Any]:
    highd_cosine_dist = cosine_distance_matrix(np, x_highd)
    projected_euclidean_dist = euclidean_distance_matrix(np, emb_2d)
    pca_ratios = pca_explained_variance_ratio(np, x_highd, n_components=2)
    while len(pca_ratios) < 2:
        pca_ratios.append(float("nan"))
    centroid_summary = centroid_distance_summary(np, x_highd, labels, class_names)
    dispersions = [
        float(row["within_class_mean_cosine_distance"])
        for row in centroid_summary["class_dispersion"]
        if math.isfinite(float(row["within_class_mean_cosine_distance"]))
    ]
    return {
        "layer": layer_idx,
        "head": head_idx,
        "reason": reason,
        "projection": args.projection,
        "highd_silhouette_cosine": silhouette_from_distance(np, highd_cosine_dist, labels),
        "projected_silhouette_euclidean": silhouette_from_distance(np, projected_euclidean_dist, labels),
        "knn_recall_highd_cosine_to_projected_euclidean": knn_recall(
            np,
            highd_cosine_dist,
            projected_euclidean_dist,
            args.projection_knn_k,
        ),
        "projection_knn_k": args.projection_knn_k,
        "pca_explained_variance_1": pca_ratios[0],
        "pca_explained_variance_2": pca_ratios[1],
        "pca_explained_variance_2d_sum": float(sum(value for value in pca_ratios[:2] if math.isfinite(value))),
        "highd_mean_centroid_pair_cosine_distance": centroid_summary["mean_centroid_pair_cosine_distance"],
        "highd_mean_within_class_cosine_distance": float(np.mean(dispersions)) if dispersions else float("nan"),
    }


def ridge_loo_probe(np: Any, x: Any, labels: Any, *, alpha: float) -> dict[str, Any]:
    x = np.asarray(x, dtype="float32")
    labels_np = np.asarray(labels)
    class_values = sorted(int(value) for value in set(labels_np))
    if x.ndim != 2 or len(x) < 2 or len(class_values) < 2:
        return {
            "accuracy": float("nan"),
            "macro_accuracy": float("nan"),
            "predictions": [],
            "confusion": [],
            "class_accuracy": {},
        }

    predictions = []
    for held_out_idx in range(len(x)):
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[held_out_idx] = False
        x_train = x[train_mask]
        y_train = labels_np[train_mask]
        x_test = x[held_out_idx : held_out_idx + 1]
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_train_aug = np.concatenate([np.ones((len(x_train), 1), dtype="float32"), x_train], axis=1)
        x_test_aug = np.concatenate([np.ones((len(x_test), 1), dtype="float32"), x_test], axis=1)
        target = np.zeros((len(x_train), len(class_values)), dtype="float32")
        for row_idx, label in enumerate(y_train):
            target[row_idx, class_values.index(int(label))] = 1.0
        reg = float(alpha) * np.eye(x_train_aug.shape[1], dtype="float32")
        reg[0, 0] = 0.0
        try:
            weights = np.linalg.solve(x_train_aug.T @ x_train_aug + reg, x_train_aug.T @ target)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(x_train_aug.T @ x_train_aug + reg) @ x_train_aug.T @ target
        scores = x_test_aug @ weights
        predictions.append(class_values[int(np.argmax(scores[0]))])

    predictions_np = np.asarray(predictions)
    accuracy = float(np.mean(predictions_np == labels_np))
    confusion = np.zeros((len(class_values), len(class_values)), dtype=int)
    class_accuracy: dict[str, float] = {}
    for actual, predicted in zip(labels_np, predictions_np):
        confusion[class_values.index(int(actual)), class_values.index(int(predicted))] += 1
    per_class = []
    for class_idx, class_value in enumerate(class_values):
        total = int(confusion[class_idx].sum())
        acc = float(confusion[class_idx, class_idx] / total) if total else float("nan")
        class_accuracy[str(class_value)] = acc
        if math.isfinite(acc):
            per_class.append(acc)
    return {
        "accuracy": accuracy,
        "macro_accuracy": float(np.mean(per_class)) if per_class else float("nan"),
        "predictions": [int(value) for value in predictions],
        "confusion": confusion.tolist(),
        "class_accuracy": class_accuracy,
    }


def linear_probe_rows(
    np: Any,
    final_q_all: Any,
    labels: Any,
    probes: Sequence[dict[str, Any]],
    *,
    alpha: float,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    rows = []
    for probe in probes:
        layer_idx = int(probe["layer"])
        head_idx = int(probe["head"])
        x = final_q_all[:, layer_idx, head_idx, :]
        actual = ridge_loo_probe(np, x, labels_array, alpha=alpha)
        null_accuracies = []
        for _ in range(n_permutations):
            shuffled = rng.permutation(labels_array)
            shuffled_result = ridge_loo_probe(np, x, shuffled, alpha=alpha)
            acc = float(shuffled_result["accuracy"])
            if math.isfinite(acc):
                null_accuracies.append(acc)
        actual_accuracy = float(actual["accuracy"])
        if null_accuracies and math.isfinite(actual_accuracy):
            null_mean = float(np.mean(null_accuracies))
            null_std = float(np.std(null_accuracies))
            null_max = float(np.max(null_accuracies))
            p_ge_actual = (1 + sum(acc >= actual_accuracy for acc in null_accuracies)) / (len(null_accuracies) + 1)
        else:
            null_mean = float("nan")
            null_std = float("nan")
            null_max = float("nan")
            p_ge_actual = float("nan")
        rows.append(
            {
                "layer": layer_idx,
                "head": head_idx,
                "reason": probe.get("reason", ""),
                "linear_probe_accuracy_loo": actual_accuracy,
                "linear_probe_macro_accuracy_loo": actual["macro_accuracy"],
                "null_accuracy_mean": null_mean,
                "null_accuracy_std": null_std,
                "null_accuracy_max": null_max,
                "p_ge_actual_accuracy": p_ge_actual,
                "n_permutations": n_permutations,
                "finite_null_count": len(null_accuracies),
                "linear_probe_alpha": alpha,
                "confusion_json": json.dumps(actual["confusion"]),
                "class_accuracy_json": json.dumps(actual["class_accuracy"], sort_keys=True),
            }
        )
    return rows


def plot_query_flow(
    np: Any,
    plt: Any,
    all_emb: Any,
    paths: Sequence[Any],
    token_record_paths: Sequence[Sequence[dict[str, Any]]],
    labels: Any,
    class_names: Sequence[str],
    texts: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    layer_idx: int,
    head_idx: int,
    text_index: int | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    scale = float(np.ptp(all_emb, axis=0).mean()) if len(all_emb) else 1.0
    head_width = max(1e-6, 0.015 * scale)
    color_mode = args.color_flow_by
    colorbar = None
    if color_mode != "class":
        all_values = [
            float(record[color_mode])
            for records in token_record_paths
            for record in records
        ]
        vmin = min(all_values) if all_values else 0.0
        vmax = max(all_values) if all_values else 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
        cmap = plt.get_cmap("viridis")
    if text_index is None:
        seen = set()
        for sample_idx in plot_sample_indices(np, len(paths), args):
            path = paths[sample_idx]
            if len(path) == 0:
                continue
            label = int(labels[sample_idx])
            class_name = class_names[label]
            if color_mode == "class":
                legend_label = class_name if class_name not in seen else None
                seen.add(class_name)
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    marker="o",
                    alpha=0.35,
                    linewidth=1.5,
                    color=class_color(label),
                    label=legend_label,
                )
            else:
                values = [float(record[color_mode]) for record in token_record_paths[sample_idx]]
                ax.plot(path[:, 0], path[:, 1], alpha=0.18, linewidth=1.2, color="0.45")
                colorbar = ax.scatter(
                    path[:, 0],
                    path[:, 1],
                    c=values,
                    cmap=cmap,
                    norm=norm,
                    s=45,
                    alpha=0.75,
                )
            for token_idx in range(len(path) - 1):
                dx, dy = path[token_idx + 1] - path[token_idx]
                ax.arrow(
                    path[token_idx, 0],
                    path[token_idx, 1],
                    dx,
                    dy,
                    alpha=0.18,
                    color=class_color(label) if color_mode == "class" else "0.45",
                    length_includes_head=True,
                    head_width=head_width,
                )
        ax.set_title(
            f"Question Stance Field / Token Q-flow\n"
            f"Layer {layer_idx}, Head {head_idx}\n{q_capture_subtitle(args)}"
        )
        if color_mode == "class":
            ax.legend()
        elif colorbar is not None:
            fig.colorbar(colorbar, ax=ax, label=color_mode)
        suffix = "" if color_mode == "class" else f"_by_{color_mode}"
        output_name = f"query_flow_layer_{layer_idx}_head_{head_idx}_all{suffix}.png"
    else:
        path = paths[text_index]
        if len(path) == 0:
            ax.text(0.5, 0.5, "No tokens after flow filter", transform=ax.transAxes, ha="center")
        else:
            if color_mode == "class":
                ax.plot(path[:, 0], path[:, 1], marker="o", linewidth=2.5)
            else:
                values = [float(record[color_mode]) for record in token_record_paths[text_index]]
                ax.plot(path[:, 0], path[:, 1], linewidth=2.0, alpha=0.55, color="0.45")
                colorbar = ax.scatter(
                    path[:, 0],
                    path[:, 1],
                    c=values,
                    cmap=cmap,
                    norm=norm,
                    s=70,
                    alpha=0.9,
                )
            for token_idx in range(len(path) - 1):
                dx, dy = path[token_idx + 1] - path[token_idx]
                ax.arrow(
                    path[token_idx, 0],
                    path[token_idx, 1],
                    dx,
                    dy,
                    alpha=0.5,
                    length_includes_head=True,
                    head_width=head_width,
                )
            for token_idx, record in enumerate(token_record_paths[text_index]):
                token_label = f"{record['token_idx']}:{record['token']}"
                ax.text(path[token_idx, 0], path[token_idx, 1], token_label, fontsize=9)
            if color_mode != "class" and colorbar is not None:
                fig.colorbar(colorbar, ax=ax, label=color_mode)
        ax.set_title(
            f"Token-level Q-flow\nLayer {layer_idx}, Head {head_idx}\n{texts[text_index]}"
            f"\n{q_capture_subtitle(args)}"
        )
        suffix = "" if color_mode == "class" else f"_by_{color_mode}"
        output_name = f"query_flow_layer_{layer_idx}_head_{head_idx}_text_{text_index}{suffix}.png"
    ax.set_xlabel(projection_axis_label(args, 1))
    ax.set_ylabel(projection_axis_label(args, 2))
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, output_dir / output_name, show=args.show)
    plt.close(fig)


def plot_query_flow_3d(
    np: Any,
    plt: Any,
    all_emb: Any,
    paths: Sequence[Any],
    token_record_paths: Sequence[Sequence[dict[str, Any]]],
    labels: Any,
    class_names: Sequence[str],
    texts: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    layer_idx: int,
    head_idx: int,
    text_index: int | None,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    color_mode = args.color_flow_by
    colorbar = None
    if color_mode != "class":
        all_values = [
            float(record[color_mode])
            for records in token_record_paths
            for record in records
        ]
        vmin = min(all_values) if all_values else 0.0
        vmax = max(all_values) if all_values else 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
        cmap = plt.get_cmap("viridis")
    if text_index is None:
        seen = set()
        for sample_idx in plot_sample_indices(np, len(paths), args):
            path = paths[sample_idx]
            if len(path) == 0:
                continue
            label = int(labels[sample_idx])
            class_name = class_names[label]
            if color_mode == "class":
                legend_label = class_name if class_name not in seen else None
                seen.add(class_name)
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    marker="o",
                    markersize=3,
                    alpha=0.28,
                    linewidth=1.2,
                    color=class_color(label),
                    label=legend_label,
                )
            else:
                values = [float(record[color_mode]) for record in token_record_paths[sample_idx]]
                ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.16, linewidth=1.0, color="0.45")
                colorbar = ax.scatter(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    c=values,
                    cmap=cmap,
                    norm=norm,
                    s=24,
                    alpha=0.75,
                )
            ax.scatter(
                path[0, 0],
                path[0, 1],
                path[0, 2],
                marker="s",
                s=18,
                color=class_color(label) if color_mode == "class" else "0.25",
                alpha=0.55,
            )
            ax.scatter(
                path[-1, 0],
                path[-1, 1],
                path[-1, 2],
                marker="^",
                s=22,
                color=class_color(label) if color_mode == "class" else "0.25",
                alpha=0.75,
            )
        ax.set_title(
            f"3D Question Stance Field / Token Q-flow\n"
            f"Layer {layer_idx}, Head {head_idx}\n{q_capture_subtitle(args)}"
        )
        if color_mode == "class":
            ax.legend()
        elif colorbar is not None:
            fig.colorbar(colorbar, ax=ax, label=color_mode, shrink=0.7)
        suffix = "" if color_mode == "class" else f"_by_{color_mode}"
        output_name = f"query_flow_3d_layer_{layer_idx}_head_{head_idx}_all{suffix}.png"
    else:
        path = paths[text_index]
        if len(path) == 0:
            ax.text2D(0.5, 0.5, "No tokens after flow filter", transform=ax.transAxes, ha="center")
        else:
            if color_mode == "class":
                ax.plot(path[:, 0], path[:, 1], path[:, 2], marker="o", linewidth=2.2)
            else:
                values = [float(record[color_mode]) for record in token_record_paths[text_index]]
                ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=1.8, alpha=0.55, color="0.45")
                colorbar = ax.scatter(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    c=values,
                    cmap=cmap,
                    norm=norm,
                    s=48,
                    alpha=0.9,
                )
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], marker="s", s=45, alpha=0.75)
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], marker="^", s=55, alpha=0.9)
            for token_idx, record in enumerate(token_record_paths[text_index]):
                token_label = f"{record['token_idx']}:{record['token']}"
                ax.text(path[token_idx, 0], path[token_idx, 1], path[token_idx, 2], token_label, fontsize=8)
            if color_mode != "class" and colorbar is not None:
                fig.colorbar(colorbar, ax=ax, label=color_mode, shrink=0.7)
        ax.set_title(
            f"3D Token-level Q-flow\nLayer {layer_idx}, Head {head_idx}\n{texts[text_index]}"
            f"\n{q_capture_subtitle(args)}"
        )
        suffix = "" if color_mode == "class" else f"_by_{color_mode}"
        output_name = f"query_flow_3d_layer_{layer_idx}_head_{head_idx}_text_{text_index}{suffix}.png"
    ax.view_init(elev=args.plot_3d_elev, azim=args.plot_3d_azim)
    ax.set_xlabel(projection_axis_label(args, 1))
    ax.set_ylabel(projection_axis_label(args, 2))
    ax.set_zlabel(projection_axis_label(args, 3))
    fig.tight_layout()
    save_figure(fig, output_dir / output_name, show=args.show)
    plt.close(fig)


def compute_query_flow_field(
    np: Any,
    all_emb: Any,
    paths: Sequence[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    starts = []
    vecs = []
    for path in paths:
        if len(path) < 2:
            continue
        starts.append(path[:-1])
        vecs.append(np.diff(path, axis=0))
    if not starts:
        return {"summary": {"observed_cells": 0}}
    starts = np.vstack(starts)
    vecs = np.vstack(vecs)
    x_min, y_min = starts.min(axis=0)
    x_max, y_max = starts.max(axis=0)
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    grid_size = args.field_grid_size
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    u = np.zeros((grid_size, grid_size))
    v = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))
    ix = np.clip(np.digitize(starts[:, 0], x_bins) - 1, 0, grid_size - 1)
    iy = np.clip(np.digitize(starts[:, 1], y_bins) - 1, 0, grid_size - 1)
    for row_idx in range(len(starts)):
        x_i = ix[row_idx]
        y_i = iy[row_idx]
        u[y_i, x_i] += vecs[row_idx, 0]
        v[y_i, x_i] += vecs[row_idx, 1]
        counts[y_i, x_i] += 1
    mask = counts > 0
    u[mask] /= counts[mask]
    v[mask] /= counts[mask]
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size

    gradient_mask = np.zeros_like(mask, dtype=bool)
    curl = np.full((grid_size, grid_size), np.nan, dtype="float32")
    divergence = np.full((grid_size, grid_size), np.nan, dtype="float32")
    if grid_size >= 3:
        cross_observed = (
            mask[1:-1, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
        )
        gradient_mask[1:-1, 1:-1] = cross_observed
        if np.any(cross_observed):
            dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
            du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
            du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
            dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
            curl[1:-1, 1:-1] = dv_dx - du_dy
            divergence[1:-1, 1:-1] = du_dx + dv_dy

    gradient_cells = int(gradient_mask.sum())
    if gradient_cells:
        mean_abs_curl = float(np.nanmean(np.abs(curl[gradient_mask])))
        mean_divergence = float(np.nanmean(divergence[gradient_mask]))
        mean_abs_divergence = float(np.nanmean(np.abs(divergence[gradient_mask])))
    else:
        mean_abs_curl = float("nan")
        mean_divergence = float("nan")
        mean_abs_divergence = float("nan")
    return {
        "all_emb": all_emb,
        "x_centers": 0.5 * (x_bins[:-1] + x_bins[1:]),
        "y_centers": 0.5 * (y_bins[:-1] + y_bins[1:]),
        "u": u,
        "v": v,
        "mask": mask,
        "curl": curl,
        "divergence": divergence,
        "gradient_mask": gradient_mask,
        "summary": {
            "field_metric_status": "exploratory_projection_space_summary",
            "bin_edge_policy": "digitize_indices_clipped_to_grid_edges",
            "curl_divergence_method": "central_difference_on_cells_with_observed_cross_neighbors",
            "curl_divergence_caveat": (
                "curl/divergence are computed on the 2D projection field, not on the "
                "original Q-space; cells without observed left/right/up/down neighbors "
                "are excluded to avoid zero-fill boundary artifacts"
            ),
            "mean_abs_curl_on_evaluation_cells": mean_abs_curl,
            "mean_divergence_on_evaluation_cells": mean_divergence,
            "mean_abs_divergence_on_evaluation_cells": mean_abs_divergence,
            "observed_cells": int(mask.sum()),
            "gradient_evaluation_cells": gradient_cells,
        },
    }


def plot_query_flow_field(
    np: Any,
    plt: Any,
    field: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    layer_idx: int,
    head_idx: int,
) -> None:
    if "mask" not in field:
        return
    xg, yg = np.meshgrid(field["x_centers"], field["y_centers"])
    mask = field["mask"]
    fig, ax = plt.subplots(figsize=(8, 7))
    all_emb = field["all_emb"]
    ax.scatter(all_emb[:, 0], all_emb[:, 1], s=15, alpha=0.2)
    ax.quiver(
        xg[mask],
        yg[mask],
        field["u"][mask],
        field["v"][mask],
        angles="xy",
        scale_units="xy",
        scale=1,
        alpha=0.8,
    )
    ax.set_title(f"Coarse Q-flow Vector Field\nLayer {layer_idx}, Head {head_idx}\n{q_capture_subtitle(args)}")
    ax.set_xlabel(projection_axis_label(args, 1))
    ax.set_ylabel(projection_axis_label(args, 2))
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, output_dir / f"query_flow_field_layer_{layer_idx}_head_{head_idx}.png", show=args.show)
    plt.close(fig)


def analyze_bundle(args: argparse.Namespace, dataset: TextDataset, bundle: CaptureBundle) -> dict[str, Any]:
    np = load_numpy()
    labels = np.asarray(dataset.labels)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**vars(args))
    args.q_capture_stage = bundle.model_info.get("q_capture_stage", "")
    args.q_capture_position_note = bundle.model_info.get("q_capture_position_note", "")
    args.activation_space = bundle.model_info.get("activation_space", getattr(args, "activation_space", "q"))

    final_q_all = np.asarray(bundle.final_q_all)
    sample_count, n_layers, num_heads, head_dim = final_q_all.shape
    if args.target_layer_fraction is not None:
        args = argparse.Namespace(**vars(args))
        args.target_layer = int(round(float(args.target_layer_fraction) * (n_layers - 1)))
        args.target_layer = max(0, min(args.target_layer, n_layers - 1))
    if n_layers < 1:
        raise SystemExit("model produced no layers to analyze")
    if num_heads < 1:
        raise SystemExit("model produced no attention heads to analyze")
    if not (0 <= args.target_layer < n_layers):
        raise SystemExit(f"--target-layer must be in [0, {n_layers - 1}], got {args.target_layer}")
    if not (0 <= args.target_head < num_heads):
        raise SystemExit(f"--target-head must be in [0, {num_heads - 1}], got {args.target_head}")
    for label in labels:
        if int(label) >= len(dataset.class_names):
            raise SystemExit(f"label {label} has no class_names entry")

    np.savez_compressed(
        output_dir / "q_space_vectors.npz",
        final_q_all=final_q_all,
        labels=labels,
        texts=np.asarray(dataset.texts, dtype=object),
        class_names=np.asarray(dataset.class_names, dtype=object),
    )
    write_csv_rows(
        output_dir / "dataset_rows.csv",
        [
            {
                "sample_idx": idx,
                "label": int(label),
                "class_name": dataset.class_names[int(label)],
                "text": text,
            }
            for idx, (text, label) in enumerate(zip(dataset.texts, dataset.labels))
        ],
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_alias": getattr(args, "model_alias", ""),
            "sample_count": sample_count,
            "dataset": dataset.metadata,
            "class_names": dataset.class_names,
            "target_layer": args.target_layer,
            "target_head": args.target_head,
            "target_layer_fraction": args.target_layer_fraction,
            "projection": args.projection,
            "metric": args.metric,
            "flow_start_token_index": args.flow_start_token_index,
            "drop_special_tokens": args.drop_special_tokens,
            "max_token_length": args.max_token_length,
            "token_truncation_side": args.token_truncation_side,
            "max_stored_tokens": args.max_stored_tokens,
            "stored_token_selection": args.stored_token_selection,
            "token_q_storage_dtype": args.token_q_storage_dtype,
            "activation_space": args.activation_space,
            "activation_space_label": activation_space_label(args.activation_space),
            "color_flow_by": args.color_flow_by,
            "plot_3d": args.plot_3d,
            "plot_sample_limit": args.plot_sample_limit,
            "plot_3d_elev": args.plot_3d_elev,
            "plot_3d_azim": args.plot_3d_azim,
            "detail_layer_heads": args.detail_layer_heads,
            "label_permutation_n": args.label_permutation_n,
            "top_layer_head_null_rank_limit": args.top_layer_head_null_rank_limit,
            "label_shuffle_seed": args.label_shuffle_seed if args.label_shuffle_seed is not None else args.random_state,
            "high_d_flow_metrics": args.high_d_flow_metrics,
            "projection_diagnostics": args.projection_diagnostics,
            "projection_knn_k": args.projection_knn_k,
            "probe_linear": args.probe_linear,
            "linear_probe_alpha": args.linear_probe_alpha,
            "linear_probe_permutation_n": effective_linear_probe_permutation_n(args),
            "head_similarity": args.head_similarity,
            "head_similarity_layers": args.head_similarity_layers,
            "q_capture_stage": args.q_capture_stage,
            "q_capture_label": q_capture_label(args),
            "q_capture_position_note": args.q_capture_position_note,
            **bundle.model_info,
        },
    )

    plt = None if args.no_plots else load_pyplot(args.show)
    if plt is None:
        head_scores = [
            {
                "layer": args.target_layer,
                "head": head_idx,
                "silhouette_cosine": safe_silhouette(np, final_q_all[:, args.target_layer, head_idx, :], labels),
            }
            for head_idx in range(num_heads)
        ]
        head_scores.sort(key=score_sort_key, reverse=True)
    else:
        head_scores = plot_head_manifolds(np, plt, final_q_all, labels, dataset.class_names, output_dir, args)
    head_scores = normalize_rows(head_scores)
    write_csv_rows(output_dir / "head_scores.csv", head_scores)

    layer_head_scores = layer_head_score_rows(np, final_q_all, labels)
    write_csv_rows(output_dir / "layer_head_scores.csv", layer_head_scores)
    if plt is not None:
        plot_layer_head_heatmap(np, plt, layer_head_scores, n_layers, num_heads, output_dir, args)

    valid_head_scores = [row for row in head_scores if not math.isnan(float(row["silhouette_cosine"]))]
    if valid_head_scores:
        best_head = int(valid_head_scores[0]["head"])
    else:
        print("Warning: all head silhouette scores are NaN. Falling back to target_head.", file=sys.stderr)
        best_head = args.target_head
    valid_layer_head_scores = [
        row for row in layer_head_scores if not math.isnan(float(row["silhouette_cosine"]))
    ]
    best_layer_head = valid_layer_head_scores[0] if valid_layer_head_scores else None
    detail_layer_heads = []
    add_unique_pair(
        detail_layer_heads,
        layer=args.target_layer,
        head=args.target_head,
        reason="target_layer_head",
    )
    if best_head != args.target_head and not args.skip_best_head:
        add_unique_pair(
            detail_layer_heads,
            layer=args.target_layer,
            head=best_head,
            reason="best_head_at_target_layer",
        )
    if args.detail_best_layer_head and best_layer_head is not None:
        add_unique_pair(
            detail_layer_heads,
            layer=int(best_layer_head["layer"]),
            head=int(best_layer_head["head"]),
            reason="best_layer_head_overall",
        )
    for layer_idx, head_idx in parse_layer_head_specs(args.detail_layer_heads):
        if not (0 <= layer_idx < n_layers):
            raise SystemExit(f"--detail-layer-heads layer must be in [0, {n_layers - 1}], got {layer_idx}")
        if not (0 <= head_idx < num_heads):
            raise SystemExit(f"--detail-layer-heads head must be in [0, {num_heads - 1}], got {head_idx}")
        add_unique_pair(
            detail_layer_heads,
            layer=layer_idx,
            head=head_idx,
            reason="explicit_detail_layer_heads",
        )

    head_similarity_summaries: dict[str, Any] = {}
    if args.head_similarity:
        similarity_layers = resolve_head_similarity_layers(
            args,
            n_layers=n_layers,
            detail_layer_heads=detail_layer_heads,
            best_layer_head=best_layer_head,
        )
        for layer_idx in similarity_layers:
            head_similarity_summaries[str(layer_idx)] = write_head_similarity_outputs(
                np,
                plt,
                final_q_all,
                output_dir,
                args,
                layer_idx=layer_idx,
            )

    permutation_rows = label_permutation_rows(
        np,
        final_q_all,
        labels,
        detail_layer_heads,
        n_permutations=args.label_permutation_n,
        seed=args.label_shuffle_seed if args.label_shuffle_seed is not None else args.random_state,
    )
    if permutation_rows:
        write_csv_rows(output_dir / "label_permutation_summary.csv", permutation_rows)

    top_layer_head_permutation_summary = top_layer_head_permutation_rows(
        np,
        final_q_all,
        labels,
        layer_head_scores,
        rank_limit=args.top_layer_head_null_rank_limit,
        n_permutations=args.label_permutation_n,
        seed=(args.label_shuffle_seed if args.label_shuffle_seed is not None else args.random_state) + 503,
    )
    if top_layer_head_permutation_summary:
        write_csv_rows(
            output_dir / "top_layer_head_label_permutation_summary.csv",
            top_layer_head_permutation_summary,
        )

    linear_probe_summary = (
        linear_probe_rows(
            np,
            final_q_all,
            labels,
            detail_layer_heads,
            alpha=args.linear_probe_alpha,
            n_permutations=effective_linear_probe_permutation_n(args),
            seed=(args.label_shuffle_seed if args.label_shuffle_seed is not None else args.random_state) + 1009,
        )
        if args.probe_linear
        else []
    )
    if linear_probe_summary:
        write_csv_rows(output_dir / "linear_probe_summary.csv", linear_probe_summary)

    flow_summaries: dict[str, Any] = {}
    highd_flow_summaries: dict[str, Any] = {}
    projection_diagnostic_rows: list[dict[str, Any]] = []
    written_layer_curve_heads = set()
    for pair in detail_layer_heads:
        layer_idx = int(pair["layer"])
        head_idx = int(pair["head"])
        layer_rows = layer_separation_rows(np, final_q_all, labels, head_idx)
        if head_idx not in written_layer_curve_heads:
            write_csv_rows(output_dir / f"layer_separation_head_{head_idx}.csv", layer_rows)
            written_layer_curve_heads.add(head_idx)
        if args.projection_diagnostics:
            x_final = final_q_all[:, layer_idx, head_idx, :]
            final_emb = compute_2d(
                np,
                x_final,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                metric=args.metric,
                random_state=args.random_state,
                projection=args.projection,
            )
            projection_diagnostic_rows.append(
                projection_diagnostic_row(
                    np,
                    x_final,
                    final_emb,
                    labels,
                    dataset.class_names,
                    args,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    reason=str(pair.get("reason", "")),
                )
            )
        all_emb, paths, highd_paths, token_record_paths, meta_rows = project_token_qs(
            np,
            bundle.token_q_records,
            bundle.token_records,
            labels,
            dataset.class_names,
            dataset.texts,
            args,
            layer_idx=layer_idx,
            head_idx=head_idx,
        )
        flow_rows = build_flow_rows(np, paths, labels, dataset.class_names, dataset.texts)
        write_csv_rows(output_dir / f"token_flow_metrics_layer_{layer_idx}_head_{head_idx}.csv", flow_rows)
        write_csv_rows(output_dir / f"token_flow_meta_layer_{layer_idx}_head_{head_idx}.csv", meta_rows)
        if args.high_d_flow_metrics:
            highd_flow_rows = build_highd_flow_rows(np, highd_paths, labels, dataset.class_names, dataset.texts)
            write_csv_rows(output_dir / f"highd_token_flow_metrics_layer_{layer_idx}_head_{head_idx}.csv", highd_flow_rows)
            highd_flow_summaries[f"{layer_idx}:{head_idx}"] = highd_flow_summary(np, highd_flow_rows)

        field = compute_query_flow_field(np, all_emb, paths, args)
        flow_summaries[f"{layer_idx}:{head_idx}"] = field["summary"]

        if plt is not None:
            plot_layer_separation_curve(
                plt,
                layer_rows,
                output_dir,
                args,
                head_idx=head_idx,
                focus_layer_idx=layer_idx,
            )
            plot_layer_trajectory(
                np,
                plt,
                final_q_all,
                labels,
                dataset.class_names,
                output_dir,
                args,
                head_idx=head_idx,
                focus_layer_idx=layer_idx,
            )
            if args.plot_3d:
                plot_layer_trajectory_3d(
                    np,
                    plt,
                    final_q_all,
                    labels,
                    dataset.class_names,
                    output_dir,
                    args,
                    head_idx=head_idx,
                    focus_layer_idx=layer_idx,
                )
            plot_query_flow(
                np,
                plt,
                all_emb,
                paths,
                token_record_paths,
                labels,
                dataset.class_names,
                dataset.texts,
                output_dir,
                args,
                layer_idx=layer_idx,
                head_idx=head_idx,
                text_index=None,
            )
            paths_3d = None
            if args.plot_3d:
                _, paths_3d = project_highd_paths(np, highd_paths, args, n_components=3)
                plot_query_flow_3d(
                    np,
                    plt,
                    np.concatenate(paths_3d, axis=0) if paths_3d else np.zeros((0, 3), dtype="float32"),
                    paths_3d,
                    token_record_paths,
                    labels,
                    dataset.class_names,
                    dataset.texts,
                    output_dir,
                    args,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    text_index=None,
                )
            if args.detail_text_index is not None:
                plot_query_flow(
                    np,
                    plt,
                    all_emb,
                    paths,
                    token_record_paths,
                    labels,
                    dataset.class_names,
                    dataset.texts,
                    output_dir,
                    args,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    text_index=args.detail_text_index,
                )
                if args.plot_3d:
                    if paths_3d is None:
                        _, paths_3d = project_highd_paths(np, highd_paths, args, n_components=3)
                    plot_query_flow_3d(
                        np,
                        plt,
                        np.concatenate(paths_3d, axis=0) if paths_3d else np.zeros((0, 3), dtype="float32"),
                        paths_3d,
                        token_record_paths,
                        labels,
                        dataset.class_names,
                        dataset.texts,
                        output_dir,
                        args,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        text_index=args.detail_text_index,
                    )
            plot_query_flow_field(
                np,
                plt,
                field,
                output_dir,
                args,
                layer_idx=layer_idx,
                head_idx=head_idx,
            )

    if projection_diagnostic_rows:
        write_csv_rows(output_dir / "projection_diagnostics.csv", projection_diagnostic_rows)

    summary = {
        "output_dir": portable_output_path(output_dir),
        "model_alias": getattr(args, "model_alias", ""),
        "model_path": args.model_path,
        "backend": args.backend,
        "dataset": dataset.metadata,
        "sample_count": sample_count,
        "n_layers": n_layers,
        "num_heads": num_heads,
        "query_heads": bundle.model_info.get("query_heads", num_heads),
        "head_dim": head_dim,
        "projection_path": bundle.model_info.get("projection_path", ""),
        "projection_kind": bundle.model_info.get("projection_kind", ""),
        "rope_path": bundle.model_info.get("rope_path", ""),
        "pool_last_k": args.pool_last_k,
        "activation_space": args.activation_space,
        "activation_space_label": activation_space_label(args.activation_space),
        "q_capture_stage": args.q_capture_stage,
        "activation_capture_stage": activation_capture_stage(args.q_capture_stage, args.activation_space),
        "q_capture_label": q_capture_label(args),
        "q_capture_position_note": args.q_capture_position_note,
        "best_head_at_target_layer": best_head,
        "best_layer_head_by_silhouette": best_layer_head,
        "best_layer_head_relative_depth": (
            float(best_layer_head["layer"]) / max(1, n_layers - 1)
            if best_layer_head is not None
            else None
        ),
        "target_layer": args.target_layer,
        "target_layer_relative_depth": float(args.target_layer) / max(1, n_layers - 1),
        "target_head": args.target_head,
        "flow_start_token_index": args.flow_start_token_index,
        "drop_special_tokens": args.drop_special_tokens,
        "max_token_length": args.max_token_length,
        "token_truncation_side": args.token_truncation_side,
        "max_stored_tokens": args.max_stored_tokens,
        "stored_token_selection": args.stored_token_selection,
        "token_q_storage_dtype": args.token_q_storage_dtype,
        "color_flow_by": args.color_flow_by,
        "label_permutation_n": args.label_permutation_n,
        "top_layer_head_null_rank_limit": args.top_layer_head_null_rank_limit,
        "label_shuffle_seed": args.label_shuffle_seed if args.label_shuffle_seed is not None else args.random_state,
        "top_head_scores": head_scores[: min(5, len(head_scores))],
        "top_layer_head_scores": layer_head_scores[: min(10, len(layer_head_scores))],
        "detailed_layer_heads": detail_layer_heads,
        "label_permutation_summary": permutation_rows,
        "top_layer_head_label_permutation_summary": top_layer_head_permutation_summary,
        "linear_probe_summary": linear_probe_summary,
        "projection_diagnostics": projection_diagnostic_rows,
        "head_similarity_summaries": head_similarity_summaries,
        "flow_field_summaries": flow_summaries,
        "highd_flow_summaries": highd_flow_summaries,
    }
    write_json(output_dir / "analysis_summary.json", summary)

    print("=== Q-space manifold monolith complete ===")
    print(f"output_dir: {output_dir}")
    print(f"final_q_all shape: {tuple(final_q_all.shape)}")
    print(f"best_head_at_layer_{args.target_layer}: {best_head}")
    if best_layer_head is not None:
        print(
            "best_layer_head_by_silhouette: "
            f"layer={best_layer_head['layer']} head={best_layer_head['head']} "
            f"score={best_layer_head['silhouette_cosine']:.4f}"
        )
    print("top head scores:")
    for row in summary["top_head_scores"]:
        print(f"  head={row['head']} silhouette_cosine={row['silhouette_cosine']:.4f}")
    return summary


def collect_bundle(args: argparse.Namespace, dataset: TextDataset) -> CaptureBundle:
    if args.backend == "torch":
        return collect_with_torch(args, dataset)
    return collect_with_mlx(args, dataset)


def run_single_analysis(args: argparse.Namespace, dataset: TextDataset) -> dict[str, Any]:
    bundle = collect_bundle(args, dataset)
    return analyze_bundle(args, dataset, bundle)


def batch_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_layer_head_by_silhouette") or {}
    target_top = (summary.get("top_head_scores") or [{}])[0]
    null_lookup = permutation_rows_by_layer_head(summary.get("top_layer_head_label_permutation_summary") or [])
    best_null = None
    if best.get("layer") != "" and best.get("head") != "":
        best_null = null_lookup.get((int(best.get("layer")), int(best.get("head"))))
    row = {
        "model_alias": summary.get("model_alias", ""),
        "backend": summary.get("backend", ""),
        "model_path": summary.get("model_path", ""),
        "dataset_source": (summary.get("dataset") or {}).get("dataset_source", ""),
        "sample_count": summary.get("sample_count", ""),
        "n_layers": summary.get("n_layers", ""),
        "num_heads": summary.get("num_heads", ""),
        "query_heads": summary.get("query_heads", ""),
        "head_dim": summary.get("head_dim", ""),
        "pool_last_k": summary.get("pool_last_k", ""),
        "activation_space": summary.get("activation_space", "q"),
        "activation_space_label": summary.get("activation_space_label", ""),
        "q_capture_stage": summary.get("q_capture_stage", ""),
        "activation_capture_stage": summary.get("activation_capture_stage", ""),
        "q_capture_label": summary.get("q_capture_label", ""),
        "target_layer": summary.get("target_layer", ""),
        "target_layer_relative_depth": summary.get("target_layer_relative_depth", ""),
        "target_head": summary.get("target_head", ""),
        "best_head_at_target_layer": summary.get("best_head_at_target_layer", ""),
        "best_head_at_target_layer_score": target_top.get("silhouette_cosine", ""),
        "best_layer": best.get("layer", ""),
        "best_head": best.get("head", ""),
        "best_layer_relative_depth": summary.get("best_layer_head_relative_depth", ""),
        "best_layer_head_silhouette_cosine": best.get("silhouette_cosine", ""),
        "output_dir": portable_output_path(summary.get("output_dir", "")),
    }
    row.update(silhouette_null_columns(best_null))
    return row


def batch_top_layer_head_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    null_lookup = permutation_rows_by_layer_head(summary.get("top_layer_head_label_permutation_summary") or [])
    for rank, row in enumerate(summary.get("top_layer_head_scores") or [], start=1):
        out_row = {
            "model_alias": summary.get("model_alias", ""),
            "backend": summary.get("backend", ""),
            "model_path": summary.get("model_path", ""),
            "dataset_source": (summary.get("dataset") or {}).get("dataset_source", ""),
            "activation_space": summary.get("activation_space", "q"),
            "q_capture_stage": summary.get("q_capture_stage", ""),
            "activation_capture_stage": summary.get("activation_capture_stage", ""),
            "rank": rank,
            "layer": row.get("layer", ""),
            "head": row.get("head", ""),
            "relative_depth": float(row["layer"]) / max(1, int(summary.get("n_layers", 1)) - 1)
            if row.get("layer") != ""
            else "",
            "silhouette_cosine": row.get("silhouette_cosine", ""),
        }
        null_row = None
        if row.get("layer") != "" and row.get("head") != "":
            null_row = null_lookup.get((int(row.get("layer")), int(row.get("head"))))
        out_row.update(silhouette_null_columns(null_row))
        rows.append(out_row)
    return rows


def run_batch(args: argparse.Namespace, dataset: TextDataset, specs: Sequence[ModelRunSpec]) -> list[dict[str, Any]]:
    root_dir = args.output_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    summary_rows = []
    top_rows = []
    for spec in specs:
        run_args = args_for_model_spec(args, spec, root_dir)
        print(f"=== batch model: {spec.alias} ({spec.backend}) {spec.model_path} ===")
        summary = run_single_analysis(run_args, dataset)
        summaries.append(summary)
        summary_rows.append(batch_summary_row(summary))
        top_rows.extend(batch_top_layer_head_rows(summary))
    write_csv_rows(root_dir / "batch_model_summary.csv", summary_rows)
    write_csv_rows(root_dir / "batch_top_layer_heads.csv", top_rows)
    write_json(
        root_dir / "batch_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset.metadata,
            "model_count": len(specs),
            "models": [
                {
                    "alias": spec.alias,
                    "backend": spec.backend,
                    "model_path": spec.model_path,
                    "activation_space": spec.activation_space or getattr(args, "activation_space", "q"),
                }
                for spec in specs
            ],
            "summary_rows": summary_rows,
        },
    )
    print("=== Q-space batch compare complete ===")
    print(f"output_dir: {root_dir}")
    print("best layer/head by model:")
    for row in summary_rows:
        print(
            f"  {row['model_alias']}: layer={row['best_layer']} "
            f"head={row['best_head']} rel_depth={row['best_layer_relative_depth']} "
            f"score={row['best_layer_head_silhouette_cosine']}"
        )
    return summary_rows


def parse_positive_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise SystemExit(f"invalid integer in list: {item!r}") from exc
        if value < 1:
            raise SystemExit(f"integer list values must be >= 1, got {value}")
        values.append(value)
    return sorted(set(values))


def recompute_final_q_all_for_pool(np: Any, token_q_records: Sequence[Any], pool_last_k: int) -> Any:
    final_q_records = []
    for q_by_layer in token_q_records:
        q_by_layer_np = np.asarray(q_by_layer)
        k = min(pool_last_k, q_by_layer_np.shape[2])
        final_q_records.append(q_by_layer_np[:, :, -k:, :].mean(axis=2))
    return np.stack(final_q_records, axis=0)


def bundle_with_pool_last_k(np: Any, bundle: CaptureBundle, pool_last_k: int) -> CaptureBundle:
    model_info = dict(bundle.model_info)
    model_info["pool_last_k"] = pool_last_k
    return CaptureBundle(
        final_q_all=recompute_final_q_all_for_pool(np, bundle.token_q_records, pool_last_k),
        token_q_records=bundle.token_q_records,
        token_records=bundle.token_records,
        model_info=model_info,
    )


def run_pool_last_k_sweep(
    args: argparse.Namespace,
    dataset: TextDataset,
    specs: Sequence[ModelRunSpec],
    pool_values: Sequence[int],
) -> None:
    np = load_numpy()
    if args.max_stored_tokens > 0:
        if args.stored_token_selection != "tail":
            raise SystemExit("--pool-last-k-sweep with --max-stored-tokens requires --stored-token-selection tail")
        if args.max_stored_tokens < max(pool_values):
            raise SystemExit(
                "--pool-last-k-sweep requires --max-stored-tokens to be at least the largest pool value "
                f"({max(pool_values)}), got {args.max_stored_tokens}"
            )
    root_dir = args.output_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    run_specs = list(specs)
    if not run_specs:
        run_specs = [
            ModelRunSpec(
                alias=slugify(getattr(args, "model_alias", "") or args.model_path.split("/")[-1]),
                backend=args.backend,
                model_path=args.model_path,
                target_layer=args.target_layer,
                target_head=args.target_head,
                target_layer_fraction=args.target_layer_fraction,
                detail_layer_heads=args.detail_layer_heads,
            )
        ]

    for spec in run_specs:
        cleanup_runtime_caches(spec.backend)
        base_args = args_for_model_spec(args, spec, root_dir)
        missing_pool_values = []
        for pool_last_k in pool_values:
            if specs:
                output_dir = root_dir / f"pool_last_k_{pool_last_k}" / spec.alias
            else:
                output_dir = root_dir / f"pool_last_k_{pool_last_k}"
            summary_path = output_dir / "analysis_summary.json"
            if getattr(args, "resume_existing", False) and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row = batch_summary_row(summary)
                row["pool_last_k"] = pool_last_k
                all_rows.append(row)
                write_csv_rows(root_dir / "pool_last_k_sweep_summary.csv", all_rows)
                print(f"=== pool_last_k sweep resume: skipping existing {output_dir} ===")
                continue
            missing_pool_values.append(pool_last_k)
        if not missing_pool_values:
            continue

        capture_args = argparse.Namespace(**vars(base_args))
        capture_args.pool_last_k = max(missing_pool_values)
        print(f"=== pool_last_k sweep capture: {spec.alias} ({spec.backend}) {spec.model_path} ===")
        bundle = collect_bundle(capture_args, dataset)
        try:
            for pool_last_k in missing_pool_values:
                run_args = argparse.Namespace(**vars(base_args))
                run_args.pool_last_k = pool_last_k
                if specs:
                    run_args.output_dir = root_dir / f"pool_last_k_{pool_last_k}" / spec.alias
                else:
                    run_args.output_dir = root_dir / f"pool_last_k_{pool_last_k}"
                pooled_bundle = bundle_with_pool_last_k(np, bundle, pool_last_k)
                try:
                    summary = analyze_bundle(run_args, dataset, pooled_bundle)
                finally:
                    del pooled_bundle
                    cleanup_runtime_caches(spec.backend)
                row = batch_summary_row(summary)
                row["pool_last_k"] = pool_last_k
                all_rows.append(row)
                write_csv_rows(root_dir / "pool_last_k_sweep_summary.csv", all_rows)
                del summary, run_args, row
                cleanup_runtime_caches(spec.backend)
        finally:
            del bundle
            cleanup_runtime_caches(spec.backend)

    write_csv_rows(root_dir / "pool_last_k_sweep_summary.csv", all_rows)
    write_json(
        root_dir / "pool_last_k_sweep_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pool_last_k_values": list(pool_values),
            "dataset": dataset.metadata,
            "summary_rows": all_rows,
        },
    )
    print("=== pool_last_k sweep complete ===")
    print(f"output_dir: {root_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    parser.add_argument("--model-path", default="gpt2", help="Hugging Face repo id or local model path.")
    parser.add_argument(
        "--batch-models",
        default="",
        help=(
            "Comma-separated model specs for cross-model comparison. "
            "Use alias=backend:model-path, e.g. mistral_it=mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit."
        ),
    )
    parser.add_argument(
        "--model-list-json",
        type=Path,
        default=None,
        help="JSON list of model specs with alias/backend/model_path and optional target overrides.",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["default", "json", "sst2", "subj", "hf"],
        default="default",
        help="Dataset source. sst2 uses GLUE/SST-2; subj defaults to SetFit/subj.",
    )
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--hf-dataset-name", default="")
    parser.add_argument("--hf-dataset-config", default=None)
    parser.add_argument("--text-column", default="")
    parser.add_argument("--label-column", default="")
    parser.add_argument(
        "--text-template",
        default="",
        help=(
            "Optional format template applied to each sample after loading, e.g. "
            "'Review: {text}\\nSentiment:'. Available fields: {text}, {label}, {class_name}, {index}."
        ),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Balanced sample count per class. 0 keeps all rows after any max-samples limit.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum dataset rows after optional class balancing. 0 keeps all rows.",
    )
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/q_space_manifold_monolith"))
    parser.add_argument("--target-layer", type=int, default=6)
    parser.add_argument(
        "--target-layer-fraction",
        type=float,
        default=None,
        help="Override --target-layer after model load using round(fraction * (n_layers - 1)).",
    )
    parser.add_argument("--target-head", type=int, default=3)
    parser.add_argument("--pool-last-k", type=int, default=1)
    parser.add_argument(
        "--activation-space",
        default="q",
        help=(
            "Which attention projection space to analyze: q/query, k/key, or v/value. "
            "K/V capture currently uses pre-RoPE projection outputs."
        ),
    )
    parser.add_argument(
        "--q-capture-stage",
        default="pre-rope",
        help=(
            "Which capture stage to analyze. Use pre-rope for projection outputs, or "
            "post-rope for RoPE-applied Q before attention scoring. post-rope currently requires "
            "--backend mlx and --activation-space q."
        ),
    )
    parser.add_argument(
        "--pool-last-k-sweep",
        default="",
        help="Comma-separated pool_last_k values, e.g. 1,3,5. Reuses captured token Q per model.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="For sweep runs, skip pool/model output directories that already have analysis_summary.json.",
    )
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--projection", choices=["umap", "pca"], default="umap")
    parser.add_argument("--field-grid-size", type=int, default=9)
    parser.add_argument(
        "--high-d-flow-metrics",
        action="store_true",
        help="Also compute token-flow metrics in the original high-dimensional Q space.",
    )
    parser.add_argument(
        "--projection-diagnostics",
        action="store_true",
        help="Write projection_diagnostics.csv comparing high-D geometry with the 2D projection.",
    )
    parser.add_argument(
        "--projection-knn-k",
        type=int,
        default=5,
        help="k for high-D to projected kNN recall in projection diagnostics.",
    )
    parser.add_argument(
        "--probe-linear",
        action="store_true",
        help="Run a leave-one-out ridge linear probe on final-token Q vectors for detailed heads.",
    )
    parser.add_argument(
        "--linear-probe-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for --probe-linear.",
    )
    parser.add_argument(
        "--linear-probe-permutation-n",
        type=int,
        default=None,
        help=(
            "Random-label null permutations for --probe-linear. "
            "Defaults to --label-permutation-n; set 0 for large plotting runs."
        ),
    )
    parser.add_argument(
        "--head-similarity",
        action="store_true",
        help="Compute within-layer head RSA and linear CKA matrices for selected layers.",
    )
    parser.add_argument(
        "--head-similarity-layers",
        default="detail",
        help="Layers for --head-similarity: detail, target, best, all, or comma-separated indices.",
    )
    parser.add_argument("--detail-text-index", type=int, default=0)
    parser.add_argument(
        "--detail-layer-heads",
        default="",
        help="Comma-separated extra layer/head probes such as 10:21,11:4,L12:H10.",
    )
    parser.add_argument(
        "--label-permutation-n",
        type=int,
        default=0,
        help="Run N random-label silhouette controls for detailed layer/head probes.",
    )
    parser.add_argument(
        "--top-layer-head-null-rank-limit",
        type=int,
        default=5,
        help=(
            "When --label-permutation-n is positive, also compute silhouette null statistics "
            "for the top K layer/head rows and merge them into batch summaries. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--label-shuffle-seed",
        type=int,
        default=None,
        help="Seed for random-label controls; defaults to --random-state.",
    )
    parser.add_argument(
        "--color-flow-by",
        choices=["class", "token_idx", "flow_position"],
        default="class",
        help="Color token Q-flow plots by class or token position.",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Also write 3D layer-trajectory and token Q-flow plots for detailed layer/head probes.",
    )
    parser.add_argument(
        "--plot-sample-limit",
        type=int,
        default=0,
        help="Limit all-sample trajectory/flow plots to N sampled texts. 0 plots all samples; metrics still use all samples.",
    )
    parser.add_argument(
        "--plot-3d-elev",
        type=float,
        default=22.0,
        help="Matplotlib elevation angle for --plot-3d outputs.",
    )
    parser.add_argument(
        "--plot-3d-azim",
        type=float,
        default=-58.0,
        help="Matplotlib azimuth angle for --plot-3d outputs.",
    )
    parser.add_argument(
        "--flow-start-token-index",
        type=int,
        default=0,
        help="Drop earlier token positions from token Q-flow plots/metrics only.",
    )
    parser.add_argument(
        "--drop-special-tokens",
        action="store_true",
        help="Drop common tokenizer special tokens such as <s>, </s>, and <|endoftext|> from token Q-flow.",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=0,
        help="Truncate model inputs to at most N tokens before Q capture. 0 keeps the tokenizer output unchanged.",
    )
    parser.add_argument(
        "--token-truncation-side",
        choices=["head", "tail"],
        default="head",
        help="When --max-token-length is set, keep the head or tail side of the tokenized input.",
    )
    parser.add_argument(
        "--max-stored-tokens",
        type=int,
        default=0,
        help=(
            "Store at most N token positions per sample for token-flow analysis and pool-last-k sweeps. "
            "0 stores all captured token Qs. Use with --stored-token-selection tail for memory-heavy code runs."
        ),
    )
    parser.add_argument(
        "--stored-token-selection",
        choices=["head", "tail"],
        default="tail",
        help="Which token positions to keep when --max-stored-tokens is positive.",
    )
    parser.add_argument(
        "--token-q-storage-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Storage dtype for retained token activation records. final_q_all remains float32.",
    )
    parser.add_argument(
        "--detail-best-layer-head",
        action="store_true",
        help="Also run detailed flow/projection/probe outputs for the best layer/head over the full atlas.",
    )
    parser.add_argument("--skip-best-head", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--show-individual-trajectories",
        dest="show_individual_trajectories",
        action="store_true",
    )
    parser.add_argument(
        "--hide-individual-trajectories",
        dest="show_individual_trajectories",
        action="store_false",
    )
    parser.add_argument("--device", default="auto", help="torch only: auto, cpu, cuda, mps, ...")
    parser.add_argument("--torch-dtype", default="auto", help="torch only: auto, float16, bfloat16, float32, ...")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(show_individual_trajectories=True)
    args = parser.parse_args()
    if args.pool_last_k < 1:
        parser.error("--pool-last-k must be >= 1")
    try:
        args.q_capture_stage = normalize_q_capture_stage(args.q_capture_stage)
    except SystemExit as exc:
        parser.error(str(exc))
    try:
        args.activation_space = normalize_activation_space(args.activation_space)
    except SystemExit as exc:
        parser.error(str(exc))
    if args.q_capture_stage == Q_STAGE_POST_ROPE and args.activation_space != "q":
        parser.error("--q-capture-stage post-rope currently supports --activation-space q only")
    if args.pool_last_k_sweep:
        try:
            parse_positive_int_list(args.pool_last_k_sweep)
        except SystemExit as exc:
            parser.error(str(exc))
    if args.samples_per_class < 0:
        parser.error("--samples-per-class must be >= 0")
    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0")
    if args.target_layer_fraction is not None and not (0.0 <= args.target_layer_fraction <= 1.0):
        parser.error("--target-layer-fraction must be in [0, 1]")
    if args.field_grid_size < 2:
        parser.error("--field-grid-size must be >= 2")
    if args.projection_knn_k < 1:
        parser.error("--projection-knn-k must be >= 1")
    if args.linear_probe_alpha <= 0:
        parser.error("--linear-probe-alpha must be > 0")
    if args.linear_probe_permutation_n is not None and args.linear_probe_permutation_n < 0:
        parser.error("--linear-probe-permutation-n must be >= 0")
    if args.label_permutation_n < 0:
        parser.error("--label-permutation-n must be >= 0")
    if args.top_layer_head_null_rank_limit < 0:
        parser.error("--top-layer-head-null-rank-limit must be >= 0")
    if args.detail_text_index is not None and args.detail_text_index < 0:
        parser.error("--detail-text-index must be >= 0")
    if args.plot_sample_limit < 0:
        parser.error("--plot-sample-limit must be >= 0")
    if args.flow_start_token_index < 0:
        parser.error("--flow-start-token-index must be >= 0")
    if args.max_token_length < 0:
        parser.error("--max-token-length must be >= 0")
    if args.max_stored_tokens < 0:
        parser.error("--max-stored-tokens must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    dataset = load_dataset_from_args(args)
    dataset = apply_text_template(dataset, args.text_template)
    if args.detail_text_index is not None and args.detail_text_index >= len(dataset.texts):
        raise SystemExit(f"--detail-text-index must be in [0, {len(dataset.texts) - 1}]")
    specs = load_model_run_specs(args)
    pool_values = parse_positive_int_list(args.pool_last_k_sweep) if args.pool_last_k_sweep else []
    if pool_values:
        run_pool_last_k_sweep(args, dataset, specs, pool_values)
    elif specs:
        run_batch(args, dataset, specs)
    else:
        run_single_analysis(args, dataset)


if __name__ == "__main__":
    main()
