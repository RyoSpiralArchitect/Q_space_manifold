#!/usr/bin/env python3
"""Audit compact-cluster geometry versus distributed linear readout in Q-space.

This script consumes an existing ``q_space_vectors.npz`` artifact emitted by
``q_space_manifold_monolith.py``. It intentionally avoids changing the main
capture pipeline: the goal is a lightweight second-pass diagnostic for cases
where cosine silhouette is modest but linear probe accuracy is high.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_run_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).expanduser()
        vectors_path = Path(args.vectors).expanduser() if args.vectors else run_dir / "q_space_vectors.npz"
        summary_path = (
            Path(args.analysis_summary).expanduser()
            if args.analysis_summary
            else run_dir / "analysis_summary.json"
        )
        return vectors_path, summary_path
    if args.vectors is None:
        raise SystemExit("Provide --run-dir or --vectors.")
    vectors_path = Path(args.vectors).expanduser()
    summary_path = Path(args.analysis_summary).expanduser() if args.analysis_summary else None
    return vectors_path, summary_path


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(int(part))
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return result


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def class_name_map(labels: np.ndarray, class_names: np.ndarray | None) -> dict[int, str]:
    classes = sorted(int(c) for c in np.unique(labels))
    mapping: dict[int, str] = {}
    for c in classes:
        if class_names is not None and 0 <= c < len(class_names):
            mapping[c] = str(class_names[c])
        else:
            mapping[c] = f"class_{c}"
    return mapping


def class_counts(labels: np.ndarray, names: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for c in sorted(names):
        rows.append({"label": c, "class_name": names[c], "count": int(np.sum(labels == c))})
    return rows


def stratified_sample_indices(labels: np.ndarray, max_n: int, random_state: int) -> np.ndarray:
    n = len(labels)
    if max_n <= 0 or n <= max_n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(labels, return_counts=True)
    allocation = np.floor(max_n * counts / n).astype(int)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)

    while allocation.sum() > max_n:
        candidates = np.where(allocation > 1)[0]
        if len(candidates) == 0:
            break
        largest = candidates[np.argmax(allocation[candidates])]
        allocation[largest] -= 1

    while allocation.sum() < max_n:
        candidates = np.where(allocation < counts)[0]
        if len(candidates) == 0:
            break
        ratios = allocation[candidates] / counts[candidates]
        smallest = candidates[np.argmin(ratios)]
        allocation[smallest] += 1

    sampled: list[np.ndarray] = []
    for c, k in zip(classes, allocation):
        idx = np.where(labels == c)[0]
        sampled.append(rng.choice(idx, size=int(k), replace=False))
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out


def centroid_geometry(
    x: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    xu = l2_normalize(x)
    classes = np.array(sorted(names), dtype=int)
    centroids = []
    class_rows = []
    for c in classes:
        mask = labels == c
        x_c = xu[mask]
        centroid = x_c.mean(axis=0, keepdims=True)
        centroid = l2_normalize(centroid)[0]
        centroids.append(centroid)
        distances = 1.0 - x_c @ centroid
        class_rows.append(
            {
                "label": int(c),
                "class_name": names[int(c)],
                "count": int(mask.sum()),
                "within_centroid_cosine_distance_mean": float(np.mean(distances)),
                "within_centroid_cosine_distance_std": float(np.std(distances)),
                "within_centroid_cosine_distance_median": float(np.median(distances)),
            }
        )

    centroid_matrix = np.stack(centroids, axis=0)
    scores = xu @ centroid_matrix.T
    pred = classes[np.argmax(scores, axis=1)]
    nearest_acc = accuracy_score(labels, pred)
    nearest_bal_acc = balanced_accuracy_score(labels, pred)

    for row in class_rows:
        c = int(row["label"])
        mask = labels == c
        row["nearest_centroid_accuracy"] = float(accuracy_score(labels[mask], pred[mask]))

    pair_rows = []
    pair_distances = []
    for i, c_i in enumerate(classes):
        for j, c_j in enumerate(classes):
            if j <= i:
                continue
            dist = 1.0 - float(np.dot(centroid_matrix[i], centroid_matrix[j]))
            pair_distances.append(dist)
            pair_rows.append(
                {
                    "label_a": int(c_i),
                    "class_name_a": names[int(c_i)],
                    "label_b": int(c_j),
                    "class_name_b": names[int(c_j)],
                    "centroid_cosine_distance": dist,
                }
            )

    within_means = [float(row["within_centroid_cosine_distance_mean"]) for row in class_rows]
    mean_within = float(np.mean(within_means))
    mean_between = float(np.mean(pair_distances)) if pair_distances else math.nan
    summary = {
        "nearest_centroid_accuracy": float(nearest_acc),
        "nearest_centroid_balanced_accuracy": float(nearest_bal_acc),
        "mean_within_centroid_cosine_distance": mean_within,
        "mean_between_centroid_cosine_distance": mean_between,
        "between_within_distance_ratio": float(mean_between / (mean_within + 1e-12)),
    }
    return summary, class_rows, pair_rows


def prefixed(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in row.items()}


def knn_accuracy_rows(
    x: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    k_values: list[int],
    sample_size: int,
    random_state: int,
) -> list[dict[str, Any]]:
    idx = stratified_sample_indices(labels, sample_size, random_state)
    xs = l2_normalize(x[idx])
    ys = labels[idx]
    sim = xs @ xs.T
    np.fill_diagonal(sim, -np.inf)
    max_k = min(max(k_values), len(idx) - 1)
    order = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    order_sorted = np.take_along_axis(
        order,
        np.argsort(-np.take_along_axis(sim, order, axis=1), axis=1),
        axis=1,
    )
    classes = np.array(sorted(names), dtype=int)
    class_to_pos = {int(c): pos for pos, c in enumerate(classes)}
    y_pos = np.array([class_to_pos[int(y)] for y in ys], dtype=int)

    rows = []
    for k in k_values:
        k = min(k, max_k)
        preds = []
        for neighbors in order_sorted[:, :k]:
            counts = np.bincount(y_pos[neighbors], minlength=len(classes))
            preds.append(classes[int(np.argmax(counts))])
        pred_arr = np.asarray(preds, dtype=int)
        rows.append(
            {
                "k": int(k),
                "sample_size": int(len(idx)),
                "accuracy": float(accuracy_score(ys, pred_arr)),
                "balanced_accuracy": float(balanced_accuracy_score(ys, pred_arr)),
            }
        )
    return rows


def cv_probe_scores(
    estimator: Any,
    x: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    _, counts = np.unique(labels, return_counts=True)
    actual_splits = min(n_splits, int(counts.min()))
    if actual_splits < 2:
        return {
            "cv_splits": actual_splits,
            "cv_accuracy_mean": math.nan,
            "cv_accuracy_std": math.nan,
            "cv_balanced_accuracy_mean": math.nan,
            "cv_balanced_accuracy_std": math.nan,
        }
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    accs = []
    bals = []
    for train_idx, test_idx in skf.split(x, labels):
        model = clone(estimator)
        model.fit(x[train_idx], labels[train_idx])
        pred = model.predict(x[test_idx])
        accs.append(accuracy_score(labels[test_idx], pred))
        bals.append(balanced_accuracy_score(labels[test_idx], pred))
    return {
        "cv_splits": int(actual_splits),
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_balanced_accuracy_mean": float(np.mean(bals)),
        "cv_balanced_accuracy_std": float(np.std(bals)),
    }


def pca_probe_curve(
    x: np.ndarray,
    labels: np.ndarray,
    dims: list[int],
    alpha: float,
    cv_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    max_dim = min(x.shape[1], x.shape[0] - len(np.unique(labels)))
    dims = sorted({d for d in dims if 1 <= d <= max_dim})
    if not dims:
        return []

    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=max(dims), random_state=random_state)
    pca.fit(x_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    rows = []
    for dim in dims:
        estimator = make_pipeline(
            StandardScaler(),
            PCA(n_components=dim, random_state=random_state),
            RidgeClassifier(alpha=alpha),
        )
        scores = cv_probe_scores(estimator, x, labels, cv_splits, random_state)
        rows.append(
            {
                "pca_dim": int(dim),
                "explained_variance_ratio_cumulative": float(cumulative[dim - 1]),
                **scores,
            }
        )
    return rows


def one_vs_rest_margin_rows(
    x: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    alpha: float,
) -> list[dict[str, Any]]:
    classes = np.array(sorted(names), dtype=int)
    x_scaled = StandardScaler().fit_transform(x)
    model = OneVsRestClassifier(RidgeClassifier(alpha=alpha))
    model.fit(x_scaled, labels)
    margins = model.decision_function(x_scaled)
    if margins.ndim == 1:
        margins = margins[:, None]

    rows = []
    for col, c in enumerate(classes):
        binary = labels == c
        scores = margins[:, col]
        auc = math.nan
        if len(np.unique(binary)) == 2:
            auc = float(roc_auc_score(binary.astype(int), scores))
        pos_scores = scores[binary]
        neg_scores = scores[~binary]
        rows.append(
            {
                "label": int(c),
                "class_name": names[int(c)],
                "positive_count": int(binary.sum()),
                "negative_count": int((~binary).sum()),
                "train_fit_ovr_auc": auc,
                "positive_margin_mean": float(np.mean(pos_scores)),
                "negative_margin_mean": float(np.mean(neg_scores)),
                "margin_gap": float(np.mean(pos_scores) - np.mean(neg_scores)),
                "positive_margin_median": float(np.median(pos_scores)),
                "negative_margin_median": float(np.median(neg_scores)),
            }
        )
    return rows


def sampled_silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    idx = stratified_sample_indices(labels, sample_size, random_state)
    if len(np.unique(labels[idx])) < 2 or len(idx) <= len(np.unique(labels[idx])):
        value = math.nan
    else:
        value = float(silhouette_score(x[idx], labels[idx], metric="cosine"))
    return {
        "silhouette_cosine_sampled": value,
        "silhouette_sample_size": int(len(idx)),
    }


def select_layer_head(args: argparse.Namespace, summary: dict[str, Any]) -> tuple[int, int]:
    layer = args.layer
    head = args.head
    best = summary.get("best_layer_head_by_silhouette") or {}
    if layer is None:
        layer = best.get("layer")
    if head is None:
        head = best.get("head")
    if layer is None or head is None:
        raise SystemExit("Provide --layer/--head or an analysis_summary.json with best_layer_head_by_silhouette.")
    return int(layer), int(head)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Existing monolith run directory.")
    parser.add_argument("--vectors", type=Path, help="Path to q_space_vectors.npz.")
    parser.add_argument("--analysis-summary", type=Path, help="Path to analysis_summary.json.")
    parser.add_argument("--layer", type=int, help="Layer index. Defaults to summary best layer.")
    parser.add_argument("--head", type=int, help="Head index. Defaults to summary best head.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for audit CSV/JSON outputs.")
    parser.add_argument("--pca-dims", type=parse_int_list, default=parse_int_list("1,2,4,8,16,32,64,128"))
    parser.add_argument("--knn-k", type=parse_int_list, default=parse_int_list("1,3,5,10"))
    parser.add_argument("--knn-sample-size", type=int, default=2000)
    parser.add_argument("--silhouette-sample-size", type=int, default=2000)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--linear-probe-alpha", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    vectors_path, summary_path = infer_run_paths(args)
    summary = load_summary(summary_path)
    layer, head = select_layer_head(args, summary)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vectors: {vectors_path}")
    with np.load(vectors_path, allow_pickle=True) as data:
        final_q_all = data["final_q_all"]
        labels = np.asarray(data["labels"], dtype=int)
        class_names = np.asarray(data["class_names"], dtype=object) if "class_names" in data.files else None
        if final_q_all.ndim != 4:
            raise SystemExit(f"Expected final_q_all [sample, layer, head, dim], got {final_q_all.shape}")
        if not (0 <= layer < final_q_all.shape[1]):
            raise SystemExit(f"--layer must be in [0, {final_q_all.shape[1] - 1}], got {layer}")
        if not (0 <= head < final_q_all.shape[2]):
            raise SystemExit(f"--head must be in [0, {final_q_all.shape[2] - 1}], got {head}")
        x = np.asarray(final_q_all[:, layer, head, :], dtype=np.float32)

    names = class_name_map(labels, class_names)
    print(f"Auditing layer={layer} head={head} X={x.shape} classes={len(names)}")

    centroid_summary, class_rows, centroid_pair_rows = centroid_geometry(x, labels, names)
    x_centered = x - x.mean(axis=0, keepdims=True)
    centered_centroid_summary, _, _ = centroid_geometry(x_centered, labels, names)
    knn_rows = knn_accuracy_rows(
        x,
        labels,
        names,
        args.knn_k,
        args.knn_sample_size,
        args.random_state,
    )
    sil_summary = sampled_silhouette(x, labels, args.silhouette_sample_size, args.random_state)
    centered_sil_summary = sampled_silhouette(
        x_centered,
        labels,
        args.silhouette_sample_size,
        args.random_state,
    )
    raw_probe = cv_probe_scores(
        make_pipeline(StandardScaler(), RidgeClassifier(alpha=args.linear_probe_alpha)),
        x,
        labels,
        args.cv_splits,
        args.random_state,
    )
    pca_rows = pca_probe_curve(
        x,
        labels,
        args.pca_dims,
        args.linear_probe_alpha,
        args.cv_splits,
        args.random_state,
    )
    margin_rows = one_vs_rest_margin_rows(x, labels, names, args.linear_probe_alpha)

    best_pca_row = max(pca_rows, key=lambda row: row["cv_accuracy_mean"]) if pca_rows else {}
    summary_row = {
        "vectors_path": str(vectors_path),
        "analysis_summary_path": str(summary_path) if summary_path else "",
        "layer": layer,
        "head": head,
        "sample_count": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "class_count": int(len(names)),
        "source_best_silhouette_cosine": (summary.get("best_layer_head_by_silhouette") or {}).get(
            "silhouette_cosine", ""
        ),
        **sil_summary,
        **prefixed(centered_sil_summary, "centered_"),
        **centroid_summary,
        **prefixed(centered_centroid_summary, "centered_"),
        "global_mean_norm": float(np.linalg.norm(x.mean(axis=0))),
        "mean_sample_norm": float(np.mean(np.linalg.norm(x, axis=1))),
        "knn_k1_accuracy": next((row["accuracy"] for row in knn_rows if row["k"] == 1), ""),
        "knn_k5_accuracy": next((row["accuracy"] for row in knn_rows if row["k"] == 5), ""),
        "raw_linear_probe_cv_accuracy": raw_probe["cv_accuracy_mean"],
        "raw_linear_probe_cv_balanced_accuracy": raw_probe["cv_balanced_accuracy_mean"],
        "raw_linear_probe_cv_splits": raw_probe["cv_splits"],
        "best_pca_dim": best_pca_row.get("pca_dim", ""),
        "best_pca_cv_accuracy": best_pca_row.get("cv_accuracy_mean", ""),
        "best_pca_cv_balanced_accuracy": best_pca_row.get("cv_balanced_accuracy_mean", ""),
        "best_pca_explained_variance_ratio_cumulative": best_pca_row.get(
            "explained_variance_ratio_cumulative", ""
        ),
    }

    write_csv_rows(output_dir / "geometry_summary.csv", [summary_row])
    write_csv_rows(output_dir / "class_scatter.csv", class_rows)
    write_csv_rows(output_dir / "centroid_pair_distances.csv", centroid_pair_rows)
    write_csv_rows(output_dir / "knn_accuracy.csv", knn_rows)
    write_csv_rows(output_dir / "pca_probe_curve.csv", pca_rows)
    write_csv_rows(output_dir / "one_vs_rest_margins.csv", margin_rows)
    write_csv_rows(output_dir / "class_counts.csv", class_counts(labels, names))
    write_json(
        output_dir / "geometry_audit_manifest.json",
        {
            "vectors_path": vectors_path,
            "analysis_summary_path": summary_path,
            "layer": layer,
            "head": head,
            "args": vars(args),
            "summary": summary_row,
            "class_counts": class_counts(labels, names),
            "outputs": [
                "geometry_summary.csv",
                "class_scatter.csv",
                "centroid_pair_distances.csv",
                "knn_accuracy.csv",
                "pca_probe_curve.csv",
                "one_vs_rest_margins.csv",
                "class_counts.csv",
            ],
        },
    )

    print("=== Q-space geometry audit complete ===")
    print(f"output_dir: {output_dir}")
    print(
        "summary: "
        f"sil_sample={summary_row['silhouette_cosine_sampled']:.4f} "
        f"nearest_centroid={summary_row['nearest_centroid_accuracy']:.4f} "
        f"raw_probe_cv={summary_row['raw_linear_probe_cv_accuracy']:.4f} "
        f"best_pca_dim={summary_row['best_pca_dim']} "
        f"best_pca_cv={summary_row['best_pca_cv_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
