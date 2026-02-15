#!/usr/bin/env python3
"""
Fetch attention sharpness metrics from W&B and write per-run heatmaps.

Filters:
  - config.test_lengths != "max_train_length"
  - summary.test/attention_top1_mass != null

Outputs: png heatmaps in ./wandb_attention_heatmaps/
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm


ENTITY = "akshitsinha3"
PROJECT = "GDL_CLRS30"
OUT_DIR = "wandb_attention_heatmaps"
FILTERS = None
CACHE_FILENAME = "metrics_cache.json"
CACHE_VERSION = 2

METRICS = [
    ("attention_entropy", "test/attention_entropy_table"),
    ("attention_neff", "test/attention_neff_table"),
    ("attention_top1_mass", "test/attention_top1_mass_table"),
    ("attention_top2_mass", "test/attention_top2_mass_table"),
    ("attention_top4_mass", "test/attention_top4_mass_table"),
    ("output_entropy", None),
]


def _sanitize(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s or "unknown"


def _get_ref(run: wandb.apis.public.Run, key: str) -> Optional[Dict[str, Any]]:
    if key in run.summary:
        ref = run.summary[key]
        if isinstance(ref, dict) and "path" in ref:
            return ref
    # Try history as a fallback
    for row in run.history(keys=[key], pandas=False):
        if key in row and isinstance(row[key], dict):
            ref = row[key]
            if "path" in ref:
                return ref
    return None


def _load_json_file(run: wandb.apis.public.Run, ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = ref.get("path")
    if not path:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        file = run.file(path)
        local_path = file.download(root=tmpdir, replace=True).name
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)


def _table_to_array(table: Dict[str, Any]) -> Optional[np.ndarray]:
    columns = table.get("columns")
    data = table.get("data")
    if not columns or not data:
        return None
    try:
        layer_idx = columns.index("layer")
        head_idx = columns.index("head")
        value_idx = columns.index("value")
    except ValueError:
        return None
    max_layer = max(int(row[layer_idx]) for row in data)
    max_head = max(int(row[head_idx]) for row in data)
    arr = np.full((max_layer + 1, max_head + 1), np.nan, dtype=np.float32)
    for row in data:
        l = int(row[layer_idx])
        h = int(row[head_idx])
        v = float(row[value_idx])
        arr[l, h] = v
    return arr


def _load_image_array(run: wandb.apis.public.Run, key: str) -> Optional[np.ndarray]:
    if key not in run.summary:
        return None
    ref = run.summary[key]
    if not isinstance(ref, dict) or "path" not in ref:
        return None
    path = ref["path"]
    with tempfile.TemporaryDirectory() as tmpdir:
        file = run.file(path)
        local_path = file.download(root=tmpdir, replace=True).name
        img = plt.imread(local_path)
        return img


def _summary_scalar_to_array(
    run: wandb.apis.public.Run,
    key: str,
) -> Optional[np.ndarray]:
    value = run.summary.get(key)
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return np.array([[v]], dtype=np.float32)


def _algo_and_test_len(run: wandb.apis.public.Run) -> Tuple[str, str]:
    algo = "unknown"
    test_len = "unknown"

    name = run.name or ""
    if name:
        parts = [p for p in name.split("-") if p]
        if parts:
            algo = parts[0]
            for p in parts:
                if p.isdigit():
                    test_len = p
                    break

    return algo, test_len


def _length_at_least(test_len: str, min_len: int = 8) -> bool:
    try:
        return float(test_len) >= min_len
    except ValueError:
        return False


def _algo_group(algo_label: str) -> Optional[str]:
    selection_pointer_heavy = {
        "articulation_points",
        "binary_search",
        "bridges",
        "dfs",
        "dijkstra",
        "floyd_warshall",
        "mst_kruskal",
        "mst_prim",
        "topological_sort",
    }
    aggregation_tolerant = {
        "bfs",
        "find_maximum_subarray_kadane",
        "lcs_length",
        "minimum",
    }
    if algo_label in selection_pointer_heavy:
        return "Selection / pointer-heavy"
    if algo_label in aggregation_tolerant:
        return "Aggregation-tolerant"
    return None


def _matches_filters(run: wandb.apis.public.Run) -> bool:
    cfg = run.config or {}
    test_lengths = cfg.get("test_lengths")
    if isinstance(test_lengths, (list, tuple)):
        if "max_train_length" in test_lengths:
            return False
    elif isinstance(test_lengths, str):
        if test_lengths == "max_train_length":
            return False

    if run.summary.get("test/attention_top1_mass") is None:
        return False
    return True


def _run_updated_token(run: wandb.apis.public.Run) -> str:
    for attr in ("updated_at", "created_at"):
        if hasattr(run, attr):
            value = getattr(run, attr)
            if value:
                return str(value)
    summary = run.summary or {}
    for key in ("_timestamp", "_runtime", "_step"):
        if key in summary and summary[key] is not None:
            return str(summary[key])
    return ""


def _load_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"version": CACHE_VERSION, "runs": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "runs": {}}
    if not isinstance(data, dict):
        return {"version": CACHE_VERSION, "runs": {}}
    if data.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "runs": {}}
    runs = data.get("runs")
    if not isinstance(runs, dict):
        data["runs"] = {}
    return data


def _save_cache(path: str, cache: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _cache_array(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    data = entry.get("data")
    if data is None:
        return None
    try:
        arr = np.array(data, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    return arr


def _write_heatmap(ax, arr: np.ndarray, title: str, cmap: str,
                   vmin: Optional[float], vmax: Optional[float]) -> None:
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _write_superplot(
    algo: str,
    metric_rows: Dict[str, Dict[str, np.ndarray]],
    test_lens: Iterable[str],
    out_path: str,
    metric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    metric_cmaps: Dict[str, str],
) -> None:
    test_lens = [t for t in test_lens if _length_at_least(t)]
    if not test_lens:
        return
    nrows = len(METRICS)
    ncols = max(1, len(test_lens))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3.2 * nrows),
        squeeze=False,
    )
    for r, (metric, _) in enumerate(METRICS):
        for c, test_len in enumerate(test_lens):
            ax = axes[r][c]
            arr = metric_rows.get(metric, {}).get(test_len)
            if arr is None:
                ax.axis("off")
                ax.set_title(f"{metric} | test_len={test_len}\n(no data)")
                continue
            vmin, vmax = metric_ranges.get(metric, (None, None))
            cmap = metric_cmaps.get(metric, "Blues")
            _write_heatmap(ax, arr, f"{metric} | test_len={test_len}", cmap, vmin, vmax)
    fig.suptitle(f"{algo}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_top1_max_plot(
    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
) -> None:
    # Build mapping algo -> list of (test_len, max_top1)
    data = {}
    for algo_label, metric_rows in by_algo.items():
        top1_by_len = metric_rows.get("attention_top1_mass", {})
        series = []
        for test_len, arr in top1_by_len.items():
            if not _length_at_least(test_len):
                continue
            if arr is None:
                continue
            series.append((test_len, float(np.nanmax(arr))))
        if series:
            data[algo_label] = series

    if not data:
        return

    def _sort_key(x: str):
        return (len(x), x)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for algo_label, series in sorted(data.items()):
        series = sorted(series, key=lambda x: _sort_key(x[0]))
        xs = [s[0] for s in series]
        ys = [s[1] for s in series]
        ax.plot(xs, ys, marker="o", label=algo_label)
    ax.set_title("Max top-1 attention vs test length")
    ax.set_xlabel("test length")
    ax.set_ylabel("max top-1 attention")
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(data), 6),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_top1_vs_score_plot(
    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    score_by_algo_len: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    points = []
    for algo_label, metric_rows in by_algo.items():
        top1_by_len = metric_rows.get("attention_top1_mass", {})
        scores = score_by_algo_len.get(algo_label, {})
        group = _algo_group(algo_label)
        if group is None:
            continue
        for test_len, arr in top1_by_len.items():
            if not _length_at_least(test_len):
                continue
            if arr is None:
                continue
            if test_len not in scores:
                continue
            max_top1 = float(np.nanmax(arr))
            score = float(scores[test_len])
            points.append((algo_label, group, test_len, max_top1, score))

    if not points:
        return

    def _len_to_size(tlen: str) -> float:
        try:
            val = float(tlen)
            return 30.0 + 6.0 * val
        except ValueError:
            return 40.0

    def _len_sort_key(tlen: str):
        try:
            return (0, float(tlen))
        except ValueError:
            return (1, tlen)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    group_order = ["Selection / pointer-heavy", "Aggregation-tolerant"]
    markers = ["o", "s"]

    for ax, group_name in zip(axes, group_order):
        group_points = [p for p in points if p[1] == group_name]
        if not group_points:
            ax.axis("off")
            ax.set_title(f"{group_name}\n(no data)")
            continue

        algo_labels = sorted({p[0] for p in group_points})
        for idx, algo_label in enumerate(algo_labels):
            algo_points = [p for p in group_points if p[0] == algo_label]
            algo_points.sort(key=lambda p: _len_sort_key(p[2]))
            xs = [p[3] for p in algo_points]
            ys = [p[4] for p in algo_points]
            sizes = [_len_to_size(p[2]) for p in algo_points]
            ax.scatter(
                xs,
                ys,
                s=sizes,
                marker=markers[idx % len(markers)],
                label=algo_label,
                alpha=0.8,
            )
        ax.set_title(group_name)
        ax.set_xlabel("max top-1 attention")

    axes[0].set_ylabel("score")
    fig.suptitle("Max top-1 attention vs score")

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(labels), 6),
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.18, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_min_entropy_vs_score_plot(
    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    score_by_algo_len: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    points = []
    for algo_label, metric_rows in by_algo.items():
        ent_by_len = metric_rows.get("attention_entropy", {})
        scores = score_by_algo_len.get(algo_label, {})
        for test_len, arr in ent_by_len.items():
            if not _length_at_least(test_len):
                continue
            if arr is None:
                continue
            if test_len not in scores:
                continue
            min_entropy = float(np.nanmin(arr))
            score = float(scores[test_len])
            points.append((algo_label, test_len, min_entropy, score))

    if not points:
        return

    def _len_to_size(tlen: str) -> float:
        try:
            val = float(tlen)
            return 30.0 + 25.0 * np.log1p(val)
        except ValueError:
            return 40.0

    plt.figure(figsize=(6.5, 4.5))
    for algo_label in sorted({p[0] for p in points}):
        algo_points = [p for p in points if p[0] == algo_label]
        algo_points.sort(key=lambda p: _len_to_size(p[1]))
        xs = [p[2] for p in algo_points]
        ys = [p[3] for p in algo_points]
        sizes = [_len_to_size(p[1]) for p in algo_points]
        plt.plot(xs, ys, linewidth=1.0, alpha=0.6)
        plt.scatter(xs, ys, s=sizes, label=algo_label, alpha=0.7)
    plt.title("Min attention entropy vs score")
    plt.xlabel("min attention entropy")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _write_neff_vs_score_by_len_plot(
    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    score_by_algo_len: Dict[str, Dict[str, float]],
    out_path: str,
    lengths: Tuple[str, str] = ("16", "32"),
) -> None:
    points_by_len = {lengths[0]: [], lengths[1]: []}
    for algo_label, metric_rows in by_algo.items():
        neff_by_len = metric_rows.get("attention_neff", {})
        scores = score_by_algo_len.get(algo_label, {})
        for tlen in lengths:
            if not _length_at_least(tlen):
                continue
            if tlen not in neff_by_len or tlen not in scores:
                continue
            arr = neff_by_len[tlen]
            if arr is None:
                continue
            neff_mean = float(np.nanmean(arr))
            score = float(scores[tlen])
            points_by_len[tlen].append((neff_mean, score, algo_label))

    if not points_by_len[lengths[0]] and not points_by_len[lengths[1]]:
        return

    algos = sorted({p[2] for pts in points_by_len.values() for p in pts})
    cmap = plt.get_cmap("tab20")
    algo_colors = {a: cmap(i % cmap.N) for i, a in enumerate(algos)}

    plt.figure(figsize=(6.5, 4.5))
    markers = {lengths[0]: "o", lengths[1]: "s"}
    for tlen, pts in points_by_len.items():
        if not pts:
            continue
        for neff_mean, score, algo_label in pts:
            plt.scatter(
                neff_mean,
                score,
                marker=markers.get(tlen, "o"),
                color=algo_colors.get(algo_label),
                alpha=0.8,
            )

    # Legend: algos (colors)
    algo_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=algo_colors[a], markersize=7, label=a)
        for a in algos
    ]
    length_handles = [
        plt.Line2D([0], [0], marker=markers[lengths[0]], color="k",
                   linestyle="None", label=f"L={lengths[0]}"),
        plt.Line2D([0], [0], marker=markers[lengths[1]], color="k",
                   linestyle="None", label=f"L={lengths[1]}"),
    ]
    first_legend = plt.legend(handles=algo_handles, title="Algo", loc="best", fontsize=8)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=length_handles, title="Length", loc="upper right", fontsize=8)
    plt.title("N_eff (e^H) vs score (L=16,32)")
    plt.xlabel("N_eff (mean)")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _write_entropy_score_vs_length_plot(
    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    score_by_algo_len: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    entropy_data: Dict[str, list[tuple[str, float]]] = {}
    score_data: Dict[str, list[tuple[str, float]]] = {}

    for algo_label, metric_rows in by_algo.items():
        ent_by_len = metric_rows.get("attention_entropy", {})
        series = []
        for test_len, arr in ent_by_len.items():
            if not _length_at_least(test_len):
                continue
            if arr is None:
                continue
            series.append((test_len, float(np.nanmean(arr))))
        if series:
            entropy_data[algo_label] = series

    for algo_label, scores in score_by_algo_len.items():
        series = [
            (test_len, float(score))
            for test_len, score in scores.items()
            if _length_at_least(test_len)
        ]
        if series:
            score_data[algo_label] = series

    if not entropy_data and not score_data:
        return

    def _sort_key(x: str):
        try:
            return (0, float(x))
        except ValueError:
            return (1, x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if entropy_data:
        for algo_label, series in sorted(entropy_data.items()):
            series = sorted(series, key=lambda x: _sort_key(x[0]))
            xs = [s[0] for s in series]
            ys = [s[1] for s in series]
            axes[0].plot(xs, ys, marker="o", label=algo_label)
        axes[0].set_title("Mean attention entropy vs test length")
        axes[0].set_xlabel("test length")
        axes[0].set_ylabel("mean attention entropy")
    else:
        axes[0].axis("off")
        axes[0].set_title("Mean attention entropy vs test length\n(no data)")

    if score_data:
        for algo_label, series in sorted(score_data.items()):
            series = sorted(series, key=lambda x: _sort_key(x[0]))
            xs = [s[0] for s in series]
            ys = [s[1] for s in series]
            axes[1].plot(xs, ys, marker="o", label=algo_label)
        axes[1].set_title("Score vs test length")
        axes[1].set_xlabel("test length")
        axes[1].set_ylabel("score")
    else:
        axes[1].axis("off")
        axes[1].set_title("Score vs test length\n(no data)")

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 6),
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.14, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_output_entropy_score_vs_length_plot(
    output_entropy_by_algo_len: Dict[str, Dict[str, float]],
    output_entropy_norm_by_algo_len: Dict[str, Dict[str, float]],
    score_by_algo_len: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    if not output_entropy_by_algo_len and not output_entropy_norm_by_algo_len:
        return

    def _sort_key(x: str):
        try:
            return (0, float(x))
        except ValueError:
            return (1, x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)

    if output_entropy_by_algo_len:
        for algo_label, series_map in sorted(output_entropy_by_algo_len.items()):
            series = [
                (test_len, float(v))
                for test_len, v in series_map.items()
                if _length_at_least(test_len)
            ]
            if not series:
                continue
            series = sorted(series, key=lambda x: _sort_key(x[0]))
            xs = [s[0] for s in series]
            ys = [s[1] for s in series]
            axes[0].plot(xs, ys, marker="o", label=algo_label)
        axes[0].set_title("Output entropy vs test length")
        axes[0].set_xlabel("test length")
        axes[0].set_ylabel("output entropy")
    else:
        axes[0].axis("off")
        axes[0].set_title("Output entropy vs test length\n(no data)")

    if output_entropy_norm_by_algo_len:
        for algo_label, series_map in sorted(output_entropy_norm_by_algo_len.items()):
            series = [
                (test_len, float(v))
                for test_len, v in series_map.items()
                if _length_at_least(test_len)
            ]
            if not series:
                continue
            series = sorted(series, key=lambda x: _sort_key(x[0]))
            xs = [s[0] for s in series]
            ys = [s[1] for s in series]
            axes[1].plot(xs, ys, marker="o", label=algo_label)
        axes[1].set_title("Output entropy (normalized) vs test length")
        axes[1].set_xlabel("test length")
        axes[1].set_ylabel("normalized output entropy")
    else:
        axes[1].axis("off")
        axes[1].set_title("Output entropy (normalized) vs test length\n(no data)")

    if score_by_algo_len:
        for algo_label, series_map in sorted(score_by_algo_len.items()):
            series = [
                (test_len, float(v))
                for test_len, v in series_map.items()
                if _length_at_least(test_len)
            ]
            if not series:
                continue
            series = sorted(series, key=lambda x: _sort_key(x[0]))
            xs = [s[0] for s in series]
            ys = [s[1] for s in series]
            axes[2].plot(xs, ys, marker="o", label=algo_label)
        axes[2].set_title("Score vs test length")
        axes[2].set_xlabel("test length")
        axes[2].set_ylabel("score")
    else:
        axes[2].axis("off")
        axes[2].set_title("Score vs test length\n(no data)")

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 6),
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.14, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUT_DIR, CACHE_FILENAME)
    cache = _load_cache(cache_path)
    cache_runs: Dict[str, Any] = cache.get("runs", {})
    cache["runs"] = cache_runs
    cache_hits = 0
    cache_misses = 0
    cache_stale = 0
    cache_dirty = False
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=FILTERS))
    logging.info("Fetched %d runs from %s/%s", len(runs), ENTITY, PROJECT)

    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    score_by_algo_len: Dict[str, Dict[str, float]] = {}
    output_entropy_by_algo_len: Dict[str, Dict[str, float]] = {}
    output_entropy_norm_by_algo_len: Dict[str, Dict[str, float]] = {}
    global_metric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    metric_cmaps = {
        "attention_entropy": "Purples",
        "attention_neff": "Oranges",
        "attention_top1_mass": "Reds",
        "attention_top2_mass": "Blues",
        "attention_top4_mass": "Greens",
        "output_entropy": "Greys",
    }

    for run in tqdm(runs, desc="Processing runs"):
        run_token = _run_updated_token(run)
        if not _matches_filters(run):
            logging.info("Skipping run %s due to filters", run.id)
            cached = cache_runs.get(run.id)
            if not cached or cached.get("updated_token") != run_token:
                cache_runs[run.id] = {
                    "updated_token": run_token,
                    "skipped": True,
                }
                cache_dirty = True
            continue

        cached = cache_runs.get(run.id)
        if cached and cached.get("updated_token") == run_token and not cached.get("skipped"):
            algo_label = cached.get("algo_label", "unknown")
            len_label = cached.get("len_label", "unknown")
            score = cached.get("score")
            output_entropy = cached.get("output_entropy")
            output_entropy_norm = cached.get("output_entropy_normalized")
            if score is not None:
                score_by_algo_len.setdefault(algo_label, {})[len_label] = float(score)
            if output_entropy is not None:
                output_entropy_by_algo_len.setdefault(algo_label, {})[len_label] = float(
                    output_entropy)
            if output_entropy_norm is not None:
                output_entropy_norm_by_algo_len.setdefault(algo_label, {})[len_label] = float(
                    output_entropy_norm)
            metrics_cache = cached.get("metrics", {})
            for metric, _ in METRICS:
                metric_entry = metrics_cache.get(metric)
                if not metric_entry:
                    continue
                arr = _cache_array(metric_entry)
                if arr is None:
                    continue
                by_algo.setdefault(algo_label, {}).setdefault(metric, {})[len_label] = arr
            cache_hits += 1
            continue

        if cached and cached.get("updated_token") != run_token:
            cache_stale += 1
        else:
            cache_misses += 1

        algo, test_len = _algo_and_test_len(run)
        algo_label = _sanitize(algo)
        len_label = _sanitize(test_len)
        score = run.summary.get("test/score")
        output_entropy = run.summary.get("test/output_entropy")
        output_entropy_norm = run.summary.get("test/output_entropy_normalized")
        if score is not None:
            score_by_algo_len.setdefault(algo_label, {})[len_label] = float(score)
        if output_entropy is not None:
            output_entropy_by_algo_len.setdefault(algo_label, {})[len_label] = float(
                output_entropy)
        if output_entropy_norm is not None:
            output_entropy_norm_by_algo_len.setdefault(algo_label, {})[len_label] = float(
                output_entropy_norm)
        metrics_cache: Dict[str, Any] = {}

        for metric, table_key in METRICS:
            arr = None
            if table_key is not None:
                ref = _get_ref(run, table_key)
                if ref:
                    table = _load_json_file(run, ref)
                    if table:
                        arr = _table_to_array(table)
            else:
                if metric == "output_entropy":
                    arr = _summary_scalar_to_array(run, "test/output_entropy")

            if arr is None:
                logging.warning("No data found for run %s, metric %s", run.id, metric)
                continue

            by_algo.setdefault(algo_label, {}).setdefault(metric, {})[len_label] = arr
            metrics_cache[metric] = {"data": arr.tolist()}

        cache_runs[run.id] = {
            "updated_token": run_token,
            "skipped": False,
            "algo_label": algo_label,
            "len_label": len_label,
            "score": float(score) if score is not None else None,
            "output_entropy": (
                float(output_entropy) if output_entropy is not None else None
            ),
            "output_entropy_normalized": (
                float(output_entropy_norm) if output_entropy_norm is not None else None
            ),
            "metrics": metrics_cache,
        }
        cache_dirty = True

    for metric, _ in METRICS:
        values = []
        for algo_label in by_algo:
            metric_dict = by_algo[algo_label].get(metric, {})
            for arr in metric_dict.values():
                if arr is not None:
                    values.append(arr)
        if values:
            stacked = np.stack(values, axis=0)
            vmin = float(np.nanmin(stacked))
            vmax = float(np.nanmax(stacked))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin == vmax:
                vmin = vmin - 1e-6
                vmax = vmax + 1e-6
            global_metric_ranges[metric] = (vmin, vmax)
        else:
            global_metric_ranges[metric] = (None, None)

    for algo_label, metric_rows in by_algo.items():
        test_lens = set()
        for metric, _ in METRICS:
            test_lens.update(metric_rows.get(metric, {}).keys())
        test_lens = sorted(
            [t for t in test_lens if _length_at_least(t)],
            key=lambda x: (len(x), x),
        )
        if not test_lens:
            continue
        out_name = f"{algo_label}__superplot.png"
        out_path = os.path.join(OUT_DIR, out_name)
        _write_superplot(
            algo_label,
            metric_rows,
            test_lens,
            out_path,
            global_metric_ranges,
            metric_cmaps,
        )
        logging.info("Wrote %s", out_path)

    top1_out = os.path.join(OUT_DIR, "top1_max_across_test_sizes.png")
    _write_top1_max_plot(by_algo, top1_out)
    logging.info("Wrote %s", top1_out)

    score_out = os.path.join(OUT_DIR, "top1_max_vs_score.png")
    _write_top1_vs_score_plot(by_algo, score_by_algo_len, score_out)
    logging.info("Wrote %s", score_out)

    min_entropy_out = os.path.join(OUT_DIR, "min_entropy_vs_score.png")
    _write_min_entropy_vs_score_plot(by_algo, score_by_algo_len, min_entropy_out)
    logging.info("Wrote %s", min_entropy_out)

    neff_out = os.path.join(OUT_DIR, "neff_vs_score_L16_L32.png")
    _write_neff_vs_score_by_len_plot(by_algo, score_by_algo_len, neff_out)
    logging.info("Wrote %s", neff_out)

    entropy_score_out = os.path.join(OUT_DIR, "entropy_and_score_vs_length.png")
    _write_entropy_score_vs_length_plot(by_algo, score_by_algo_len, entropy_score_out)
    logging.info("Wrote %s", entropy_score_out)

    output_entropy_score_out = os.path.join(
        OUT_DIR, "output_entropy_and_score_vs_length.png"
    )
    _write_output_entropy_score_vs_length_plot(
        output_entropy_by_algo_len,
        output_entropy_norm_by_algo_len,
        score_by_algo_len,
        output_entropy_score_out,
    )
    logging.info("Wrote %s", output_entropy_score_out)

    logging.info(
        "Cache stats: hits=%d, misses=%d, stale=%d",
        cache_hits,
        cache_misses,
        cache_stale,
    )
    if cache_dirty:
        _save_cache(cache_path, cache)
        logging.info("Saved cache to %s", cache_path)


if __name__ == "__main__":
    main()
