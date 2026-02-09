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

METRICS = [
    ("attention_entropy", "test/attention_entropy_table"),
    ("attention_neff", "test/attention_neff_table"),
    ("attention_top1_mass", "test/attention_top1_mass_table"),
    ("attention_top2_mass", "test/attention_top2_mass_table"),
    ("attention_top4_mass", "test/attention_top4_mass_table"),
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
    test_lens = list(test_lens)
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
            if arr is None:
                continue
            series.append((test_len, float(np.nanmax(arr))))
        if series:
            data[algo_label] = series

    if not data:
        return

    def _sort_key(x: str):
        return (len(x), x)

    plt.figure(figsize=(6 + 1.2 * max(1, len(data)), 4.5))
    for algo_label, series in sorted(data.items()):
        series = sorted(series, key=lambda x: _sort_key(x[0]))
        xs = [s[0] for s in series]
        ys = [s[1] for s in series]
        plt.plot(xs, ys, marker="o", label=algo_label)
    plt.title("Max top-1 attention vs test length")
    plt.xlabel("test length")
    plt.ylabel("max top-1 attention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _write_top1_vs_score_plot(
    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    score_by_algo_len: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    points = []
    for algo_label, metric_rows in by_algo.items():
        top1_by_len = metric_rows.get("attention_top1_mass", {})
        scores = score_by_algo_len.get(algo_label, {})
        for test_len, arr in top1_by_len.items():
            if arr is None:
                continue
            if test_len not in scores:
                continue
            max_top1 = float(np.nanmax(arr))
            score = float(scores[test_len])
            points.append((algo_label, test_len, max_top1, score))

    if not points:
        return

    plt.figure(figsize=(6.5, 4.5))
    for algo_label in sorted({p[0] for p in points}):
        xs = [p[2] for p in points if p[0] == algo_label]
        ys = [p[3] for p in points if p[0] == algo_label]
        plt.scatter(xs, ys, label=algo_label, alpha=0.8)
    plt.title("Max top-1 attention vs score")
    plt.xlabel("max top-1 attention")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=FILTERS))
    logging.info("Fetched %d runs from %s/%s", len(runs), ENTITY, PROJECT)

    by_algo: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    score_by_algo_len: Dict[str, Dict[str, float]] = {}
    global_metric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    metric_cmaps = {
        "attention_entropy": "Purples",
        "attention_neff": "Oranges",
        "attention_top1_mass": "Reds",
        "attention_top2_mass": "Blues",
        "attention_top4_mass": "Greens",
    }

    for run in tqdm(runs, desc="Processing runs"):
        if not _matches_filters(run):
            logging.info("Skipping run %s due to filters", run.id)
            continue
        algo, test_len = _algo_and_test_len(run)
        algo_label = _sanitize(algo)
        len_label = _sanitize(test_len)
        score = run.summary.get("test/score")

        for metric, table_key in METRICS:
            arr = None
            ref = _get_ref(run, table_key)
            if ref:
                table = _load_json_file(run, ref)
                if table:
                    arr = _table_to_array(table)

            if arr is None:
                logging.warning("No data found for run %s, metric %s", run.id, metric)
                continue

            by_algo.setdefault(algo_label, {}).setdefault(metric, {})[len_label] = arr
            if score is not None:
                score_by_algo_len.setdefault(algo_label, {})[len_label] = float(score)

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
        test_lens = sorted(test_lens, key=lambda x: (len(x), x))
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


if __name__ == "__main__":
    main()
