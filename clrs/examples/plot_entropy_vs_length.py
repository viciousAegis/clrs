# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Plot attention entropy vs length from edge_attention_outputs-len-XX folders."""

import glob
import os
from typing import Dict, List, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


flags.DEFINE_string('root_dir', '.', 'Directory to search for output folders.')
flags.DEFINE_string('pattern', 'edge-attn-fw-len-*',
                    'Glob pattern for output folders.')
flags.DEFINE_string('output_path', 'entropy_vs_length.png',
                    'Path to save the plot.')
flags.DEFINE_boolean('use_median', False,
                     'Use median instead of mean for aggregation.')

FLAGS = flags.FLAGS


def _parse_length(folder_name: str) -> int:
  parts = folder_name.split('-len-')
  if len(parts) < 2:
    raise ValueError(f'Folder name does not contain length: {folder_name}')
  return int(parts[-1])


def _read_entropy_csv(
    csv_path: str,
) -> Tuple[List[float], List[Tuple[int, int, float]]]:
  values = []
  head_values = []
  with open(csv_path, 'r') as f:
    header = f.readline()
    if not header.startswith('layer,head,mean_entropy'):
      logging.warning('Unexpected header in %s: %s', csv_path, header.strip())
    for line in f:
      parts = line.strip().split(',')
      if len(parts) != 3:
        continue
      layer = int(parts[0])
      head = int(parts[1])
      entropy = float(parts[2])
      values.append(entropy)
      head_values.append((layer, head, entropy))
  return values, head_values


def _collect_entropies() -> Tuple[
  Dict[int, List[float]],
  Dict[int, List[Tuple[int, int, float]]],
]:
  base = os.path.join(FLAGS.root_dir, FLAGS.pattern)
  folders = sorted(glob.glob(base))
  if not folders:
    raise FileNotFoundError(f'No folders matched pattern: {base}')

  length_to_values: Dict[int, List[float]] = {}
  length_to_heads: Dict[int, List[Tuple[int, int, float]]] = {}
  for folder in folders:
    length = _parse_length(os.path.basename(folder))
    csv_paths = glob.glob(os.path.join(folder, 'example_*', 'entropy.csv'))
    if not csv_paths:
      logging.warning('No entropy.csv found under %s', folder)
      continue
    values: List[float] = []
    head_values: List[Tuple[int, int, float]] = []
    for csv_path in csv_paths:
      v, h = _read_entropy_csv(csv_path)
      values.extend(v)
      head_values.extend(h)
    if values:
      length_to_values.setdefault(length, []).extend(values)
    if head_values:
      length_to_heads.setdefault(length, []).extend(head_values)
  return length_to_values, length_to_heads


def _aggregate(values: List[float]) -> float:
  if FLAGS.use_median:
    return float(np.median(values))
  return float(np.mean(values))


def _plot(
    length_to_values: Dict[int, List[float]],
    length_to_heads: Dict[int, List[Tuple[int, int, float]]],
) -> None:
  items: List[Tuple[int, float]] = []
  std_items: List[Tuple[int, float]] = []
  for length, values in length_to_values.items():
    if values:
      items.append((length, _aggregate(values)))
      std_items.append((length, float(np.std(values))))
  if not items:
    raise ValueError('No entropy values found to plot.')

  items.sort(key=lambda x: x[0])
  lengths = [x[0] for x in items]
  entropies = [x[1] for x in items]
  stds = [x[1] for x in sorted(std_items, key=lambda x: x[0])]

  plt.figure(figsize=(6, 4))
  mean_line = np.array(entropies)
  std_band = np.array(stds)

  # Plot individual head entropies as lines, colored by layer.
  all_layers = sorted({layer for heads in length_to_heads.values()
                       for layer, _, _ in heads})
  cmap = plt.get_cmap('tab10', max(1, len(all_layers)))
  layer_to_color = {layer: cmap(i) for i, layer in enumerate(all_layers)}

  # Aggregate per (layer, head) across lengths.
  head_series: Dict[Tuple[int, int], Dict[int, List[float]]] = {}
  for length, heads in length_to_heads.items():
    for layer, head, entropy in heads:
      head_series.setdefault((layer, head), {}).setdefault(length, []).append(entropy)

  # for (layer, head), series in head_series.items():
  #   ys = []
  #   for length in lengths:
  #     vals = series.get(length)
  #     ys.append(float(np.mean(vals)) if vals else np.nan)
  #   plt.plot(lengths, ys, color=layer_to_color[layer], alpha=0.4,
  #            linewidth=1.0, zorder=1)

  # Mean line and band on top for readability.
  plt.fill_between(lengths, mean_line - std_band, mean_line + std_band,
                   color='black', alpha=0.15, linewidth=1, zorder=2)
  plt.plot(lengths, mean_line, marker='o', color='black', label='mean',
           linewidth=2.0, zorder=3, markersize=4)

  # Theoretical maximum entropy (uniform over N keys): ln(N).
  max_entropy = np.log(np.array(lengths, dtype=float))
  plt.plot(lengths, max_entropy, color='red', linewidth=1.5,
           label='max entropy', zorder=4)

  # if all_layers:
  #   handles = [plt.Line2D([0], [0], color=layer_to_color[layer],
  #                         label=f'layer {layer}')
  #              for layer in all_layers]
    # handles.append(plt.Line2D([0], [0], color='black', label='mean'))
    # handles.append(plt.Line2D([0], [0], color='red', label='max entropy'))
    # plt.legend(handles=handles, title='Layer', fontsize=8, title_fontsize=9,
    #            loc='best')
  # else:
  plt.legend(loc='best', fontsize=8)
  plt.xlabel('Length (log2)')
  plt.xscale('log', base=2)
  plt.ylabel('Mean entropy' if not FLAGS.use_median else 'Median entropy')
  plt.title('Attention entropy vs length')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(FLAGS.output_path, dpi=200)
  plt.close()


def main(unused_argv):
  length_to_values, length_to_heads = _collect_entropies()
  _plot(length_to_values, length_to_heads)
  logging.info('Saved plot to %s', FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)
