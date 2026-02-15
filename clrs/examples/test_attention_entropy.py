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
"""Test script for attention entropy behavior on synthetic matrices."""

from absl import app
from absl import flags
from absl import logging
import numpy as np


flags.DEFINE_integer('n', 8, 'Number of nodes (N).')
flags.DEFINE_integer('heads', 12, 'Number of heads (H).')
flags.DEFINE_integer('batch', 1, 'Batch size (B).')
flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS


def _entropy(att: np.ndarray) -> float:
  # att: [B, N, N, N, H], entropy over keys (axis=1)
  att = np.clip(att, 1e-12, 1.0)
  ent = -np.sum(att * np.log(att), axis=1)
  return float(np.mean(ent))


def _uniform_attention(b: int, n: int, h: int) -> np.ndarray:
  att = np.full((b, n, n, n, h), 1.0 / n, dtype=np.float32)
  return att


def _sharp_attention(b: int, n: int, h: int) -> np.ndarray:
  att = np.zeros((b, n, n, n, h), dtype=np.float32)
  idx = np.random.randint(0, n, size=(b, n, n, h))
  for bi in range(b):
    for q in range(n):
      for t in range(n):
        for hi in range(h):
          att[bi, idx[bi, q, t, hi], q, t, hi] = 1.0
  return att


def _peaked_attention(b: int, n: int, h: int, peak: float = 0.8) -> np.ndarray:
  att = np.full((b, n, n, n, h), (1.0 - peak) / (n - 1), dtype=np.float32)
  idx = np.random.randint(0, n, size=(b, n, n, h))
  for bi in range(b):
    for q in range(n):
      for t in range(n):
        for hi in range(h):
          att[bi, idx[bi, q, t, hi], q, t, hi] = peak
  return att


def _noisy_uniform(b: int, n: int, h: int, noise: float = 0.05) -> np.ndarray:
  base = np.full((b, n, n, n, h), 1.0 / n, dtype=np.float32)
  jitter = np.random.randn(b, n, n, n, h).astype(np.float32) * noise
  att = np.clip(base + jitter, 1e-6, None)
  att /= np.sum(att, axis=1, keepdims=True)
  return att


def main(unused_argv):
  np.random.seed(FLAGS.seed)
  b, n, h = FLAGS.batch, FLAGS.n, FLAGS.heads

  examples = {
      'uniform': _uniform_attention(b, n, h),
      'sharp': _sharp_attention(b, n, h),
      'peaked_0.8': _peaked_attention(b, n, h, peak=0.8),
      'noisy_uniform': _noisy_uniform(b, n, h, noise=0.05),
  }

  logging.info('Theoretical max entropy (ln N): %.6f', np.log(n))
  for name, att in examples.items():
    logging.info('%s entropy: %.6f', name, _entropy(att))


if __name__ == '__main__':
  app.run(main)
