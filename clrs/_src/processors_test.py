# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Tests for processors.py."""

from absl.testing import absltest
import chex
from clrs._src import processors
import haiku as hk
import jax.numpy as jnp


class MemnetTest(absltest.TestCase):

  def test_simple_run_and_check_shapes(self):

    batch_size = 64
    vocab_size = 177
    embedding_size = 64
    sentence_size = 11
    memory_size = 320
    linear_output_size = 128
    num_hops = 2
    use_ln = True

    def forward_fn(queries, stories):
      model = processors.MemNetFull(
          vocab_size=vocab_size,
          embedding_size=embedding_size,
          sentence_size=sentence_size,
          memory_size=memory_size,
          linear_output_size=linear_output_size,
          num_hops=num_hops,
          use_ln=use_ln)
      return model._apply(queries, stories)

    forward = hk.transform(forward_fn)

    queries = jnp.ones([batch_size, sentence_size], dtype=jnp.int32)
    stories = jnp.ones([batch_size, memory_size, sentence_size],
                       dtype=jnp.int32)

    key = hk.PRNGSequence(42)
    params = forward.init(next(key), queries, stories)

    model_output = forward.apply(params, None, queries, stories)
    chex.assert_shape(model_output, [batch_size, vocab_size])
    chex.assert_type(model_output, jnp.float32)


class EdgeTransformerVariantsTest(absltest.TestCase):

  def _run_variant(self, kind: str):
    batch_size = 2
    num_nodes = 4
    out_size = 8
    node_dim = 6
    hidden_dim = 5
    graph_dim = 3
    edge_dim = out_size

    def forward_fn(node_fts, edge_fts, graph_fts, adj_mat, hidden):
      factory = processors.get_processor_factory(
          kind=kind,
          use_ln=False,
          nb_triplet_fts=0,
          nb_heads=2,
      )
      processor = factory(
          out_size=out_size,
          num_layers=1,
          attention_dropout=0.0,
          activation='relu',
          norm_first=False,
      )
      return processor(
          node_fts=node_fts,
          edge_fts=edge_fts,
          graph_fts=graph_fts,
          adj_mat=adj_mat,
          hidden=hidden,
          repred=False,
          readout='diagonal',
          is_graph_fts_avail=True,
      )

    forward = hk.transform(forward_fn)

    node_fts = jnp.ones((batch_size, num_nodes, node_dim), dtype=jnp.float32)
    edge_fts = jnp.ones((batch_size, num_nodes, num_nodes, edge_dim), dtype=jnp.float32)
    graph_fts = jnp.ones((batch_size, graph_dim), dtype=jnp.float32)
    adj_mat = jnp.ones((batch_size, num_nodes, num_nodes), dtype=jnp.float32)
    hidden = jnp.ones((batch_size, num_nodes, hidden_dim), dtype=jnp.float32)

    key = hk.PRNGSequence(0)
    params = forward.init(next(key), node_fts, edge_fts, graph_fts, adj_mat, hidden)
    node_out, edge_out = forward.apply(
        params, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
    )

    chex.assert_shape(node_out, (batch_size, num_nodes, out_size))
    chex.assert_shape(edge_out, (batch_size, num_nodes, num_nodes, out_size))
    chex.assert_type(node_out, jnp.float32)
    chex.assert_type(edge_out, jnp.float32)

  def test_edge_t_adpt_simple_run_and_check_shapes(self):
    self._run_variant('edge_t_adpt')

  def test_edge_t_gated_simple_run_and_check_shapes(self):
    self._run_variant('edge_t_gated')

  def test_edge_t_adpt_inc_simple_run_and_check_shapes(self):
    self._run_variant('edge_t_adpt_inc')

  def test_edge_t_adpt_dec_simple_run_and_check_shapes(self):
    self._run_variant('edge_t_adpt_dec')

  def test_edge_t_zero_simple_run_and_check_shapes(self):
    self._run_variant('edge_t_zero')


class LinearProcessorTest(absltest.TestCase):

  def test_linear_simple_run_and_check_shapes(self):
    batch_size = 2
    num_nodes = 4
    out_size = 8
    node_dim = 6
    hidden_dim = 5
    graph_dim = 3
    edge_dim = out_size

    def forward_fn(node_fts, edge_fts, graph_fts, adj_mat, hidden):
      factory = processors.get_processor_factory(
          kind='linear',
          use_ln=False,
          nb_triplet_fts=0,
          nb_heads=2,
      )
      processor = factory(out_size=out_size)
      return processor(
          node_fts=node_fts,
          edge_fts=edge_fts,
          graph_fts=graph_fts,
          adj_mat=adj_mat,
          hidden=hidden,
          repred=False,
          readout='diagonal',
          is_graph_fts_avail=True,
      )

    forward = hk.transform(forward_fn)

    node_fts = jnp.ones((batch_size, num_nodes, node_dim), dtype=jnp.float32)
    edge_fts = jnp.ones((batch_size, num_nodes, num_nodes, edge_dim), dtype=jnp.float32)
    graph_fts = jnp.ones((batch_size, graph_dim), dtype=jnp.float32)
    adj_mat = jnp.ones((batch_size, num_nodes, num_nodes), dtype=jnp.float32)
    hidden = jnp.ones((batch_size, num_nodes, hidden_dim), dtype=jnp.float32)

    key = hk.PRNGSequence(0)
    params = forward.init(next(key), node_fts, edge_fts, graph_fts, adj_mat, hidden)
    node_out, edge_out = forward.apply(
        params, None, node_fts, edge_fts, graph_fts, adj_mat, hidden
    )

    chex.assert_shape(node_out, (batch_size, num_nodes, out_size))
    chex.assert_type(node_out, jnp.float32)
    self.assertIsNone(edge_out)


if __name__ == '__main__':
  absltest.main()
