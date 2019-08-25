# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoderOutput, _mask_probs, _maybe_tensor_gather_helper, _tensor_gather_helper

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


__all__ = [
    "TrieDecoder",
]

class TrieDecoderState(collections.namedtuple("TrieDecoderState", ("cell_state", "trie_nodes"))):
    pass

class BeamTrieDecoderState(
    collections.namedtuple("BeamTrieDecoderState", ("cell_state", "log_probs", "finished", "lengths", "trie_nodes"))):
  pass

def _get_scores(log_probs, sequence_lengths, length_penalty_weight):
    length_penality_ = _length_penalty(sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)
    return log_probs + length_penality_
                                  
def _length_penalty(sequence_lengths, penalty_factor):
    penalty_factor = ops.convert_to_tensor(penalty_factor, name="penalty_factor")
    penalty_factor.set_shape(())  # penalty should be a scalar.
    static_penalty = tensor_util.constant_value(penalty_factor)
    if static_penalty is not None and static_penalty == 0:
        return 1.0 
    return math_ops.to_float(sequence_lengths) * penalty_factor

class TrieDecoder(BasicDecoder):
  def initialize(self, name=None):
    finished, first_inputs, trie_nodes = self._helper.initialize()
    return (finished, first_inputs, TrieDecoderState(self._initial_state, trie_nodes))

  def step(self, time, inputs, state, name=None):
    with ops.name_scope(name, "TrieDecoderStep", (time, inputs, state)):
      cell_outputs, next_cell_state = self._cell(inputs, state.cell_state) 
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids, next_trie_nodes = self._helper.sample(
          time=time, outputs=cell_outputs, state=state)
      trie_decoder_state = TrieDecoderState(next_cell_state, next_trie_nodes)
      (finished, next_inputs, trie_decoder_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=trie_decoder_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, trie_decoder_state, next_inputs, finished)

class BeamTrieDecoder(BeamSearchDecoder):
  def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               decoder_trie,
               output_layer=None,
               length_penalty_weight=0.0):

    super(BeamTrieDecoder, self).__init__(
            cell=cell,
            embedding=embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=initial_state,
            beam_width=beam_width,
            output_layer=output_layer,
            length_penalty_weight=length_penalty_weight
            )
    self._decoder_trie_fn = (lambda ids: embedding_ops.embedding_lookup(decoder_trie, ids))

  def initialize(self, name=None):
    finished, start_inputs = self._finished, self._start_inputs

    initial_state = BeamTrieDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=array_ops.zeros(
            [self._batch_size, self._beam_width],
            dtype=nest.flatten(self._initial_cell_state)[0].dtype),
        finished=finished,
        lengths=array_ops.zeros(
            [self._batch_size, self._beam_width], dtype=dtypes.int64),
        trie_nodes=array_ops.zeros(
            [self._batch_size, self._beam_width], dtype=dtypes.int32)
        )
    return (finished, start_inputs, initial_state)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    batch_size = self._batch_size
    beam_width = self._beam_width
    end_token = self._end_token
    length_penalty_weight = self._length_penalty_weight

    with ops.name_scope(name, "BeamTrieDecoderStep", (time, inputs, state)):
      cell_state = state.cell_state
      inputs = nest.map_structure(
          lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
      cell_state = nest.map_structure(
          self._maybe_merge_batch_beams,
          cell_state, self._cell.state_size)
      cell_outputs, next_cell_state = self._cell(inputs, cell_state)
      cell_outputs = nest.map_structure(
          lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
      next_cell_state = nest.map_structure(
          self._maybe_split_batch_beams,
          next_cell_state, self._cell.state_size)

      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)

      beam_search_output, beam_search_state = self.trie_beam_step(
          time=time,
          logits=cell_outputs,
          next_cell_state=next_cell_state,
          beam_state=state,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight)

      finished = beam_search_state.finished
      sample_ids = beam_search_output.predicted_ids
      next_inputs = control_flow_ops.cond(
          math_ops.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))

    return (beam_search_output, beam_search_state, next_inputs, finished)

  def trie_beam_step(self, time, logits, next_cell_state, beam_state, batch_size,
                        beam_width, end_token, length_penalty_weight, diversity_lambda=1.0, name=None):
    
    time = ops.convert_to_tensor(time, name="time")
    static_batch_size = tensor_util.constant_value(batch_size)
  
    # Calculate the current lengths of the predictions
    prediction_lengths = beam_state.lengths
    previously_finished = beam_state.finished
  
    # Calculate the total log probs for the new hypotheses
    # Final Shape: [batch_size, beam_width, vocab_size]
    step_probs = nn_ops.softmax(logits)
    step_log_probs = math_ops.log(step_probs)

    #negative inf
    inf = 10000.0 

    #vocab size
    vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]
  
    # Mask trie
    child_nodes = self._decoder_trie_fn(beam_state.trie_nodes)
  
    trie_finished_mask = array_ops.expand_dims(
            math_ops.to_float(1. - math_ops.to_float(previously_finished)), 2)
    trie_mask_neg = array_ops.ones_like(step_log_probs, dtype=step_log_probs.dtype)
    trie_mask_neg = trie_mask_neg * step_log_probs.dtype.min
    trie_mask_pos = array_ops.zeros_like(step_log_probs, dtype=step_log_probs.dtype)
    trie_mask = array_ops.where(gen_math_ops.equal(child_nodes, 0),
                            trie_mask_neg,
                            trie_mask_pos)
    step_log_probs = step_log_probs + trie_mask * trie_finished_mask

    # choose by prob
    sign = math_ops.less(step_probs - random_ops.random_uniform(shape=[batch_size, beam_width, vocab_size],dtype=step_probs.dtype), 0)
    random_mask = math_ops.cast(sign, step_log_probs.dtype) * inf

    # diversity penalty

    step_log_probs = array_ops.reshape(step_log_probs, [-1, vocab_size])
    _, indices = nn_ops.top_k(step_log_probs, k=vocab_size, sorted=True)
    score_rank = functional_ops.map_fn(gen_array_ops.invert_permutation, indices)
    #diversity_penalty = math_ops.cast(score_rank, step_log_probs.dtype)
    #diversity_penalty = control_flow_ops.case(
    #{
    #    gen_math_ops.equal(time, 0): lambda: inf * math_ops.cast(math_ops.mod(score_rank, 10), diversity_penalty.dtype),
    #},
    #default=lambda: diversity_penalty * inf,
    #)
    step_log_probs -= math_ops.cast(score_rank * time, dtype=step_log_probs.dtype)

    step_log_probs = array_ops.reshape(step_log_probs, [batch_size, beam_width, vocab_size])

    #step_log_probs -= random_mask

    # Mask finish
    step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
    total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs
  
    # Calculate the continuation lengths by adding to all continuing beams.
    lengths_to_add = array_ops.one_hot(
        indices=array_ops.fill([batch_size, beam_width], end_token),
        depth=vocab_size,
        on_value=np.int64(0), off_value=np.int64(1),
        dtype=dtypes.int64)
    add_mask = math_ops.to_int64(math_ops.logical_not(previously_finished))
    lengths_to_add *= array_ops.expand_dims(add_mask, 2)
    new_prediction_lengths = (
        lengths_to_add + array_ops.expand_dims(prediction_lengths, 2))
  
    # Calculate the scores for each beam
    scores = _get_scores(
        log_probs=total_probs,
        sequence_lengths=new_prediction_lengths,
        length_penalty_weight=length_penalty_weight)
  
    # During the first time step we only consider the initial beam
    scores_shape = array_ops.shape(scores)
    scores_flat = control_flow_ops.cond(
        time > 0,
        lambda: array_ops.reshape(scores, [batch_size, -1]),
        lambda: scores[:, 0])
    num_available_beam = control_flow_ops.cond(
        time > 0, lambda: math_ops.reduce_prod(scores_shape[1:]),
        lambda: math_ops.reduce_prod(scores_shape[2:]))

    # Pick the next beams according to the specified successors function
    next_beam_size = math_ops.minimum(
        ops.convert_to_tensor(beam_width, dtype=dtypes.int32, name="beam_width"),
        num_available_beam)

    # [batch_size, next_beam_width]
    next_beam_scores, word_indices = nn_ops.top_k(scores_flat, k=next_beam_size)
  
    next_beam_scores.set_shape([static_batch_size, beam_width])
    word_indices.set_shape([static_batch_size, beam_width])
  
    # Pick out the probs, beam_ids, and states according to the chosen predictions
    next_beam_probs = _tensor_gather_helper(
        gather_indices=word_indices,
        gather_from=total_probs,
        batch_size=batch_size,
        range_size=beam_width * vocab_size,
        gather_shape=[-1])
    # Note: just doing the following
    #   math_ops.to_int32(word_indices % vocab_size,
    #       name="next_beam_word_ids")
    # would be a lot cleaner but for reasons unclear, that hides the results of
    # the op which prevents capturing it with tfdbg debug ops.
    raw_next_word_ids = math_ops.mod(word_indices, vocab_size,
                                     name="next_beam_word_ids")
    next_word_ids = math_ops.to_int32(raw_next_word_ids)
    next_beam_ids = math_ops.to_int32(word_indices / vocab_size,
                                      name="next_beam_parent_ids")
  
    # Append new ids to current predictions
    previously_finished = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=previously_finished,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[-1])
    next_finished = math_ops.logical_or(previously_finished,
                                        math_ops.equal(next_word_ids, end_token),
                                        name="next_beam_finished")
  
    # Calculate the length of the next predictions.
    # 1. Finished beams remain unchanged
    # 2. Beams that are now finished (EOS predicted) remain unchanged
    # 3. Beams that are not yet finished have their length increased by 1
    lengths_to_add = math_ops.to_int64(math_ops.logical_not(previously_finished))
    next_prediction_len = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=beam_state.lengths,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[-1])
    next_prediction_len += lengths_to_add
  
    # Pick out the cell_states according to the next_beam_ids. We use a
    # different gather_shape here because the cell_state tensors, i.e.
    # the tensors that would be gathered from, all have dimension
    # greater than two and we need to preserve those dimensions.
    # pylint: disable=g-long-lambda
    next_cell_state = nest.map_structure(
        lambda gather_from: _maybe_tensor_gather_helper(
            gather_indices=next_beam_ids,
            gather_from=gather_from,
            batch_size=batch_size,
            range_size=beam_width,
            gather_shape=[batch_size * beam_width, -1]),
        next_cell_state)
    # pylint: enable=g-long-lambda
  
    next_trie_nodes = _tensor_gather_helper(
        gather_indices=word_indices,
        gather_from=child_nodes,
        batch_size=batch_size,
        range_size=beam_width * vocab_size,
        gather_shape=[-1])
  
    next_state = BeamTrieDecoderState(
        cell_state=next_cell_state,
        log_probs=next_beam_probs,
        lengths=next_prediction_len,
        finished=next_finished,
        trie_nodes=next_trie_nodes)
  
    output = BeamSearchDecoderOutput(
        scores=next_beam_scores,
        predicted_ids=next_word_ids,
        parent_ids=next_beam_ids)
  
    return output, next_state
