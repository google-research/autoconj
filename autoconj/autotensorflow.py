# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert expressions to TensorFlow graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import operator

from absl import app

import tensorflow as tf
import autograd.numpy as np
from autograd.fmap_util import container_fmap
from autograd.util import func
from autograd.core import VSpace
from autograd.numpy.numpy_vjps import cast
from autograd.tracer import Node
from autograd.tracer import trace
import autograd.scipy.special as special
import autograd.scipy.misc as misc

from .tracers import one_hot
from .util import split_einsum_formula


def _reconstitute_einsum_formula(input_formulas, output_formula):
  return '{}->{}'.format(','.join(input_formulas), output_formula)


def einsum(formula, *args):
  in_formulas, out_formula = split_einsum_formula(formula)
  in_formulas = [formula.lstrip('...') for formula in in_formulas]
  out_formula = out_formula.lstrip('...')
  formula = _reconstitute_einsum_formula(in_formulas, out_formula)

  args = [tf.cast(arg, tf.float32) for arg in args]
  return tf.einsum(formula, *args)


def recast(op):
  def new_op(*args):
    return op(*map(partial(tf.cast, dtype=tf.float32), args))
  return new_op


np2tf = {
    np.add: recast(operator.add),
    np.subtract: recast(operator.sub),
    np.multiply: recast(operator.mul),
    np.divide: recast(operator.div),
    np.true_divide: recast(operator.div),
    np.einsum: einsum,
    func(VSpace.add_not_none): lambda unused_vs, x, y: x + y,
    cast: tf.cast,
    np.power: tf.pow,
    np.log: tf.log,
    one_hot: tf.one_hot,
    np.sum: tf.reduce_sum,
    special.gammaln: tf.lgamma,
    special.psi: tf.digamma,
    np.reshape: tf.reshape,
    misc.logsumexp: tf.reduce_logsumexp,
    np.exp: tf.exp,
    np.negative: tf.negative,
}


class TFNode(Node):
  __slots__ = ['tensor']

  def __init__(self, tensor):
    self.tensor = tensor

  def process_primitive(self, ans, fun, args, kwargs, parents):
    tensor_args = fun.fmap_in(lambda p, a: p.tensor if p else a, parents, args)
    tf_ans = np2tf[fun](*tensor_args, **kwargs)
    return fun.fmap_out(TFNode, tf_ans)


def make_tffun(fun, *xs):
  def tf_fun(*tf_xs):
    fmap_in = fmap_out = container_fmap
    start_nodes = fmap_in(TFNode, tf_xs)
    end_values, end_nodes = trace(fun, xs, start_nodes, fmap_in, fmap_out)
    return fmap_out(lambda n, v: v if n is None else n.tensor,
                    end_nodes, end_values)
  return tf_fun


def main(unused_argv):

  def fun(x, y):
    z = np.einsum('ij,j->i', x, y)
    out1 = 3. * z + 5.
    out2 = z / 2.
    return (out1, out2)

  np_x = np.ones((2, 3))
  np_y = np.ones(3)

  builder = make_tffun(fun, np_x, np_y)

  x = tf.placeholder(np_x.dtype, np_x.shape)
  y = tf.placeholder(np_y.dtype, np_y.shape)
  outs = builder(x, y)

  print(outs)

if __name__ == '__main__':
  app.run(main)
