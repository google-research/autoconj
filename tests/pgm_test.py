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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import autograd.numpy as np
from autograd import grad
from collections import OrderedDict, defaultdict
from itertools import product
from itertools import chain

from autoconj import pgm
from autoconj.tracers import logdet
from autoconj.exponential_families import (
    init_suffstat, StructuredNormalSuffStat, StructuredCategoricalSuffStat)

def find_argmax(natparam_grad, num_nodes):
  """Given the gradient of tree_categorical_maximum w.r.t. the natural
  parameters, finds the argmax of the tree categorical's log-joint."""
  not_found = set(range(num_nodes)) # haven't found the argmax for these nodes
  argmax = [0 for _ in range(num_nodes)]
  factor_iter = chain(natparam_grad.single_onehot_xis.iteritems(),
                      natparam_grad.joint_onehot_xis.iteritems())

  while not_found:
    factor, param = factor_iter.next()
    if not_found.intersection(set(factor)):
      nonzero = np.nonzero(param)
      for i, node in enumerate(factor):
        argmax[node] = int(nonzero[i][0])
        not_found.discard(node)
  return argmax

def _add_diag(tau, J):
  return J + np.einsum('...i,j,ij->...ij', tau, np.ones(tau.shape[-1]),
                       np.eye(tau.shape[-1]))

def make_struct_normal_natparam(factors, sizes, single_covars=True,
                                single_diags=True, means=True,
                                xi_xjtrs=True, xi_times_xjs=True):
  """Makes random natural parameter values for a structured normal distribution,
  given which factors are present in the graphical model.
  """
  natparam = init_suffstat(StructuredNormalSuffStat)
  if not single_covars and not single_diags:
    assert False
  for factor in factors:

    if len(factor) == 1:
      node, node_size = factor[0], sizes[factor[0]]
      if means:
        natparam.xis[(node,)] = np.random.randn(node_size)
      if single_covars:
        sqrtJ = np.random.randn(node_size, 2*node_size)
        halfminusJ = -0.5 * np.dot(sqrtJ, sqrtJ.T)
        natparam.xi_xitrs[(node,)] = halfminusJ
      if single_diags:
        halfminustau = -0.5 * np.exp(np.random.randn(node_size))
        natparam.xi_squareds[(node,)] = halfminustau

    else:
      v1, v2 = factor[0], factor[1]
      size1, size2 = sizes[factor[0]], sizes[factor[1]]
      if xi_xjtrs:
        natparam.xi_xjtrs[(v1,v2)] = np.random.randn(size1, size2)
      if size1 == size2 and xi_times_xjs:
        natparam.xi_times_xjs[(v1, v2)] = np.random.randn(size1)

  return natparam

def make_struct_categorical_natparam(factors, sizes):
  natparam = init_suffstat(StructuredCategoricalSuffStat)
  for factor in factors:
    factor_sizes = [sizes[node] for node in factor]
    if len(factor) > 1:
      natparam.joint_onehot_xis[factor] = np.random.randn(*factor_sizes)
    else:
      natparam.single_onehot_xis[factor] = np.random.randn(*factor_sizes)
  return natparam

def actual_categorical_log_normalizer(natparam, sizes):
  normalizer = 0
  for x in product(*[np.arange(size) for size in sizes]):
    logp_x = 0
    for factor, param in chain(natparam.single_onehot_xis.iteritems(),
                               natparam.joint_onehot_xis.iteritems()):
      idx = tuple([x[node] for node in factor])
      logp_x += param[idx]
    normalizer += np.exp(logp_x)
  return np.log(normalizer)

def actual_normal_log_normalizer(natparam, factors, sizes):

  def make_dense_precision_matrix(natparam, sizes):
    dims = [sum(sizes[:n]) for n in range(len(sizes)+1)]
    prec = np.zeros((dims[-1], dims[-1]))
    for factor, minusJ in natparam.xi_xjtrs.iteritems():
      node1, node2 = factor
      prec[dims[node1]:dims[node1+1], dims[node2]:dims[node2+1]] += -minusJ
      prec[dims[node2]:dims[node2+1], dims[node1]:dims[node1+1]] += -minusJ.T
    for factor, minustau in natparam.xi_times_xjs.iteritems():
      node1, node2 = factor
      prec[dims[node1]:dims[node1+1], dims[node2]:dims[node2+1]] += \
                                                             -pgm.diag(minustau)
      prec[dims[node2]:dims[node2+1], dims[node1]:dims[node1+1]] += \
                                                           -pgm.diag(minustau).T
    for factor, halfminusJ in natparam.xi_xitrs.iteritems():
      node, = factor
      prec[dims[node]:dims[node+1], dims[node]:dims[node+1]] += -2*halfminusJ
    for factor, halfminustau in natparam.xi_squareds.iteritems():
      node, = factor
      prec[dims[node]:dims[node+1], dims[node]:dims[node+1]] += \
                                                       -2*pgm.diag(halfminustau)
    return prec

  prec = make_dense_precision_matrix(natparam, sizes)
  inv_prec = np.linalg.inv(prec)
  h = np.concatenate([natparam.xis.get((n,), (np.zeros(sizes[n]),))
                      for n in range(len(sizes))])
  log_normalizer = 0.5 * np.dot(h, np.dot(inv_prec, h))
  log_normalizer -= 0.5*logdet(prec) + 0.5*sum(sizes)*np.log(2*np.pi)
  return log_normalizer

def actual_categorical_maximum(natparam, sizes):
  max_logp = -float('inf')
  argmax = None
  for x in product(*[np.arange(size) for size in sizes]):
    logp_x = struct_categorical_logpdf(x, natparam)
    if logp_x > max_logp:
      max_logp = logp_x
      argmax = x
  return max_logp, argmax

def struct_categorical_logpdf(x, natparam):
  logp_x = 0
  for factor, param in chain(natparam.single_onehot_xis.iteritems(),
                             natparam.joint_onehot_xis.iteritems()):
    idx = tuple([x[node] for node in factor])
    logp_x += param[idx]
  return logp_x


class PgmTest(absltest.TestCase):

  def testNormalTreeLogNormalizerChain(self):
    T = 10
    factors = [(n,) for n in range(T)] + [(n, n+1) for n in range(T-1)]
    sizes = np.random.choice(10, size=(T,))

    natparam = make_struct_normal_natparam(factors, sizes)
    elim_order = range(T)
    tree_normal_log_normalizer = pgm.make_tree_normal_log_normalizer(elim_order)

    actual_log_normalizer = actual_normal_log_normalizer(natparam, factors,
                                                         sizes)
    log_normalizer = tree_normal_log_normalizer(natparam)
    self.assertTrue(np.allclose(actual_log_normalizer, log_normalizer))

  def testNormalTreeLogNormalizerWheel(self):
    num_nodes = 10
    factors = [(n,) for n in range(num_nodes)] +\
              [(0, n) for n in range(1, num_nodes)]
    sizes = np.random.choice(10, size=(num_nodes,))

    natparam = make_struct_normal_natparam(factors, sizes)
    elim_order = range(1, num_nodes) + [0]
    tree_normal_log_normalizer = pgm.make_tree_normal_log_normalizer(elim_order)

    actual_log_normalizer = actual_normal_log_normalizer(natparam, factors,
                                                         sizes)
    log_normalizer = tree_normal_log_normalizer(natparam)
    self.assertTrue(np.allclose(actual_log_normalizer, log_normalizer))

  def testNormalTreeLogNormalizerGeneric(self):
    factors = [(n,) for n in range(10)]
    factors += [(0,1), (0,8), (1,4), (1,5), (1,2), (2,3), (2,6), (2,7), (2,9)]
    sizes = np.random.choice(9, size=(10,)) + 1

    natparam = make_struct_normal_natparam(factors, sizes)
    elim_order = [9, 4, 5, 6, 7, 3, 2, 1, 8, 0]
    tree_normal_log_normalizer = pgm.make_tree_normal_log_normalizer(elim_order)

    actual_log_normalizer = actual_normal_log_normalizer(natparam, factors,
                                                         sizes)
    log_normalizer = tree_normal_log_normalizer(natparam)
    self.assertTrue(np.allclose(actual_log_normalizer, log_normalizer))

  def testCategoricalTreeLogNormalizerSimple(self):
    factors = [(0,), (1,), (2,), (0,1,2)]
    sizes = [2, 3, 4]

    natparam = make_struct_categorical_natparam(factors, sizes)
    elim_order = [0, 1, 2]
    categorical_tree_log_normalizer =\
        pgm.make_tree_categorical_log_normalizer(elim_order)

    actual_log_normalizer = actual_categorical_log_normalizer(natparam,
                                                              sizes=(2,3,4))
    log_normalizer = categorical_tree_log_normalizer(natparam)

    self.assertTrue(np.allclose(actual_log_normalizer, log_normalizer))

  def testCategoricalTreeLogNormalizerGeneric(self):
    factors = [(0,1,2,3), (1,4,5), (5,), (2,), (3,6,7), (0,8)]
    sizes = [2, 3, 4, 2, 3, 4, 2, 3, 4]

    natparam = make_struct_categorical_natparam(factors, sizes)
    elim_order = [5, 6, 8, 2, 7, 4, 3, 1, 0]
    categorical_tree_log_normalizer =\
        pgm.make_tree_categorical_log_normalizer(elim_order)

    actual_log_normalizer = actual_categorical_log_normalizer(natparam, sizes)
    log_normalizer = categorical_tree_log_normalizer(natparam)
    self.assertTrue(np.allclose(actual_log_normalizer, log_normalizer))

  # def testCategoricalTreeFindMaximumSimple(self):
  #   factors = [(0,), (1,), (2,), (0,1,2)]
  #   sizes = [2, 3, 4]

  #   natparam = make_struct_categorical_natparam(factors, sizes)
  #   elim_order = [0, 1, 2]
  #   tree_categorical_maximum = pgm.make_tree_categorical_maximum(elim_order)

  #   actual_maximum, actual_argmax = actual_categorical_maximum(natparam, sizes)
  #   maximum = tree_categorical_maximum(natparam)
  #   argmax = find_argmax(grad(tree_categorical_maximum)(natparam), num_nodes=3)

  #   self.assertTrue(np.allclose(actual_maximum, maximum))
  #   self.assertTrue(np.allclose(actual_maximum,
  #                               struct_categorical_logpdf(argmax, natparam)))

  # def testCategoricalTreeFindMaximumGeneric(self):
  #   factors = [(0,1,2,3), (1,4,5), (5,), (2,), (3,6,7), (0,8)]
  #   sizes = [2, 3, 4, 2, 3, 4, 2, 3, 4]

  #   natparam = make_struct_categorical_natparam(factors, sizes)
  #   elim_order = [5, 6, 8, 2, 7, 4, 3, 1, 0]
  #   tree_categorical_maximum = pgm.make_tree_categorical_maximum(elim_order)

  #   actual_maximum, actual_argmax = actual_categorical_maximum(natparam, sizes)
  #   maximum = tree_categorical_maximum(natparam)
  #   argmax = find_argmax(grad(tree_categorical_maximum)(natparam), num_nodes=9)

  #   self.assertTrue(np.allclose(actual_maximum, maximum))
  #   self.assertTrue(np.allclose(actual_maximum,
  #                               struct_categorical_logpdf(argmax, natparam)))

if __name__ == '__main__':
  absltest.main()
