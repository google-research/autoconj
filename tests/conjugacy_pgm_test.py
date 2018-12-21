from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import chain
import autograd.numpy as np
import time

from absl.testing import absltest

from autoconj.conjugacy import complete_conditional, marginalize
from autoconj.tracers import one_hot
from autoconj.util import SupportTypes
from pgm_test import (actual_categorical_log_normalizer,
                      actual_normal_log_normalizer,
                      make_struct_categorical_natparam,
                      make_struct_normal_natparam)


def structured_normal_logpdf(x, natparam):
  logp = 0
  for factor, mean in natparam.xis.iteritems():
    logp += np.dot(x[factor[0]], mean)
  for factor, halfminusJ in natparam.xi_xitrs.iteritems():
    logp += np.dot(x[factor[0]], np.dot(halfminusJ, x[factor[0]]))
  for factor, halfminustau in natparam.xi_squareds.iteritems():
    logp += np.dot(x[factor[0]], halfminustau*x[factor[0]])
  for factor, minusJ in natparam.xi_xjtrs.iteritems():
    node1, node2 = factor
    logp += np.dot(x[node1], np.dot(minusJ, x[node2]))
  for factor, minustau in natparam.xi_times_xjs.iteritems():
    node1, node2 = factor
    logp += np.dot(x[node1], minustau*x[node2])
  return logp


def structured_categorical_logpdf(x, natparam, dim):
  logp = 0
  alphabet = 'abcdefghijklmnopqrstuvwxyz'
  factor_iter = chain(natparam.single_onehot_xis.iteritems(),
                      natparam.joint_onehot_xis.iteritems())
  for factor, param in factor_iter:
    factor_idxs = ''.join(alphabet[i] for i in range(len(factor)))
    in_formula = ','.join([factor_idxs] +
                          [alphabet[i] for i in range(len(factor))])
    formula = '{}->'.format(in_formula)
    logp += np.einsum(formula, param,
                      *[one_hot(x[node], dim) for node in factor])
  return logp


def _condition_and_marginalize(log_joint, argnum, support, *args):
  sub_args = args[:argnum] + args[argnum + 1:]

  marginalized = marginalize(log_joint, argnum, support, *args)
  marginalized_value = marginalized(*sub_args)

  conditional_factory = complete_conditional(log_joint, argnum, support, *args)
  conditional = conditional_factory(*sub_args)

  return conditional, marginalized_value


class ConjugacyPgmTest(absltest.TestCase):

  def testNormalChain(self):
    n_timesteps = 10
    dim = 10
    factors = ([(n,) for n in range(n_timesteps)] +
               [(n, n+1) for n in range(n_timesteps-1)])
    sizes = [dim]*n_timesteps

    natparam = make_struct_normal_natparam(factors, sizes)
    log_joint = lambda x: structured_normal_logpdf(x, natparam)
    x = np.ones((n_timesteps, dim))

    start_time = time.time()
    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.REAL, x))

    start_time = time.time()
    correct_marginalized_value = actual_normal_log_normalizer(natparam, factors,
                                                              sizes)
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

  def testNormalGenericTree(self):
    factors = [(n,) for n in range(10)]
    factors += [(0,1), (0,8), (1,4), (1,5), (1,2), (2,3), (2,6), (2,7), (2,9)]
    dim = 10
    sizes = [dim]*10

    natparam = make_struct_normal_natparam(factors, sizes)
    log_joint = lambda x: structured_normal_logpdf(x, natparam)
    x = np.ones((10, dim))

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.REAL, x))

    correct_marginalized_value = actual_normal_log_normalizer(natparam, factors,
                                                              sizes)
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

  def testCategoricalGenericTree(self):
    num_nodes = 9
    dim = 2
    factors = [(0,1,2,3), (1,4,5), (5,), (2,), (3,6,7), (0,8)]
    sizes = [dim]*num_nodes

    natparam = make_struct_categorical_natparam(factors, sizes)
    log_joint = lambda x: structured_categorical_logpdf(x, natparam, dim)
    x = np.random.choice(dim, size=(num_nodes,))

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.INTEGER, x))

    correct_marginalized_value = actual_categorical_log_normalizer(natparam,
                                                                   sizes)
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value,
                           places=4)


if __name__ == '__main__':
  absltest.main()
