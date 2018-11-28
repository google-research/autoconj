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

from absl import app

from .canonicalize import canonicalize
from .conjugacy import (complete_conditional, _extract_conditional_factors,
                        find_sufficient_statistic_nodes, marginalize,
                        split_einsum_node, SupportTypes)
from .exponential_families import batch_dirichlet
from .tracers import (eval_expr, eval_node, GraphExpr, make_expr, one_hot,
                      print_expr)
from . import log_probs

import autograd.numpy as np
from autograd.scipy import special
from autograd.scipy import misc

from scipy import stats


def _condition_and_marginalize(log_joint, argnum, support, *args):
  sub_args = args[:argnum] + args[argnum + 1:]

  marginalized = marginalize(log_joint, argnum, support, *args)
  marginalized_value = marginalized(*sub_args)

  conditional_factory = complete_conditional(log_joint, argnum, support, *args)
  conditional = conditional_factory(*sub_args)

  return conditional, marginalized_value


def testBetaBernoulli():
  def log_joint(p, x, a, b):
    log_prior = ((a - 1) * np.log(p) + (b - 1) * np.log1p(-p) -
                 special.gammaln(a) - special.gammaln(b) +
                 special.gammaln(a + b)).sum()
    log_likelihood = (x * np.log(p) + (1 - x) * np.log1p(-p)).sum()
    return log_prior + log_likelihood
  n_examples = 10
  a = 1.3
  b = 2.4
  p = np.random.beta(a, b, [3, 4])
  x = np.random.uniform(size=(n_examples,) + p.shape) < p
  x = x.astype(np.float32)

  conditional, marginalized_value = (
      _condition_and_marginalize(log_joint, 0, SupportTypes.UNIT_INTERVAL,
                                 p, x, a, b))

  new_a = a + x.sum(0)
  new_b = b + x.shape[0] - x.sum(0)
  correct_marginalized_value = (
      (-special.gammaln(a) - special.gammaln(b)
       + special.gammaln(a + b)) * p.size
      + (special.gammaln(new_a) + special.gammaln(new_b)
         - special.gammaln(new_a + new_b)).sum())
  # self.assertAlmostEqual(marginalized_value, correct_marginalized_value,
  #                        places=4)

  # self.assertTrue(np.allclose(new_a, conditional.args[0]))
  # self.assertTrue(np.allclose(new_b, conditional.args[1]))


def testFactorAnalysis():
  def log_joint(x, w, epsilon, tau, alpha, beta):
    log_p_epsilon = log_probs.norm_gen_log_prob(epsilon, 0, 1)
    log_p_w = log_probs.norm_gen_log_prob(w, 0, 1)
    log_p_tau = log_probs.gamma_gen_log_prob(tau, alpha, beta)
    # TODO(mhoffman): The transposed version below should work.
  #   log_p_x = log_probs.norm_gen_log_prob(x, np.dot(epsilon, w), 1. / np.sqrt(tau))
    log_p_x = log_probs.norm_gen_log_prob(x, np.einsum('ik,jk->ij', epsilon, w),
                                          1. / np.sqrt(tau))
    return log_p_epsilon + log_p_w + log_p_tau + log_p_x

  n_examples = 200
  D = 10
  K = 5
  alpha = 2.
  beta = 8.
  tau = np.random.gamma(alpha, beta)
  w = np.random.normal(loc=0, scale=1, size=[D, K])
  epsilon = np.random.normal(loc=0, scale=1, size=[n_examples, K])
  x = np.random.normal(loc=epsilon.dot(w.T), scale=np.sqrt(tau))
  all_args = [x, w, epsilon, tau, alpha, beta]

  w_conditional_factory = complete_conditional(log_joint, 1,
                                               SupportTypes.REAL, *all_args)
  conditional = w_conditional_factory(x, epsilon, tau, alpha, beta)
  true_cov = np.linalg.inv(tau * np.einsum('nk,nl->kl', epsilon, epsilon) +
                           np.eye(K))
  true_mean = tau * np.einsum('nk,nd,kl->dl', epsilon, x, true_cov)

  epsilon_conditional_factory = complete_conditional(log_joint, 2,
                                                     SupportTypes.REAL,
                                                     *all_args)
  conditional = epsilon_conditional_factory(x, w, tau, alpha, beta)
  true_cov = np.linalg.inv(tau * np.einsum('dk,dl->kl', w, w) + np.eye(K))
  true_mean = tau * np.einsum('dk,nd,kl->nl', w, x, true_cov)

  tau_conditional_factory = complete_conditional(log_joint, 3,
                                                 SupportTypes.NONNEGATIVE,
                                                 *all_args)
  conditional = tau_conditional_factory(x, w, epsilon, alpha, beta)
  true_a = alpha + 0.5 * n_examples * D
  true_b = beta + 0.5 * np.sum(np.square(x - epsilon.dot(w.T)))


def main(argv):
  del argv  # Unused.

  for _ in range(1):
    testFactorAnalysis()
    # testBetaBernoulli()


if __name__ == '__main__':
  app.run(main)
