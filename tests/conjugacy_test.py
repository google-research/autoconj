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

from autoconj.canonicalize import canonicalize
from autoconj.conjugacy import (
    complete_conditional, find_sufficient_statistic_nodes, marginalize,
    split_einsum_node, SupportTypes, statistic_representation,
    make_initializers, grad_namedtuple)
from autoconj.exponential_families import batch_dirichlet
from autoconj.tracers import (eval_expr, eval_node, one_hot, print_expr,
                              make_expr, GraphExpr, logdet)
from autoconj import log_probs

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy import special
from autograd.scipy import misc

from scipy import stats

from absl.testing import absltest


def _match_values(set_1, set_2, close_fn=np.allclose):
  """Checks that there's a match for every element of set_1 in set_2."""
  return all(any(close_fn(a, b) for b in set_2) for a in set_1)


def _perfect_match_values(set_1, set_2, close_fn=np.allclose):
  """Checks that there's a perfect matching between set_1 and set_2."""
  if len(set_1) == len(set_2):
    matches = np.array([[close_fn(a, b) for a in set_1] for b in set_2], int)
    return np.all(matches.sum(0) == 1) and np.all(matches.sum(1) == 1)
  return False


def _condition_and_marginalize(log_joint, argnum, support, *args):
  sub_args = args[:argnum] + args[argnum + 1:]

  marginalized = marginalize(log_joint, argnum, support, *args)
  marginalized_value = marginalized(*sub_args)

  conditional_factory = complete_conditional(log_joint, argnum, support, *args)
  conditional = conditional_factory(*sub_args)

  return conditional, marginalized_value


class ConjugacyTest(absltest.TestCase):

  def testFindSufficientStatisticNodes(self):

    def log_joint(x, y, matrix):
      # Linear in x: y^T x
      result = np.einsum('i,i->', x, y)
      # Quadratic form: x^T matrix x
      result += np.einsum('ij,i,j->', matrix, x, x)
      # Rank-1 quadratic form: (x**2)^T(y**2)
      result += np.einsum('i,i,j,j->', x, y, x, y)
      # Linear in log(x): y^T log(x)
      result += np.einsum('i,i->', y, np.log(x))
      # Linear in reciprocal(x): y^T reciprocal(x)
      result += np.einsum('i,i->', y, np.reciprocal(x))
      # More obscurely linear in log(x): y^T matrix log(x)
      result += np.einsum('i,ij,j->', y, matrix, np.log(x))
      # Linear in x * log(x): y^T (x * log(x))
      result += np.einsum('i,i->', y, x * np.log(x))
      return result

    n_dimensions = 5
    x = np.exp(np.random.randn(n_dimensions))
    y = np.random.randn(n_dimensions)
    matrix = np.random.randn(n_dimensions, n_dimensions)
    env = {'x': x, 'y': y, 'matrix': matrix}

    expr = make_expr(log_joint, x, y, matrix)
    expr = canonicalize(expr)
    sufficient_statistic_nodes = find_sufficient_statistic_nodes(expr, 'x')
    suff_stats = [eval_expr(GraphExpr(node, expr.free_vars), env)
                  for node in sufficient_statistic_nodes]
    correct_suff_stats = [x, x.dot(matrix.dot(x)), np.square(x.dot(y)),
                          np.log(x), np.reciprocal(x), y.dot(x * np.log(x))]
    self.assertTrue(_perfect_match_values(suff_stats, correct_suff_stats))

    expr = make_expr(log_joint, x, y, matrix)
    expr = canonicalize(expr)
    sufficient_statistic_nodes = find_sufficient_statistic_nodes(
        expr, 'x', split_einsums=True)
    suff_stats = [eval_expr(GraphExpr(node, expr.free_vars), env)
                  for node in sufficient_statistic_nodes]
    correct_suff_stats = [x, np.outer(x, x), x * x,
                          np.log(x), np.reciprocal(x), x * np.log(x)]
    self.assertTrue(_match_values(suff_stats, correct_suff_stats))

  def testSplitEinsumNode(self):
    n_dimensions = 5
    x = np.random.randn(n_dimensions)
    y = np.random.randn(n_dimensions)
    matrix = np.random.randn(n_dimensions, n_dimensions)

    env = {'x': x, 'y': y, 'matrix': matrix}

    args = (x, y)
    f = lambda x, y: np.einsum('i,i->', x, y)
    node = make_expr(f, *args)
    val = f(*args)
    potential_node, stat_node = split_einsum_node(node.expr_node, [0])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env), x))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    potential_node, stat_node = split_einsum_node(node.expr_node, [1])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env), y))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    potential_node, stat_node = split_einsum_node(node.expr_node, [0, 1])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env), x * y))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    args = (x, y)
    f = lambda x, y: np.einsum('i,i,i->', x, y, y)
    node = make_expr(f, *args)
    val = f(*args)
    potential_node, stat_node = split_einsum_node(node.expr_node, [1, 2])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env), y * y))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))
    potential_node, stat_node = split_einsum_node(node.expr_node, [0])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env), x))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    args = (x,)
    f = lambda x: np.einsum('i,i,i->', np.ones_like(x), x, x)
    node = make_expr(f, *args)
    val = f(*args)
    potential_node, stat_node = split_einsum_node(node.expr_node, [1, 2])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env), x * x))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    args = (matrix, x, y)
    f = lambda matrix, x, y: np.einsum('ij,i,j->', matrix, x, y)
    node = make_expr(f, *args)
    val = f(*args)
    potential_node, stat_node = split_einsum_node(node.expr_node, [1, 2])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env),
                                np.outer(x, y)))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))
    potential_node, stat_node = split_einsum_node(node.expr_node, [0])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env), matrix))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    args = (matrix, x, y)
    f = lambda matrix, x, y: np.einsum('i,j,ki,kj->', x, x, matrix, matrix)
    node = make_expr(f, *args)
    val = f(*args)
    potential_node, stat_node = split_einsum_node(node.expr_node, [2, 3])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env),
                                matrix[:, None, :] * matrix[:, :, None]))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))
    potential_node, stat_node = split_einsum_node(node.expr_node, [0, 1])
    self.assertTrue(np.allclose(eval_node(stat_node, node, env),
                                np.outer(x, x)))
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    args = (matrix, x, y)
    f = lambda matrix, x, y: np.einsum(',kj,j,ka,a->', -0.5, matrix, x,
                                       matrix, y)
    node = make_expr(f, *args)
    val = f(*args)

    potential_node, stat_node = split_einsum_node(node.expr_node, [2, 4], False)
    self.assertEqual(stat_node.args[0], 'j,a->ja')
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))

    potential_node, stat_node = split_einsum_node(node.expr_node, [0, 1, 3], False)
    self.assertEqual(stat_node.args[0], ',kj,ka->kja')
    self.assertTrue(np.allclose(eval_node(potential_node, node, env), val))


  def testConditionAndMarginalizeZeroMeanScalarNormal(self):
    def log_joint(x, precision):
      return np.einsum(',,,->', -0.5, precision, x, x)
    x = np.random.randn()
    precision = np.exp(np.random.randn())

    conditional, marginalized_value = _condition_and_marginalize(log_joint, 0,
                                                                 SupportTypes.REAL,
                                                                 x, precision)
    correct_marginalized_value = (-0.5 * np.log(precision)
                                  + 0.5 * np.log(2. * np.pi))

    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)
    self.assertEqual(0, conditional.args[0])
    self.assertEqual(1. / np.sqrt(precision), conditional.args[1])

  def testBatchDirichlet(self):
    alpha = np.ones(4)
    distribution_list = batch_dirichlet(alpha)
    self.assertTrue(np.allclose(alpha, distribution_list.alpha))

    alpha = np.ones([4, 3])
    distribution_list = batch_dirichlet(alpha)
    for i in range(alpha.shape[0]):
      self.assertTrue(np.allclose(alpha[i],
                                  distribution_list[i].item(0).alpha))

    alpha = np.ones([2, 4, 3])
    distribution_list = batch_dirichlet(alpha)
    for i in range(alpha.shape[0]):
      for j in range(alpha[i].shape[0]):
        self.assertTrue(np.allclose(alpha[i, j],
                                    distribution_list[i, j].item(0).alpha))

  def testConditionAndMarginalizeScalarNormal(self):
    def log_joint(x, mu, precision):
      quadratic_term = np.einsum(',,,->', -0.5, precision, x, x)
      linear_term = np.einsum(',,->', precision, x, mu)
      return quadratic_term + linear_term
    x = np.random.randn()
    mu = np.random.randn()
    precision = np.exp(np.random.randn())

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.REAL, x, mu,
                                   precision))
    correct_marginalized_value = (-0.5 * np.log(precision)
                                  + 0.5 * mu**2 * precision
                                  + 0.5 * np.log(2. * np.pi))
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertAlmostEqual(mu, conditional.args[0])
    self.assertAlmostEqual(1. / np.sqrt(precision), conditional.args[1])

  def testConditionAndMarginalizeZeroMeanNormal(self):
    def log_joint(x, precision):
      return np.einsum(',ij,i,j->', -0.5, precision, x, x)
    n_dimensions = 5
    x = np.random.randn(n_dimensions)
    precision = np.random.randn(n_dimensions, 2 * n_dimensions)
    precision = np.dot(precision, precision.T)

    conditional, marginalized_value = _condition_and_marginalize(log_joint, 0,
                                                                 SupportTypes.REAL,
                                                                 x, precision)
    correct_marginalized_value = (-0.5 * np.linalg.slogdet(precision)[1]
                                  + 0.5 * n_dimensions * np.log(2. * np.pi))

    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)
    self.assertTrue(np.allclose(np.zeros(n_dimensions), conditional.mean))
    self.assertTrue(np.allclose(np.linalg.inv(precision), conditional.cov))

  def testConditionAndMarginalizeDiagonalZeroMeanNormal(self):
    def log_joint_einsum(x, tau):
      return np.einsum(',i,i,i->', -0.5, tau, x, x)

    def log_joint_square(x, tau):
      return np.sum(-0.5 * tau * x ** 2)

    self._test_condition_and_marginalize_diagonal_zero_mean_normal(
        log_joint_einsum)
    self._test_condition_and_marginalize_diagonal_zero_mean_normal(
        log_joint_square)

  def _test_condition_and_marginalize_diagonal_zero_mean_normal(self,
                                                                log_joint):
    n_dimensions = 5
    x = np.random.randn(n_dimensions)
    tau = np.random.randn(n_dimensions) ** 2

    end_node = make_expr(log_joint, x, tau)
    end_node = canonicalize(end_node)

    conditional, marginalized_value = _condition_and_marginalize(
        log_joint, 0, SupportTypes.REAL, x, tau)
    correct_marginalized_value = (-0.5 * np.log(tau).sum()
                                  + 0.5 * n_dimensions * np.log(2. * np.pi))
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertTrue(np.allclose(np.zeros(n_dimensions), conditional.args[0]))
    self.assertTrue(np.allclose(1. / np.sqrt(tau), conditional.args[1]))

  def testConditionAndMarginalizeNormal(self):
    def log_joint(x, mu, precision):
      quadratic = np.einsum(',ij,i,j->', -0.5, precision, x, x)
      linear = np.einsum('ij,i,j->', precision, x, mu)
      return linear + quadratic - 3.
    n_dimensions = 5
    x = np.random.randn(n_dimensions)
    mu = np.random.randn(n_dimensions)
    precision = np.random.randn(n_dimensions, 2 * n_dimensions)
    precision = np.dot(precision, precision.T)

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.REAL, x, mu,
                                   precision))

    correct_marginalized_value = (
        -0.5 * np.linalg.slogdet(precision)[1]
        + 0.5 * np.einsum('ij,i,j->', precision, mu, mu)
        + 0.5 * n_dimensions * np.log(2. * np.pi)
        - 3.)
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertTrue(np.allclose(mu, conditional.mean))
    self.assertTrue(np.allclose(np.linalg.inv(precision), conditional.cov))

  def testConditionAndMarginalizeDiagonalNormal(self):
    def log_joint_einsum(x, mu, tau):
      quadratic = np.einsum(',i,i,i->', -0.5, tau, x, x)
      linear = np.einsum('i,i,i->', tau, x, mu)
      return linear + quadratic - 3.

    def log_joint_square(x, mu, tau):
      quadratic = np.sum(-0.5 * tau * x ** 2)
      linear = np.einsum('i,i,i->', tau, x, mu)
      return linear + quadratic - 3.

    self._test_condition_and_marginalize_diagonal_normal(log_joint_einsum)
    self._test_condition_and_marginalize_diagonal_normal(log_joint_square)

  def _test_condition_and_marginalize_diagonal_normal(self, log_joint):
    n_dimensions = 5
    x = np.random.randn(n_dimensions)
    mu = np.random.randn(n_dimensions)
    tau = np.random.randn(n_dimensions) ** 2

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.REAL, x, mu,
                                   tau))

    correct_marginalized_value = (-0.5 * np.log(tau).sum()
                                  + 0.5 * np.einsum('i,i,i->', tau, mu, mu)
                                  + 0.5 * n_dimensions * np.log(2. * np.pi)
                                  - 3.)
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertTrue(np.allclose(mu, conditional.args[0]))
    self.assertTrue(np.allclose(1. / np.sqrt(tau), conditional.args[1]))

  def testConditionAndMarginalizeGamma(self):
    def log_joint(x, a, b):
      return np.sum((a - 1) * np.log(x) - b * x)
    a = np.random.gamma(1., 1., [3, 4])
    b = np.random.gamma(1., 1., 4)
    x = np.random.gamma(1., 1., [3, 4])

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.NONNEGATIVE,
                                   x, a, b))

    correct_marginalized_value = np.sum(-a * np.log(b) + special.gammaln(a))
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertTrue(np.allclose(a, conditional.args[0]))
    self.assertTrue(np.allclose(1. / b, conditional.args[2]))

  def testConditionAndMarginalizeBeta(self):
    def log_joint(x, a, b):
      return np.sum((a - 1) * np.log(x) + (b - 1) * np.log1p(-x))
    a = np.random.gamma(1., 1., [3, 4])
    b = np.random.gamma(1., 1., 4)
    x = np.random.gamma(1., 1., [3, 4])

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.UNIT_INTERVAL,
                                   x, a, b))

    correct_marginalized_value = (special.gammaln(a) + special.gammaln(b) -
                                  special.gammaln(a + b)).sum()
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertTrue(np.allclose(a, conditional.args[0]))
    self.assertTrue(np.allclose(b, conditional.args[1]))

  def testConditionAndMarginalizeDirichlet(self):
    def log_joint(x, alpha):
      return np.sum((alpha - 1) * np.log(x))
    alpha = np.random.gamma(1., 1., [3, 4])
    x = np.random.gamma(alpha, 1.)
    x /= x.sum(-1, keepdims=True)

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.SIMPLEX,
                                   x, alpha))

    correct_marginalized_value = (special.gammaln(alpha).sum() -
                                  special.gammaln(np.sum(alpha, 1)).sum())
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    for i in range(alpha.shape[0]):
      self.assertTrue(np.allclose(alpha[i], conditional[i].item(0).alpha))

  def testConditionAndMarginalizeBernoulli(self):
    def log_joint(x, logits):
      return np.sum(x * logits)
    p = np.random.beta(2., 2., [3, 4])
    logit_p = np.log(p) - np.log1p(-p)
    # TODO(mhoffman): Without the cast this gives wrong answers due to autograd
    # casts. This is scary.
    x = (np.random.uniform(size=(8,) + p.shape) < p).astype(np.float32)

    conditional, marginalized_value = _condition_and_marginalize(
        log_joint, 0, SupportTypes.BINARY, x, logit_p)

    correct_marginalized_value = np.sum(-x.shape[0] * np.log1p(-p))
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value,
                           places=4)

    self.assertTrue(np.allclose(p, conditional.args[0]))

  def testConditionAndMarginalizeCategorical(self):
    np.random.seed(0)
    vocab_size = 23
    def log_joint(x, probs):
      one_hot_x = one_hot(x, vocab_size)
      return np.sum(np.dot(one_hot_x, np.log(probs)))
    n_examples = 13
    alpha = 1.3
    probs = np.random.gamma(alpha, 1., vocab_size)
    probs /= probs.sum()
    x = np.random.choice(np.arange(vocab_size), n_examples, p=probs)

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.INTEGER,
                                   x, probs))

    self.assertTrue(np.allclose(conditional.p.sum(1), 1))
    self.assertTrue(np.allclose(conditional.p, np.ones([n_examples, 1]) * probs))
    self.assertAlmostEqual(0., marginalized_value, places=5)

    logit_probs = np.random.randn(vocab_size)
    probs = np.exp(logit_probs)
    probs /= probs.sum()

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.INTEGER,
                                   x, np.exp(logit_probs)))

    correct_marginalized_value = np.log(np.sum(np.exp(logit_probs)))
    self.assertAlmostEqual(n_examples * correct_marginalized_value,
                           marginalized_value, places=4)
    self.assertTrue(np.allclose(conditional.p, np.ones([n_examples, 1]) * probs,
                                rtol=1e-5))

  def testGammaPoisson(self):
    def log_joint(x, y, a, b):
      log_prior = log_probs.gamma_gen_log_prob(x, a, b)
      log_likelihood = np.sum(-special.gammaln(y + 1) + y * np.log(x) - x)
      return log_prior + log_likelihood
    n_examples = 10
    a = 2.3
    b = 3.
    x = np.random.gamma(a, 1. / b)
    y = np.random.poisson(x, n_examples)

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.NONNEGATIVE,
                                   x, y, a, b))

    new_a = a + y.sum()
    new_b = b + n_examples
    correct_marginalized_value = (
        a * np.log(b) - special.gammaln(a) -
        new_a * np.log(new_b) + special.gammaln(new_a) -
        special.gammaln(y + 1).sum())
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertEqual(new_a, conditional.args[0])
    self.assertAlmostEqual(new_b, 1. / conditional.args[2])

  def testGammaGamma(self):
    def log_joint(x, y, a, b):
      log_prior = log_probs.gamma_gen_log_prob(x, a, b)
      log_likelihood = np.sum(log_probs.gamma_gen_log_prob(y, a, a * x))
      return log_prior + log_likelihood
    n_examples = 10
    a = 2.3
    b = 3.
    x = np.random.gamma(a, 1. / b)
    y = np.random.gamma(a, 1. / x, n_examples)

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.NONNEGATIVE,
                                   x, y, a, b))

    new_a = a + a * n_examples
    new_b = b + a * y.sum()
    correct_marginalized_value = (
        a * np.log(b) - special.gammaln(a) -
        new_a * np.log(new_b) + special.gammaln(new_a) +
        np.sum((a - 1) * np.log(y) - special.gammaln(a) + a * np.log(a)))
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

    self.assertAlmostEqual(new_a, conditional.args[0])
    self.assertAlmostEqual(new_b, 1. / conditional.args[2])

  def testGammaNormalScaleParameter(self):
    def log_joint(x, precision, a, b):
      log_p_precision = log_probs.gamma_gen_log_prob(precision, a, b)
      log_p_x = log_probs.norm_gen_log_prob(x, 0., 1. / np.sqrt(precision))
      return log_p_precision + log_p_x
    n_examples = 10
    a = 2.3
    b = 3.
    precision = np.random.gamma(a, 1. / b)
    x = np.random.normal(0., 1. / np.sqrt(precision), n_examples)

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 1, SupportTypes.NONNEGATIVE,
                                   x, precision, a, b))
    new_a = a + n_examples / 2
    new_b = b + (x ** 2).sum() / 2
    self.assertAlmostEqual(new_a, conditional.args[0])
    self.assertAlmostEqual(new_b, 1. / conditional.args[2])
    correct_marginalized_value = (
        a * np.log(b) - special.gammaln(a) -
        new_a * np.log(new_b) + special.gammaln(new_a) -
        0.5 * n_examples * np.log(2 * np.pi))
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)

  # TODO(mhoffman): This log_joint takes way too long to canonicalize.
  def testBetaBernoulli(self):
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
    self.assertAlmostEqual(marginalized_value, correct_marginalized_value,
                           places=4)

    self.assertTrue(np.allclose(new_a, conditional.args[0]))
    self.assertTrue(np.allclose(new_b, conditional.args[1]))

  def testDirichletCategorical(self):
    def log_joint(p, x, alpha):
      log_prior = np.sum((alpha - 1) * np.log(p))
      log_prior += -special.gammaln(alpha).sum() + special.gammaln(alpha.sum())
      # TODO(mhoffman): We should make it possible to only use one-hot
      # when necessary.
      one_hot_x = one_hot(x, alpha.shape[0])
      log_likelihood = np.sum(np.dot(one_hot_x, np.log(p)))
      return log_prior + log_likelihood
    vocab_size = 5
    n_examples = 11
    alpha = 1.3 * np.ones(vocab_size)
    p = np.random.gamma(alpha, 1.)
    p /= p.sum(-1, keepdims=True)
    x = np.random.choice(np.arange(vocab_size), n_examples, p=p)

    conditional, marginalized_value = (
        _condition_and_marginalize(log_joint, 0, SupportTypes.SIMPLEX,
                                   p, x, alpha))

    new_alpha = alpha + np.histogram(x, np.arange(vocab_size + 1))[0]
    correct_marginalized_value = (
        -special.gammaln(alpha).sum() + special.gammaln(alpha.sum()) +
        special.gammaln(new_alpha).sum() - special.gammaln(new_alpha.sum()))
    self.assertAlmostEqual(correct_marginalized_value, marginalized_value)
    self.assertTrue(np.allclose(new_alpha, conditional.alpha))

  def testLinearRegression(self):
    def log_joint(X, beta, y):
      predictions = np.einsum('ij,j->i', X, beta)
      errors = y - predictions
      log_prior = np.einsum('i,i,i->', -0.5 * np.ones_like(beta), beta, beta)
      log_likelihood = np.einsum(',k,k->', -0.5, errors, errors)
      return log_prior + log_likelihood
    n_examples = 10
    n_predictors = 2
    X = np.random.randn(n_examples, n_predictors)
    beta = np.random.randn(n_predictors)
    y = np.random.randn(n_examples)
    graph = make_expr(log_joint, X, beta, y)
    graph = canonicalize(graph)

    args = graph.free_vars.keys()
    sufficient_statistic_nodes = find_sufficient_statistic_nodes(graph, args[1])
    sufficient_statistics = [eval_node(node, graph.free_vars,
                                       {'X': X, 'beta': beta, 'y': y})
                             for node in sufficient_statistic_nodes]
    correct_sufficient_statistics = [
        -0.5 * beta.dot(beta), beta,
        -0.5 * np.einsum('ij,ik,j,k', X, X, beta, beta)
    ]
    self.assertTrue(_match_values(sufficient_statistics,
                                  correct_sufficient_statistics))

    new_log_joint, _, stats_funs, _ = (
        statistic_representation(log_joint, (X, beta, y),
                               (SupportTypes.REAL,), (1,)))
    beta_stat_fun = stats_funs[0]
    beta_natparam = grad_namedtuple(new_log_joint, 1)(X, beta_stat_fun(beta), y)
    correct_beta_natparam = (-0.5 * X.T.dot(X), y.dot(X),
                             -0.5 * np.ones(n_predictors))
    self.assertTrue(_match_values(beta_natparam, correct_beta_natparam))

    conditional_factory = complete_conditional(log_joint, 1, SupportTypes.REAL,
                                               X, beta, y)
    conditional = conditional_factory(X, y)
    true_cov = np.linalg.inv(X.T.dot(X) + np.eye(n_predictors))
    true_mean = true_cov.dot(y.dot(X))
    self.assertTrue(np.allclose(true_cov, conditional.cov))
    self.assertTrue(np.allclose(true_mean, conditional.mean))

  def testMixtureOfGaussians(self):
    def log_joint(x, pi, z, mu, sigma_sq, alpha, sigma_sq_mu):
      log_p_pi = log_probs.dirichlet_gen_log_prob(pi, alpha)
      log_p_mu = log_probs.norm_gen_log_prob(mu, 0, np.sqrt(sigma_sq_mu))

      z_one_hot = one_hot(z, len(pi))
      log_p_z = np.einsum('ij,j->', z_one_hot, np.log(pi))

      mu_z = np.einsum('ij,jk->ik', z_one_hot, mu)
      log_p_x = log_probs.norm_gen_log_prob(x, mu_z, np.sqrt(sigma_sq))

      return log_p_pi + log_p_z + log_p_mu + log_p_x

    n_clusters = 5
    n_dimensions = 2
    n_observations = 200

    alpha = 3.3 * np.ones(n_clusters)
    sigma_sq_mu = 1.5 ** 2
    sigma_sq = 0.5 ** 2

    np.random.seed(10001)

    pi = np.random.gamma(alpha)
    pi /= pi.sum()
    mu = np.random.normal(0, np.sqrt(sigma_sq_mu), [n_clusters, n_dimensions])
    z = np.random.choice(np.arange(n_clusters), size=n_observations, p=pi)
    x = np.random.normal(mu[z, :], sigma_sq)

    pi_est = np.ones(n_clusters) / n_clusters
    z_est = np.random.choice(np.arange(n_clusters), size=n_observations,
                             p=pi_est)
    mu_est = np.random.normal(0., 0.01, [n_clusters, n_dimensions])

    all_args = [x, pi_est, z_est, mu_est, sigma_sq, alpha, sigma_sq_mu]
    pi_posterior_args = all_args[:1] + all_args[2:]
    z_posterior_args = all_args[:2] + all_args[3:]
    mu_posterior_args = all_args[:3] + all_args[4:]

    pi_posterior = complete_conditional(log_joint, 1, SupportTypes.SIMPLEX,
                                        *all_args)
    z_posterior = complete_conditional(log_joint, 2, SupportTypes.INTEGER,
                                       *all_args)
    mu_posterior = complete_conditional(log_joint, 3, SupportTypes.REAL,
                                        *all_args)

    self.assertTrue(np.allclose(
        pi_posterior(*pi_posterior_args).alpha,
        alpha + np.histogram(z_est, np.arange(n_clusters+1))[0]))

    correct_z_logits = -0.5 / sigma_sq * np.square(x[:, :, None] -
                                                   mu_est.T[None, :, :]).sum(1)
    correct_z_logits += np.log(pi_est)
    correct_z_posterior = np.exp(correct_z_logits -
                                 misc.logsumexp(correct_z_logits, 1,
                                                keepdims=True))
    self.assertTrue(np.allclose(correct_z_posterior,
                                z_posterior(*z_posterior_args).p))

    correct_mu_posterior_mean = np.zeros_like(mu_est)
    correct_mu_posterior_var = np.zeros_like(mu_est)
    for k in range(n_clusters):
      n_k = (z_est == k).sum()
      correct_mu_posterior_var[k] = 1. / (1. / sigma_sq_mu + n_k / sigma_sq)
      correct_mu_posterior_mean[k] = (
          x[z_est == k].sum(0) / sigma_sq * correct_mu_posterior_var[k])
    mu_posterior_val = mu_posterior(*mu_posterior_args)
    self.assertTrue(np.allclose(correct_mu_posterior_mean,
                                mu_posterior_val.args[0]))
    self.assertTrue(np.allclose(correct_mu_posterior_var,
                                mu_posterior_val.args[1] ** 2))

  def testTwoGaussians(self):
    def log_joint(x1, x2):
      log_p_x1 = -0.5 * x1 * x1
      x_diff = x2 - x1
      log_p_x2 = -0.5 * x_diff * x_diff
      return log_p_x1 + log_p_x2

    x1 = np.random.randn()
    x2 = x1 + np.random.randn()
    all_args = [x1, x2]

    marginal_p_x2 = marginalize(log_joint, 0, SupportTypes.REAL, *all_args)
    correct_marginalized_value = (
        -0.25 * x2 * x2 - 0.5 * np.log(2.) + 0.5 * np.log(2. * np.pi))
    self.assertAlmostEqual(correct_marginalized_value, marginal_p_x2(x2))

    x2_conditional = complete_conditional(marginal_p_x2, 0, SupportTypes.REAL,
                                          x2)()
    self.assertAlmostEqual(x2_conditional.args[0], 0.)
    self.assertAlmostEqual(x2_conditional.args[1] ** 2, 2.)

  def testFactorAnalysis(self):
    def log_joint(x, w, epsilon, tau, alpha, beta):
      log_p_epsilon = log_probs.norm_gen_log_prob(epsilon, 0, 1)
      log_p_w = log_probs.norm_gen_log_prob(w, 0, 1)
      log_p_tau = log_probs.gamma_gen_log_prob(tau, alpha, beta)
      # TODO(mhoffman): The transposed version below should work.
      # log_p_x = log_probs.norm_gen_log_prob(x, np.dot(epsilon, w), 1. / np.sqrt(tau))
      log_p_x = log_probs.norm_gen_log_prob(x, np.einsum('ik,jk->ij', epsilon, w),
                                            1. / np.sqrt(tau))
      return log_p_epsilon + log_p_w + log_p_tau + log_p_x

    n_examples = 20
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
    for d in range(D):
      self.assertTrue(np.allclose(conditional[d].cov, true_cov))
      self.assertTrue(np.allclose(conditional[d].mean, true_mean[d]))

    epsilon_conditional_factory = complete_conditional(log_joint, 2,
                                                       SupportTypes.REAL,
                                                       *all_args)
    conditional = epsilon_conditional_factory(x, w, tau, alpha, beta)
    true_cov = np.linalg.inv(tau * np.einsum('dk,dl->kl', w, w) + np.eye(K))
    true_mean = tau * np.einsum('dk,nd,kl->nl', w, x, true_cov)
    for n in range(n_examples):
      self.assertTrue(np.allclose(conditional[n].cov, true_cov))
      self.assertTrue(np.allclose(conditional[n].mean, true_mean[n]))

    tau_conditional_factory = complete_conditional(log_joint, 3,
                                                   SupportTypes.NONNEGATIVE,
                                                   *all_args)
    conditional = tau_conditional_factory(x, w, epsilon, alpha, beta)
    true_a = alpha + 0.5 * n_examples * D
    true_b = beta + 0.5 * np.sum(np.square(x - epsilon.dot(w.T)))
    self.assertAlmostEqual(true_a, conditional.args[0])
    self.assertAlmostEqual(true_b, 1. / conditional.args[2])

  def testStatisticRepresentation(self):

    A = np.array([[1., 0], [0., 1.], [1., 1.]])
    Sigma = 2 * np.eye(3)
    z = npr.randn(2)
    x = np.array([1000., -1000., 0.])

    def log_joint(z, x):
      log_prior = -1./2 * np.dot(z, z)
      centered = x - np.dot(A, z)
      log_like = (-1./2 * np.dot(centered, np.dot(np.linalg.inv(Sigma), centered))
          - 1./2 * logdet(Sigma))
      return log_prior + log_like

    neg_energy, normalizers, stats_funs, samplers = (
        statistic_representation(log_joint, (z, x),
                                 (SupportTypes.REAL, SupportTypes.REAL)))
    initializers, _ = make_initializers((z,x), neg_energy, normalizers,
                                        stats_funs)

    # just check that these don't crash
    natparams = [initializer() for initializer in initializers]
    neg_energy(*natparams)
    [sampler(natparam).rvs() for sampler, natparam in zip(samplers, natparams)]

    expected_post_mu = A.T.dot(x / 2.)
    computed_post_mu = grad_namedtuple(normalizers[0])(initializers[0]()).x
    self.assertTrue(np.allclose(expected_post_mu, computed_post_mu))


if __name__ == '__main__':
  absltest.main()
