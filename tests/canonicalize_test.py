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

from autoconj.canonicalize import canonicalize, is_canonical, simplify_sweep
from autoconj.tracers import add_n, eval_expr, GraphExpr, make_expr, print_expr

import autograd.extend as ag_extend
import autograd.numpy as np


class CanonicalizeTest(absltest.TestCase):

  def testDummySimplify(self):
    """Ensures simplify_sweep() works with a dummy simplification."""
    expr = make_expr(lambda x, y: x * (y * 3. + y), 3., 4.)
    self.assertFalse(simplify_sweep(expr, lambda node: False))

  def testEinsumAddSubSimplify(self):
    # TODO(mhoffman): Think about broadcasting. We need to support `x - 2.0`.

    def test_fun(x):
      return np.einsum('i->', x + np.full(x.shape, 2.0))

    expr = make_expr(test_fun, np.ones(3))
    test_x = np.full(3, 0.5)
    correct_value = eval_expr(expr, {'x': test_x})
    expr = canonicalize(expr)
    self.assertIsInstance(expr, GraphExpr)
    self.assertEqual(expr.expr_node.fun, add_n)
    self.assertEqual(expr.expr_node.parents[0].fun.__name__, 'einsum')
    new_value = eval_expr(expr, {'x': test_x})
    self.assertEqual(correct_value, new_value)

  def testCanonicalize(self):

    def mahalanobis_distance(x, y, matrix):
      x_minus_y = x - y
      return np.einsum('i,j,ij->', x_minus_y, x_minus_y, matrix)

    x = np.array([1.3, 3.6])
    y = np.array([2.3, -1.2])
    matrix = np.arange(4).reshape([2, 2])
    expr = make_expr(mahalanobis_distance, x, y, matrix)
    self.assertFalse(is_canonical(expr))
    correct_value = eval_expr(expr, {'x': x, 'y': y, 'matrix': matrix})
    expr = canonicalize(expr)
    self.assertTrue(is_canonical(expr))
    new_value = eval_expr(expr, {'x': x, 'y': y, 'matrix': matrix})
    self.assertAlmostEqual(correct_value, new_value)

  def testEinsumCompose(self):
    def Xbeta_squared(X, beta):
      Xbeta = np.einsum('ij,j->i', X, beta)
      Xbeta2 = np.einsum('lm,m->l', X, beta)
      return np.einsum('k,k->', Xbeta, Xbeta)
    n_examples = 10
    n_predictors = 2
    X = np.random.randn(n_examples, n_predictors)
    beta = np.random.randn(n_predictors)
    expr = make_expr(Xbeta_squared, X, beta)
    correct_value = eval_expr(expr, {'X': X, 'beta': beta})
    self.assertFalse(is_canonical(expr))
    expr = canonicalize(expr)
    new_value = eval_expr(expr, {'X': X, 'beta': beta})
    self.assertAlmostEqual(correct_value, new_value)
    self.assertIsInstance(expr, GraphExpr)
    self.assertEqual(expr.expr_node.fun, np.einsum)
    self.assertTrue(is_canonical(expr))

  def testLinearRegression(self):
    def squared_loss(X, beta, y):
      predictions = np.einsum('ij,j->i', X, beta)
      errors = y - predictions
      return np.einsum('k,k->', errors, errors)
    n_examples = 10
    n_predictors = 2
    X = np.random.randn(n_examples, n_predictors)
    beta = np.random.randn(n_predictors)
    y = np.random.randn(n_examples)
    expr = make_expr(squared_loss, X, beta, y)
    correct_value = eval_expr(expr, {'X': X, 'beta': beta, 'y':y})
    self.assertFalse(is_canonical(expr))
    expr = canonicalize(expr)
    self.assertTrue(is_canonical(expr))
    new_value = eval_expr(expr, {'X': X, 'beta': beta, 'y':y})
    self.assertAlmostEqual(correct_value, new_value)

  def testReciprocalToPow(self):
    def fun(x):
      return np.reciprocal(x)
    expr = make_expr(fun, 3.)
    expr = canonicalize(expr)
    self.assertIsInstance(expr, GraphExpr)
    self.assertEqual(expr.expr_node.fun, np.power)
    self.assertEqual(eval_expr(expr, {'x': 3.}), fun(3.))

  def testSquareToPow(self):
    def fun(x):
      return np.square(x)
    expr = make_expr(fun, 3.)
    expr = canonicalize(expr)
    self.assertIsInstance(expr, GraphExpr)
    self.assertEqual(expr.expr_node.fun, np.power)
    self.assertEqual(eval_expr(expr, {'x': 3.}), fun(3.))

  def testSqrtToPow(self):
    def fun(x):
      return np.sqrt(x)
    expr = make_expr(fun, 3.)
    expr = canonicalize(expr)
    self.assertIsInstance(expr, GraphExpr)
    self.assertEqual(expr.expr_node.fun, np.power)
    self.assertEqual(eval_expr(expr, {'x': 3.}), fun(3.))

if __name__ == '__main__':
  absltest.main()
