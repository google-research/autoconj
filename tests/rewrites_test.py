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

import inspect

from absl.testing import absltest

import autograd.numpy as np
import autograd.numpy.random as npr
import numpy.testing

from autoconj import rewrites
from autoconj import tracers


class NumericalTestCase(absltest.TestCase):

  def assertArraysAllClose(self, x, y, err_msg='', atol=1e-8, rtol=1e-5):
    numpy.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=err_msg)

  def assertAllClose(self, x, y, err_msg='', atol=1e-8, rtol=1e-5):
    if isinstance(x, (tuple, list)):
      self.assertEqual(type(x), type(y))
      self.assertEqual(len(x), len(y))
      for elt_a, elt_b in zip(x, y):
        self.assertAllClose(elt_a, elt_b, err_msg, atol, rtol)
    self.assertArraysAllClose(x, y, err_msg, atol, rtol)


class RewritesTest(NumericalTestCase):

  def _rewriter_test_helper(self, fun, rewrite_rule, *args, **kwargs):
    expr = kwargs.get('expr') or tracers.make_expr(fun, *args)
    self.assertIsInstance(expr, tracers.GraphExpr)

    env = dict(zip(inspect.getargspec(fun).args, args))
    self.assertAllClose(fun(*args), tracers.eval_expr(expr, env))

    rewriter = rewrites.make_rewriter(rewrite_rule)
    rewrite_node = kwargs.get('rewrite_node', expr.expr_node)
    rewriter(rewrite_node)  # modifies expr in-place

    self.assertAllClose(fun(*args), tracers.eval_expr(expr, env))
    return tracers.remake_expr(expr)  # constant folding

  def _eager_rewriter_test_helper(self, fun, rewriter, *args, **kwargs):
    expr = kwargs.get('expr') or tracers.make_expr(fun, *args)
    self.assertIsInstance(expr, tracers.GraphExpr)

    env = dict(zip(inspect.getargspec(fun).args, args))
    self.assertAllClose(fun(*args), tracers.eval_expr(expr, env))

    rewrite_node = kwargs.get('rewrite_node', expr.expr_node)
    expr = tracers.remake_expr(expr, {rewrite_node.fun: rewriter})

    self.assertAllClose(fun(*args), tracers.eval_expr(expr, env))
    return tracers.remake_expr(expr)  # constant folding

  def testDotRewriter(self):

    def fun(x, y):
      return np.dot(x, y)

    x = npr.randn(4, 3)
    y = npr.randn(3)

    expr = tracers.make_expr(fun, x, y)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'dot')

    expr = self._eager_rewriter_test_helper(fun, rewrites.dot_as_einsum, x, y)
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

  def testMultiplyRewriter(self):

    def fun(x, y):
      return x * y

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_multiply,
                                            npr.randn(4, 3), npr.randn(3))
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_multiply,
                                            npr.randn(4, 3), npr.randn(4, 1))
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_multiply,
                                            npr.randn(4, 3, 2),
                                            npr.randn(4, 1, 2))
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_multiply,
                                            npr.randn(4, 3, 2),
                                            npr.randn(4, 1, 1))
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_multiply,
                                            npr.randn(1, 1, 3),
                                            npr.randn(4, 1, 3))
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

  def testDivideToMultiply(self):

    def fun(x, y):
      return x / y

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_divide,
                                            1.3, 4.7)
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_divide,
                                            1.3 * np.ones([3, 4, 1]),
                                            4.7 * np.ones([3, 4, 5]))
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._eager_rewriter_test_helper(lambda y: np.ones([3, 4, 5]) / y,
                                            rewrites.maybe_divide,
                                            4.7 * np.ones([3, 4, 5]))
    self.assertEqual(expr.expr_node.fun.__name__, 'power')

  def testPowerRewriter(self):
    x = npr.randn(10)

    fun = lambda x: x**0
    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_power, x)
    self.assertIsInstance(expr, tracers.ConstExpr)
    self.assertEqual(expr.val, 1)

    fun = lambda x: x**1
    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_power, x)
    self.assertEqual(expr.expr_node.fun.__name__, 'env_lookup')

    fun = lambda x: x**4
    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_power, x)
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    fun = lambda x: x**-1
    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_power, x)
    self.assertEqual(expr.expr_node.fun.__name__, 'power')

  def testAddRewriter(self):
    x = npr.randn(4, 3)
    y = npr.randn(3)
    z = npr.randn(1, 3)

    expr = self._rewriter_test_helper(lambda x, y, z: x + (y + z),
                                      rewrites.replace_add, x, y, z)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')

    expr = self._rewriter_test_helper(lambda x, y, z: (x + y) + z,
                                      rewrites.replace_add, x, y, z)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')

  def testAddNRewriter(self):
    x = npr.randn(4, 3)
    y = npr.randn(3)
    z = npr.randn(1, 3)

    expr = self._rewriter_test_helper(lambda x, y, z: x + tracers.add_n(y, z),
                                      rewrites.replace_add_addn, x, y, z)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')

    expr = self._rewriter_test_helper(lambda x, y, z: tracers.add_n(x, y) + z,
                                      rewrites.replace_add_addn, x, y, z)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')

    expr = self._rewriter_test_helper(
        lambda x, y, z: tracers.add_n(tracers.add_n(x, y), z),
        rewrites.replace_addn_addn, x, y, z)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')
    self.assertTrue(all([parent.fun.__name__ == 'env_lookup'
                         for parent in expr.expr_node.parents]))

    expr = self._rewriter_test_helper(
        lambda x, y, z: tracers.add_n(x, tracers.add_n(y, z), z),
        rewrites.replace_addn_addn, x, y, z)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')
    self.assertTrue(all([parent.fun.__name__ == 'env_lookup'
                         for parent in expr.expr_node.parents]))

  def testDuplicatedAddNRewriter(self):
    x = npr.randn(4, 3)
    y = npr.randn(3)
    z = npr.randn(1, 3)

    expr = self._rewriter_test_helper(
        lambda x, y, z: tracers.add_n(x, y, z, y),
        rewrites.replace_duplicated_addn, x, y, z)
    self.assertIsInstance(expr, tracers.GraphExpr)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')
    self.assertEqual(len(expr.expr_node.parents), 3)
    self.assertTrue(all([parent.fun.__name__ in ('multiply', 'env_lookup')
                         for parent in expr.expr_node.parents]))

  def testSumRewriter(self):

    def fun(x):
      return np.sum(x, 1)

    x = npr.randn(4, 3)

    expr = tracers.make_expr(fun, x)
    self.assertEqual(expr.expr_node.fun, np.sum)

    expr = self._rewriter_test_helper(fun, rewrites.replace_sum, x)
    self.assertEqual(expr.expr_node.fun, np.einsum)

  def testSumRewriterTuple(self):

    def fun(x):
      return np.sum(x, axis=(0, 1))

    x = npr.randn(4, 3)

    expr = tracers.make_expr(fun, x)
    self.assertEqual(expr.expr_node.fun, np.sum)

    expr = self._rewriter_test_helper(fun, rewrites.replace_sum, x)
    self.assertEqual(expr.expr_node.fun, np.einsum)

  def testSumRewriterKwarg(self):

    def fun(x):
      return np.sum(x, axis=1)

    x = npr.randn(4, 3)

    expr = tracers.make_expr(fun, x)
    self.assertEqual(expr.expr_node.fun, np.sum)

    expr = self._rewriter_test_helper(fun, rewrites.replace_sum, x)
    self.assertEqual(expr.expr_node.fun, np.einsum)

  def testFullSumRewriter(self):

    def fun(x):
      return np.sum(x)

    x = npr.randn(4, 3)

    expr = tracers.make_expr(fun, x)
    self.assertEqual(expr.expr_node.fun, np.sum)

    expr = self._rewriter_test_helper(fun, rewrites.replace_sum, x)
    self.assertEqual(expr.expr_node.fun, np.einsum)

    def fun(x):
      return np.sum(x, None)

    expr = tracers.make_expr(fun, x)
    self.assertEqual(expr.expr_node.fun, np.sum)

    expr = self._rewriter_test_helper(fun, rewrites.replace_sum, x)
    self.assertEqual(expr.expr_node.fun, np.einsum)

  def testSwapaxesToEinsum(self):
    x = np.arange(9).reshape([3, 3])
    self.assertTrue((np.swapaxes(x, 0, 1) == rewrites.swapaxes(x, 0, 1)).all())

  def testSubtractToAdd(self):

    def fun(x, y):
      return x - y

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_subtract,
                                            1.3, 4.7)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_subtract,
                                            1.3 * np.ones([3, 4, 1]),
                                            4.7 * np.ones([3, 4, 5]))
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')

  def testEinsumDistributeRewriter(self):

    def fun(x, y, z):
      return np.einsum('ij,j->i', x, tracers.add_n(y, z))

    x = npr.randn(4, 3)
    y = npr.randn(3)
    z = npr.randn(3)

    expr = tracers.make_expr(fun, x, y, z)
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._rewriter_test_helper(fun, rewrites.distribute_einsum,
                                      x, y, z)
    self.assertEqual(expr.expr_node.fun.__name__, 'add_n')

  def testEinsumTransposeRewriter(self):

    def fun(x, y):
      return np.einsum('ij,j->i', x.T, y)

    x = npr.randn(4, 3)
    y = npr.randn(4)

    expr = tracers.make_expr(fun, x, y)
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._rewriter_test_helper(fun, rewrites.transpose_inside_einsum,
                                      x, y)
    self.assertFalse('transpose' in tracers.print_expr(expr))

  def testEinsumTransposeRewriter2(self):

    def fun(x, y):
      return np.einsum('ij,j,kj->ik', x.T, y, x.T)

    x = npr.randn(4, 3)
    y = npr.randn(4)

    expr = tracers.make_expr(fun, x, y)
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')

    expr = self._rewriter_test_helper(fun, rewrites.transpose_inside_einsum,
                                      x, y)
    expr = self._rewriter_test_helper(fun, rewrites.transpose_inside_einsum,
                                      x, y, expr=expr)
    self.assertFalse('transpose' in tracers.print_expr(expr))

  def testEinsumCompositionRewriter(self):

    def fun(x, y, z):
      return np.einsum('ij,jk->i', x, np.einsum('ija,ijk->ka', y, z))

    x = npr.randn(4, 3)
    y = npr.randn(5, 4, 2)
    z = npr.randn(5, 4, 3)

    expr = tracers.make_expr(fun, x, y, z)
    self.assertEqual(expr.expr_node.fun.__name__, 'einsum')
    self.assertEqual(expr.expr_node.parents[1].fun.__name__, 'einsum')

    expr = self._rewriter_test_helper(fun, rewrites.combine_einsum_compositions,
                                      x, y, z)
    self.assertNotEqual(expr.expr_node.parents[1].fun.__name__, 'einsum')

  def testLogEinsumRewriter(self):

    # TODO(matthewjmackay): fails on example below where axes are transposed
#     def fun(x, y):
#       return np.log(np.einsum('ji,ij->ij', x, y))
#
#     x = np.exp(npr.randn(3, 4))
#     y = np.exp(npr.randn(4, 3))
#     z = np.exp(npr.randn(4))

    def fun(x, y):
      return np.log(np.einsum('ij,ij->ij', x, y))

    x = np.exp(npr.randn(4, 3))
    y = np.exp(npr.randn(4, 3))
    z = np.exp(npr.randn(4))

    expr = tracers.make_expr(fun, x, y)
    self.assertEqual(expr.expr_node.fun.__name__, 'log')
    self.assertEqual(expr.expr_node.parents[0].fun.__name__, 'einsum')

    expr = self._rewriter_test_helper(fun, rewrites.replace_log_einsum,
                                      x, y)
    self.assertEqual(expr.expr_node.fun.__name__, 'add')
    self.assertEqual(expr.expr_node.parents[0].fun.__name__, 'log')
    self.assertEqual(expr.expr_node.parents[1].fun.__name__, 'log')

  def testMultiplyDistribute(self):

    def fun(x, y, z):
      return x * tracers.add_n(y, z)

    x = npr.randn(10)
    y = npr.randn(10)
    z = npr.randn(10)

    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_multiply,
                                            x, y, z)
    expr = self._rewriter_test_helper(fun, rewrites.distribute_einsum, x, y, z,
                                      expr=expr)

    self.assertEqual(expr.expr_node.fun, tracers.add_n)

  def testEinsumOneArg(self):
    x = npr.randn(10)

    def fun(x):
      return np.einsum('a->a', x)

    expr = tracers.make_expr(fun, x)
    self.assertNotEqual(expr.expr_node.fun, tracers.env_lookup)
    expr = tracers.remake_expr(expr, {np.einsum: rewrites.maybe_einsum})
    self.assertAllClose(tracers.eval_expr(expr, {'x': x}), fun(x))
    self.assertEqual(expr.expr_node.fun, tracers.env_lookup)

  def _test_einsum_zero(self, fun, x):
    expr = tracers.make_expr(fun, x)
    einsum_node = expr.expr_node.parents[0].parents[0]
    self.assertEqual(einsum_node.fun, np.einsum)
    expr = self._eager_rewriter_test_helper(fun, rewrites.maybe_einsum, x,
                                            expr=expr, rewrite_node=einsum_node)
    self.assertIsInstance(expr, tracers.ConstExpr)

  def testEinsumZero(self):
    x = npr.randn(10)
    zero = np.zeros_like(x)

    def fun(x):
      return 3.0 + np.sum(np.einsum('i,i->i', zero, x))
    self._test_einsum_zero(fun, x)

    def fun(x):
      return 3.0 + np.sum(np.einsum('i,i->i', x, zero))
    self._test_einsum_zero(fun, x)

    def fun(x):
      return 3.0 + np.sum(np.einsum('i,i,i->i', x, zero, x))
    self._test_einsum_zero(fun, x)

    def fun(x):
      return 3.0 + (0.0 + np.einsum('i,i->', x, zero))
    self._test_einsum_zero(fun, x)

    def fun(x):
      return 3.0 + (0.0 + np.einsum(',->', np.sum(x), 0.0))
    self._test_einsum_zero(fun, x)

  def testFoldedEinsum(self):
    x = npr.randn(10)
    ones = np.ones(5)

    def fun(x):
      return np.einsum(',,i,->', np.sum(x), 0.5, x, 2.0)
    self.assertAlmostEqual(fun(x), x.sum() ** 2)

    def fun(x):
      return rewrites.constant_folding_einsum(',,i,->', np.sum(x), 0.5, x, 2.0)
    self.assertAlmostEqual(fun(x), x.sum() ** 2)

    expr = tracers.make_expr(fun, x)
    self.assertEqual(len(expr.expr_node.args), 3)

    def fun(x):
      return rewrites.constant_folding_einsum(',,i,->', np.sum(x), 0., x, 2.0)
    self.assertEqual(fun(x), 0.)

    expr = tracers.make_expr(fun, x)
    self.assertIsInstance(expr, tracers.ConstExpr)
    self.assertEqual(expr.val, 0.)

    def fun(x):
      return np.einsum(',j,i,j->', np.sum(x), 0.5 * ones, x, 2.0 * ones)
    self.assertAlmostEqual(fun(x), x.sum() ** 2 * len(ones))

    def fun(x):
      return rewrites.constant_folding_einsum(',j,i,j->', np.sum(x), 0.5 * ones,
                                              x, 2.0 * ones)
    self.assertAlmostEqual(fun(x), x.sum() ** 2 * len(ones))

    expr = tracers.make_expr(fun, x)
    self.assertEqual(len(expr.expr_node.args), 4)

    def fun(x):
      return rewrites.constant_folding_einsum(',i,i,j->', np.sum(x),
                                              np.ones_like(x), x, ones)
    self.assertAlmostEqual(fun(x), x.sum() ** 2 * len(ones))

    expr = tracers.make_expr(fun, x)
    self.assertEqual(len(expr.expr_node.args), 4)

    def fun(x):
      return rewrites.constant_folding_einsum(',j,i,j->', np.sum(x), 0. * ones,
                                              x, 2.0 * ones)
    self.assertEqual(fun(x), 0.)

    expr = tracers.make_expr(fun, x)
    self.assertIsInstance(expr, tracers.ConstExpr)
    self.assertEqual(expr.val, 0.)

  def testGatherLogAddEinsum(self):
    a = abs(npr.randn())
    x = abs(npr.randn(10))

    def fun(a, x, y, z):
      return np.log(tracers.add_n(np.einsum(',a->', a, x),
                                  np.einsum(',a->', a, y),
                                  np.einsum(',a->', a, z)))

    expr = self._rewriter_test_helper(fun, rewrites.gather_log_add_einsum,
                                      a, x, x, x)
    self.assertEqual(expr.expr_node.parents[0].fun, np.log)
    self.assertEqual(expr.expr_node.parents[1].fun, np.log)

  def testAddPowersWithinEinsum(self):
    x = npr.randn()

    def fun(x):
      return np.einsum(',,->', x ** 2, x ** 2, 3.)

    expr = self._rewriter_test_helper(fun, rewrites.add_powers_within_einsum, x)
    self.assertEqual(expr.expr_node.fun, np.einsum)
    self.assertTrue(any([node.fun == np.power and node.args[1] == 4
                         for node in expr.expr_node.parents]))

  def testIncrementNegativePowerInEinsum(self):
    x = npr.randn(10)

    def fun(x):
      return np.einsum(',a,a,a,a->', 3., x, x ** -3, x, x)

    expr = self._rewriter_test_helper(
        fun, rewrites.increment_negative_power_in_einsum_r, x)
    self.assertEqual(expr.expr_node.fun, np.einsum)
    self.assertTrue(any([node.fun == np.power and node.args[1] == -2
                         for node in expr.expr_node.parents]))
    expr = self._rewriter_test_helper(
        fun, rewrites.increment_negative_power_in_einsum_r, x, expr=expr)
    self.assertEqual(expr.expr_node.fun, np.einsum)
    self.assertTrue(any([node.fun == np.power and node.args[1] == -1
                         for node in expr.expr_node.parents]))
    expr = self._rewriter_test_helper(
        fun, rewrites.increment_negative_power_in_einsum_l, x, expr=expr)
    self.assertEqual(expr.expr_node.fun, np.einsum)
    self.assertTrue(any([node.fun == np.power and node.args[1] == 0
                         for node in expr.expr_node.parents]))

  def testSwapaxesToEinsum(self):
    x = np.arange(9).reshape([3, 3])
    self.assertTrue((np.swapaxes(x, 0, 1) == rewrites.swapaxes(x, 0, 1)).all())

  def testRenameFormulaIndices(self):
    self.assertEqual(
        rewrites._rename_formula_indices('...ikj->...jk'), '...abc->...cb')

  def testDebroadcastFormula(self):
    self.assertEqual(
        rewrites.debroadcast_formula('...i,...j->...', *[1, 1]), 'a,b->')
    self.assertEqual(
        rewrites.debroadcast_formula('...i,...j->...', *[2, 2]), 'ab,ac->a')
    # _remove_ellipsis would fail this test
    self.assertEqual(
        rewrites.debroadcast_formula('...,...->...', *[1, 1]), 'a,a->a')
    self.assertEqual(
        rewrites.debroadcast_formula(
            '...a,...b->...ab', *[2, 3]), 'ab,cad->cabd')

  def testEinsumRepeatedOneHot(self):
    x = npr.randn(3, 2)
    y = npr.randn(3, 2)
    e = npr.randint(0, x.shape[0], 5)

    def fun(x, y, e):
      one_hot_e = tracers.one_hot(e, x.shape[0])
      return np.einsum('ab,bc,ad,dc->', one_hot_e, x, one_hot_e, y)

    expr = self._rewriter_test_helper(
        fun, rewrites.einsum_repeated_one_hot, x, y, e)
    self.assertEqual(len(expr.expr_node.args), 4)
    self.assertEqual(sum(node.fun == tracers.one_hot
                         for node in expr.expr_node.parents), 1)

    def fun(x, y, e):
      one_hot_e = tracers.one_hot(e, x.shape[0])
      return np.einsum('ab,bc,ad,dc->ac', one_hot_e, x, one_hot_e, y)

    expr = self._rewriter_test_helper(
        fun, rewrites.einsum_repeated_one_hot, x, y, e)
    self.assertEqual(len(expr.expr_node.args), 4)
    self.assertEqual(sum(node.fun == tracers.one_hot
                         for node in expr.expr_node.parents), 1)

  def testGatherPowAddMul(self):
    x = npr.randn()
    a = npr.randn(10)
    b = npr.randn(10)

    def fun(x, a, b):
      return tracers.add_n(np.einsum(',a->a', x, a), np.einsum(',a->a', x, b)) ** 3

    expr = self._rewriter_test_helper(fun, rewrites.gather_pow_add_einsum, x, a, b)
    self.assertEqual(expr.expr_node.fun, np.multiply)
    self.assertEqual(expr.expr_node.parents[0].fun, np.power)
    self.assertEqual(expr.expr_node.parents[1].fun, np.power)

  def testGatherInvAddMul(self):
    x = npr.randn()
    a = 2. * np.eye(2)
    b = 2.5 * np.eye(2)

    def fun(x, a, b):
      return np.linalg.inv(tracers.add_n(np.einsum(',ab->ab', x, a), np.einsum(',ab->ab', x, b)))

    expr = self._rewriter_test_helper(fun, rewrites.gather_inv_add_einsum, x, a, b)
    self.assertEqual(expr.expr_node.fun, np.multiply)
    parent_funs = [parent.fun for parent in expr.expr_node.parents]
    self.assertTrue(np.power in parent_funs)
    self.assertTrue(np.linalg.inv in parent_funs)

  def testGatherLogdetAddMul(self):
    x = np.exp(npr.randn())
    a = 2. * np.eye(2)
    b = 2.5 * np.eye(2)

    def fun(x, a, b):
      return tracers.logdet(tracers.add_n(np.einsum(',ab->ab', x, a), np.einsum(',ab->ab', x, b)))

    expr = self._rewriter_test_helper(fun, rewrites.gather_logdet_add_einsum, x, a, b)
    self.assertEqual(expr.expr_node.fun, np.add)
    parent_funs = [parent.fun for parent in expr.expr_node.parents]
    self.assertTrue(tracers.logdet in parent_funs)


if __name__ == '__main__':
  absltest.main()
