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

from autograd import grad
import autograd.numpy as np

from autoconj import tracers


class TracersTest(absltest.TestCase):

  def _CheckFunctionExprEquality(self, expected, fun_a, fun_b, *args):
    expr_a = tracers.make_expr(fun_a, *args)
    expr_b = tracers.make_expr(fun_b, *args)
    if expected:
      self.assertEqual(expr_a, expr_b)
    else:
      self.assertNotEqual(expr_a, expr_b)

  def testEquals(self):

    def fun(x):
      return f(g(h(x)))

    f = lambda x: np.power(x, 3.)
    g = lambda x: np.power(3., x)
    h = lambda x: np.power(x, x)

    x = 2.

    self._CheckFunctionExprEquality(True, fun, lambda x: f(g(h(x))), x)
    self._CheckFunctionExprEquality(False, fun, lambda x: f(g(x)), x)
    self._CheckFunctionExprEquality(False, fun, lambda x: f(h(x)), x)
    self._CheckFunctionExprEquality(False, fun, lambda x: g(h(x)), x)
    self._CheckFunctionExprEquality(True, f, f, x)
    self._CheckFunctionExprEquality(True, g, g, x)
    self._CheckFunctionExprEquality(True, h, h, x)
    self._CheckFunctionExprEquality(False, fun, f, x)
    self._CheckFunctionExprEquality(False, fun, g, x)
    self._CheckFunctionExprEquality(False, fun, h, x)
    self._CheckFunctionExprEquality(False, f, g, x)
    self._CheckFunctionExprEquality(False, f, h, x)
    self._CheckFunctionExprEquality(False, g, h, x)

  def testPrintExpr(self):

    def fun(x, y):
      return 2 * x**2 + np.tanh(y)

    expr = tracers.make_expr(fun, 4, 5)
    printed_expr = tracers.print_expr(expr)

    expected = ("temp_0 = power(x, 2)\n"
                "temp_1 = multiply(2, temp_0)\n"
                "temp_2 = tanh(y)\n"
                "temp_3 = add(temp_1, temp_2)\n")
    self.assertEqual(printed_expr, expected)

  def testEvalExpr(self):

    def fun(x, y):
      return 2 * x**2 + np.tanh(3 * y)

    expr = tracers.make_expr(fun, 4, 5)
    self.assertEqual(fun(9, 10), tracers.eval_expr(expr, {'x': 9, 'y': 10}))

  def testInlineExpr(self):

    def f(x, y):
      return 2 * x + y

    def g(z):
      return 3 * z + z**2

    expr = tracers.make_expr(f, 1, 2)
    subexpr = tracers.make_expr(g, 3)

    target_node = expr.expr_node.parents[0]
    new_expr = tracers.inline_expr(subexpr, {'z': target_node})
    printed_expr = tracers.print_expr(new_expr)
    expected = ("temp_0 = multiply(2, x)\n"
                "temp_1 = multiply(3, temp_0)\n"
                "temp_2 = power(temp_0, 2)\n"
                "temp_3 = add(temp_1, temp_2)\n")
    self.assertEqual(printed_expr, expected)
    self.assertEqual(3 * (2 * 5) + (2 * 5)**2,
                     tracers.eval_expr(new_expr, {'x': 5}))
    self.assertEqual(f(6, 7), tracers.eval_expr(expr, {'x': 6, 'y': 7}))

  def testReplaceNodeWithExpr(self):

    def f(x):
      return 2 * x

    def g(x):
      return 3 * x

    expr = tracers.make_expr(f, 5)
    new_expr = tracers.make_expr(g, 10)
    tracers.replace_node_with_expr(expr.expr_node, new_expr)

    self.assertEqual(3 * 7, tracers.eval_expr(expr, {'x': 7}))

  def testInlineExprAndReplace(self):

    def f(x, y):
      return 2 * x ** 2 + y

    def g(z):
      return 3 * z ** 3

    expr = tracers.make_expr(f, 1, 2)
    subexpr = tracers.make_expr(g, 3)

    input_node = expr.expr_node.parents[0].parents[0]  # x ** 2
    output_node = expr.expr_node.parents[0]  # 2 * x ** 2

    new_expr = tracers.inline_expr(subexpr, {'z': input_node})
    tracers.replace_node_with_expr(output_node, new_expr)  # modify expr inplace

    self.assertEqual(3 * 6 ** 6 + 7,
                     tracers.eval_expr(expr, {'x': 6, 'y': 7}))

  def testUnusedVars(self):

    def f(x, y, z):
      return 3 * x + y

    expr = tracers.make_expr(f, 1., 2., 3.)

    self.assertEqual(set(expr.free_vars.keys()), {'x', 'y'})

  def testDescendantOf(self):

    def f(x, y):
      return 2 * x ** 2 + y

    expr = tracers.make_expr(f, 1, 2)
    xnode = expr.free_vars['x']
    ynode = expr.free_vars['y']

    self.assertTrue(tracers.is_descendant_of(expr.expr_node, xnode))
    self.assertTrue(tracers.is_descendant_of(xnode, xnode))
    self.assertTrue(tracers.is_descendant_of(expr.expr_node, expr.expr_node))
    self.assertFalse(tracers.is_descendant_of(xnode, ynode))

  def testAllDescendantsOf(self):

    def f(x, y):
      return 2 * x ** 2 + y

    expr = tracers.make_expr(f, 1, 2)
    xnode = expr.free_vars['x']
    ynode = expr.free_vars['y']

    descendants = tracers.all_descendants_of(expr.expr_node, ynode)
    self.assertEqual(descendants, {ynode, expr.expr_node})

  def testCommonSubexpressionElimination(self):

    def f1(x):
      return 3 * x**2 + x**2

    def f2(x):
      y = x**2
      return 3 * y + y

    expr1 = tracers.make_expr(f1, 1)
    expr2 = tracers.make_expr(f2, 1)

    code1 = tracers.print_expr(expr1)
    code2 = tracers.print_expr(expr2)
    self.assertGreater(len(code1), len(code2))

    code1_cse = tracers.print_expr(tracers.remake_expr(expr1))  # applies cse
    self.assertEqual(len(code1_cse), len(code2))

  def testExtractSuperexpr(self):

    def f(x, y):
      return 2 * x ** 2 + y

    expr = tracers.make_expr(f, 1, 2)
    node = expr.expr_node.parents[0].parents[0]  # x ** 2

    new_expr = tracers.extract_superexpr(expr, {'x2': node})

    self.assertEqual(2 * 5 + 6, tracers.eval_expr(new_expr, {'x2': 5, 'y': 6}))

  def testExtractSuperexprWithReplaceNode(self):
    # NOTE(mattjj): this test shows an alternative way to implement, in effect,
    # tracers.extract_superexpr just using tracers.replace_node_with_expr. The
    # reason to have both is that one does in-place modification.

    def f(x, y):
      return 2 * x ** 2 + y

    expr = tracers.make_expr(f, 1, 2)
    node = expr.expr_node.parents[0].parents[0]  # x ** 2

    lookup_expr = tracers.make_expr(lambda x: x, 3, names=('x2',))
    tracers.replace_node_with_expr(node, lookup_expr)  # modify expr in-place

    self.assertEqual(2 * 5 + 6, tracers.eval_expr(expr, {'x2': 5, 'y': 6}))


if __name__ == '__main__':
  absltest.main()
