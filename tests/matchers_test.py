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
import autograd.numpy.random as npr

from autoconj.patterns import (Val, Node, Array, Str, Add, AddN, Subtract,
                               Multiply, Power, Dot, Einsum, Choice, Segment,
                               Star)
from autoconj import matchers
from autoconj import patterns
from autoconj import tracers


class MatchersTest(absltest.TestCase):

  def testOneElementPattern(self):

    def fun(x, y):
      return 3 * x + y**2

    x = np.ones(2)
    y = 2 * np.ones(2)
    end_node = tracers.make_expr(fun, x, y).expr_node

    match = matchers.matcher(Val)
    self.assertTrue(match(end_node))

    match = matchers.matcher(Add)
    self.assertTrue(match(end_node))

    match = matchers.matcher(Multiply)
    self.assertFalse(match(end_node))

  def testOneElementPatternNameBinding(self):

    def fun(x, y):
      return 3 * x + y**2

    x = np.ones(2)
    y = 2 * np.ones(2)
    end_node = tracers.make_expr(fun, x, y).expr_node

    match = matchers.matcher(Val('z'))
    self.assertEqual(match(end_node), {'z': end_node})

    match = matchers.matcher(Add('z'))
    self.assertEqual(match(end_node), {'z': end_node.fun})

    match = matchers.matcher(Multiply('z'))
    self.assertFalse(match(end_node))

  def testLiterals(self):

    match = matchers.matcher(3)
    self.assertTrue(match(3))

    def fun(x):
      return 2 + x

    x = np.ones(2)
    end_node = tracers.make_expr(fun, x).expr_node

    match = matchers.matcher((Add, 2, Val))
    self.assertTrue(match(end_node))

  def testCompoundPattern(self):

    def fun(x, y):
      return 3 * x + y**2

    x = np.ones(2)
    y = 2 * np.ones(2)
    end_node = tracers.make_expr(fun, x, y).expr_node

    match = matchers.matcher((Add, Val, Val))
    self.assertTrue(match(end_node))

    match = matchers.matcher((Add, Multiply, Val))
    self.assertTrue(match(end_node))

    match = matchers.matcher((Add, (Multiply, Val, Val), Val))
    self.assertTrue(match(end_node))

    match = matchers.matcher((Add, (Multiply, 3, Val), (Power, Val, 2)))
    self.assertTrue(match(end_node))

    match = matchers.matcher((Add, (Add, Val, Val), Val))
    self.assertFalse(match(end_node))

    match = matchers.matcher((Add, (Multiply, 4, Val), (Power, Val, 2)))
    self.assertFalse(match(end_node))

  def testCompoundPatternNameBindings(self):

    def fun(x, y):
      return 3 * x + y**2

    x = np.ones(2)
    y = 2 * np.ones(2)
    end_node = tracers.make_expr(fun, x, y).expr_node

    match = matchers.matcher((Add,
                                (Multiply, 3, Val('x')),
                                (Power, Val('y'), 2)))
    self.assertEqual(match(end_node),
                     {'x': end_node.args[0].args[1],
                      'y': end_node.args[1].args[0]})

  def testCompoundPatternNameConstraints(self):
    def fun(x, y):
      return 3 * x + y**2

    x = np.ones(2)
    y = 2 * np.ones(2)
    end_node = tracers.make_expr(fun, x, y).expr_node

    match = matchers.matcher((Add,
                                (Multiply, 3, Val('x')),
                                (Power, Val('x'), 2)))
    self.assertFalse(match(end_node))

    def fun(x, y):
      return 3 * x + x**2  # note x used twice

    x = np.ones(2)
    y = 2 * np.ones(2)
    end_node = tracers.make_expr(fun, x, y).expr_node

    self.assertEqual(match(end_node),
                     {'x': end_node.args[0].args[1]})

  def testChoices(self):
    W = npr.randn(3, 3)
    b = npr.randn(3)

    def fun(x):
      return np.dot(x, W) + b

    x = np.ones((5, 3))
    end_node = tracers.make_expr(fun, x).expr_node

    match = matchers.matcher((Add, Choice(Dot('op'), Multiply('op')), Val))
    self.assertEqual(match(end_node),
                     {'op': end_node.args[0].fun})

    match = matchers.matcher((Add, Choice(Add('op'), Multiply('op')), Val))
    self.assertFalse(match(end_node))

    match = matchers.matcher((Choice((Add, (Multiply, Val, Val)),  # backtrack
                                     (Add, (Dot, Val('x'), Val('W')), Val('b')),
                                     (Dot, Val('x'), Val('W')))))
    self.assertEqual(match(end_node),
                     {'x': end_node.args[0].args[0],
                      'W': end_node.args[0].args[1],
                      'b': end_node.args[1]})

  def testSegments(self):

    def fun(x):
      return np.einsum('i,j,,k->ijk', x, x, 2, x)

    x = np.ones(3)
    end_node = tracers.make_expr(fun, x).expr_node

    match = matchers.matcher((Einsum, Str, Segment, 2, Segment))
    self.assertTrue(match(end_node))

    match = matchers.matcher((Einsum, Str, Segment, 3, Segment))
    self.assertFalse(match(end_node))

    match = matchers.matcher((Einsum, Str, Segment('s1'), 2, Segment('s2')))
    bindings = match(end_node)
    self.assertTrue('s1' in bindings)
    self.assertEqual(len(bindings['s1']), 2)

    match = matchers.matcher((Einsum, Str, Segment('s1'), 2, Segment('s2')))
    bindings = match(end_node)
    self.assertTrue('s1' in bindings)
    self.assertEqual(len(bindings['s1']), 2)

    match = matchers.matcher((Einsum, Str, Segment('s1'), 2, Array,
                              Segment('s2')))
    bindings = match(end_node)
    self.assertTrue('s2' in bindings)
    self.assertEqual(len(bindings['s2']), 0)

  def testSegmentsEmpty(self):

    def fun(x, y, z):
      return np.einsum('i,j,ij->', x - y, x, z)

    x = np.ones(3)
    y = 2 * np.ones(3)
    z = 3 * np.ones((3, 3))
    end_node = tracers.make_expr(fun, x, y, z).expr_node

    pat = (Einsum, Str('formula'),
           Segment('args1'),
           (Choice(Subtract('op'), Add('op')), Val('x'), Val('y')),
           Segment('args2'))
    match = matchers.matcher(pat)
    self.assertTrue(match(end_node))

  def testStar(self):
    x = np.ones(3)

    def f(x):
      return np.einsum('i,j->', x, x)
    f_expr = tracers.make_expr(f, x)

    def g(x):
      return np.einsum('i,j->', x, 3 * np.ones(x.shape))
    g_expr = tracers.make_expr(g, x)

    pat = (Einsum, Str('formula'), Star(Val('x')))
    match = matchers.matcher(pat)

    self.assertTrue(match(f_expr.expr_node))
    self.assertFalse(match(g_expr.expr_node))

  def testStarRepeatedNames(self):
    x = np.ones(3)

    def f(x):
      return np.einsum('i,j,k,l,m->', x, x, 3 * np.ones(x.shape), x, x)
    f_expr = tracers.make_expr(f, x)

    def g(x):
      return np.einsum('i,j,k,l->', x, x, 3 * np.ones(x.shape), x)
    g_expr = tracers.make_expr(g, x)

    pat = (Einsum, Str('formula'),
           Star(Val('x'), 'xs'), Val, Star(Val('x'), 'xs'))
    match = matchers.matcher(pat)

    self.assertTrue(match(f_expr.expr_node))
    self.assertFalse(match(g_expr.expr_node))

  def testAccumulateInStar(self):
    def f(x):
      return np.einsum('i,j,k->', x, x, 3*np.ones(x.shape))
    x = np.ones(3)
    f_expr = tracers.make_expr(f, x)

    pat = (Einsum, Str('formula'), Star(Val('args'), accumulate=['args']))
    match_fn = matchers.matcher(pat)
    # should produce:
    # bindings = {'args': (x, x, 3*np.ones(x.shape)), 'formula': 'i,j,k->'}
    self.assertTrue(match_fn(f_expr.expr_node))

    def f(x):
      return tracers.add_n(np.einsum(',i->i', x, np.ones(3)),
                           np.einsum(',j->j', x, 2. * np.ones(3)))
    x = 2.5
    f_expr = tracers.make_expr(f, x)

    pat = (AddN, Star((Einsum, Str('formula'),
                       Segment('args1'), Node('x'), Segment('args2')),
                      accumulate=['formula', 'args1', 'args2']))
    match_fn = matchers.matcher(pat)
    match = match_fn(f_expr.expr_node)
    self.assertEqual(len(match['formula']), 2)
    self.assertEqual(len(match['args1']), 2)
    self.assertEqual(len(match['args2']), 2)
    self.assertEqual(match['x'].fun.__name__, 'env_lookup')
    self.assertIn(',i->i', match['formula'])
    self.assertIn(',j->j', match['formula'])


if __name__ == '__main__':
  absltest.main()
