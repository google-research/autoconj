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
"""Functions that transform a computation graph into a (more) canonical form.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import autograd.extend as ag_extend
import autograd.numpy as np
import autograd.util as ag_util

from . import rewrites
from .tracers import ExprNode
from .tracers import ConstExpr
from .tracers import GraphExpr
from .tracers import add_n
from .tracers import is_descendant_of
from .tracers import print_expr
from .tracers import remake_expr
from .tracers import toposort
from .tracers import env_lookup
from .util import Enum


## canonicalization rule sets


eager_simplifications = {
    np.dot: rewrites.dot_as_einsum,
    np.multiply: rewrites.maybe_multiply,
    np.divide: rewrites.maybe_divide,
    np.true_divide: rewrites.maybe_divide,
    np.add: rewrites.maybe_add,
    np.subtract: rewrites.maybe_subtract,
    np.einsum: rewrites.maybe_einsum,
    ag_util.func(ag_extend.VSpace.add): rewrites.maybe_vspace_add,
    ag_util.func(ag_extend.VSpace.mut_add): rewrites.maybe_vspace_add,
    np.reciprocal: lambda x: x ** -1,
    np.square: lambda x: x ** 2,
    np.sqrt: lambda x: x ** 0.5,
    np.power: rewrites.maybe_power,
    np.swapaxes: rewrites.swapaxes,
    add_n: lambda *args: args[0] if len(args) == 0 else add_n(*args),
}

simplification_rules = [
    rewrites.transpose_inside_einsum,
    rewrites.replace_sum,
    rewrites.combine_einsum_compositions,
    rewrites.distribute_einsum,
    rewrites.einsum_repeated_one_hot,
    rewrites.replace_log_einsum,
    rewrites.fold_power,
    rewrites.add_powers_within_einsum,
    rewrites.expand_integer_power_in_einsum,
    rewrites.increment_negative_power_in_einsum_l,
    rewrites.increment_negative_power_in_einsum_r,
    rewrites.replace_add,
    rewrites.replace_add_addn,
    rewrites.replace_addn_addn,
    rewrites.replace_duplicated_addn,
    rewrites.gather_log_add_einsum,
    rewrites.gather_pow_add_einsum,
    rewrites.gather_inv_add_einsum,
    rewrites.gather_logdet_add_einsum
]

simplifiers = [rewrites.make_rewriter(rule) for rule in simplification_rules]


## main canonicalization functions


def canonicalize(expr, env={}):
  """Canonicalize an expression in an environment."""
  simplification_env = dict(eager_simplifications, **env)
  new_expr = remake_expr(expr, simplification_env)
  while any(simplify_sweep(new_expr, rewrite) for rewrite in simplifiers):
    new_expr = remake_expr(new_expr, simplification_env)
  return new_expr


def simplify_sweep(expr, simplification):
  """Tries to apply a simplification to an expression, returns success bool."""
  if isinstance(expr, ConstExpr):
    return False
  elif isinstance(expr, GraphExpr):
    visited = set()
    def sweep(node):
      visited.add(node)
      return simplification(node) or any(sweep(p) for p in node.parents
                                         if p not in visited)
    return sweep(expr.expr_node)
  else:
    raise TypeError("Can't simplify expression type: {}".format(type(expr)))


## testing for a canonical form

# hierarchy_level[fun_1] > hierarchy_level[fun_2] implies that fun_1
# should be closer to the final node than fun_2 is.

NodeTypes = Enum('NodeTypes', ['OTHER', 'EINSUM', 'LINEAR'])
hierarchy_level = collections.defaultdict(lambda: NodeTypes.OTHER)


def register_node_type(level, *funs):
  hierarchy_level.update(zip(funs, itertools.repeat(level)))

register_node_type(NodeTypes.EINSUM, np.einsum)
register_node_type(NodeTypes.LINEAR, add_n, np.squeeze)


def is_canonical(expr):
  """Necessary but not sufficient tests for a graph to be in canonical form."""
  visited = set()
  is_ref = lambda node: node.fun == env_lookup

  def _is_canonical(node):
    visited.add(node)
    return (is_ref(node) or hierarchy_level[node.fun] == NodeTypes.OTHER
            or all(is_ref(p) or _check_parent(p, node) and _is_canonical(p)
                   for p in node.parents if p not in visited))

  return isinstance(expr, ConstExpr) or _is_canonical(expr.expr_node)


_parent_child_checks = [
    lambda p, c: hierarchy_level[p.fun] <= hierarchy_level[c.fun],
    lambda p, c: not c.fun == p.fun == np.einsum,
    lambda p, c: not (c.fun == np.einsum and p.fun in {np.add, np.subtract}),
]


def _check_parent(parent, child):
  return all(check(parent, child) for check in _parent_child_checks)
