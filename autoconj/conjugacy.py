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
"""Functions to recognize conjugacy relationships in log-joint functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import itertools
from os.path import commonprefix
from types import FunctionType, CodeType

from autograd import numpy as np
from autograd import make_vjp
from autograd import grad

from .canonicalize import canonicalize, hierarchy_level, NodeTypes
from .exponential_families import find_distributions, exp_family_stats
from .tracers import (all_descendants_of, ConstExpr, draw_expr, env_lookup ,
                      eval_expr, eval_node, ExprNode, extract_superexpr,
                      GraphExpr, is_descendant_of, make_expr, make_node,
                      _mutate_node, print_expr, remake_expr, subvals)
from .util import split_einsum_formula, support_type_to_name,  SupportTypes


def find_sufficient_statistic_nodes(expr, free_var, split_einsums=False):
  r"""Finds nodes in `expr` that represent sufficient statistics of a free var.

  This function assumes that `canonicalize()` has already been called on the
  graph. It may behave strangely if the graph is not in canonical form.

  Algebraically, we assume that expr encodes an exponential family log density
  function in free_var (potentially unnormalized), so that expr has the form
    expr = \eta \dot t(x) = eta_1 \dot t_1(x) + ... + \eta_K \dot t_K(x) + const
  where each t_k is a sufficient statistic function, which is either:
    1. an identity function, or
    2. a nonlinear function.
  The nonlinearity requirement ensures that we can separate the statistics
  functions from the linear/affine form. In terms of the GraphExpr data
  structure, a sufficient statistic function corresponds to a node `node` in the
  graph where:
    1. `node` has a variable reference to free_var as an ancestor;
    2. for each node between `node` and expr.expr_node, that node is a linear
       function;
    3. `node` is either a nonlinear function of free_var or is itself a variable
       reference to free_var.

  Note that monomials implemented by einsum are handled a little differently
  than other sufficient statistics, since an einsum can be either linear or
  nonlinear in free_var depending on the degrees of its arguments. That is,
  because we don't separate out nonlinear monomials into their own einsum nodes,
  this function will return the full einsum node (including linear interactions
  with other terms), which requires additional parsing to extract natural
  parameters.

  Args:
    expr: an expression that is an affine function of a tuple of intermediate
      (potentially nonlinear) functions of `free_var`.
    free_var: a free variable in expr, either a string name or int index.
    split_einsums: optional, bool for whether to in-place-modify expr to split
      einsums (default False).

  Returns:
    A set of ExprNodes representing sufficient statistics functions of free_var
    (but possibly containing multiple nodes that represent the same expression).
  """
  if isinstance(expr, ConstExpr):
    return set()
  elif isinstance(expr, GraphExpr):
    var_node = expr.free_vars.get(free_var) or expr.free_vars.values()[free_var]
    desc = all_descendants_of(expr.expr_node, var_node)
    visited = set()
    suff_stats = set()

    def collect_suff_stats(node):
      visited.add(node)
      lvl = hierarchy_level[node.fun]
      if lvl == NodeTypes.OTHER:
        suff_stats.add(node)
      elif lvl == NodeTypes.EINSUM and sum(p in desc for p in node.parents) > 1:
        if split_einsums:
          argnums = [i for i, a in enumerate(node.args[1:])
                     if isinstance(a, ExprNode) and a in desc]
          potential_node, stat_node = split_einsum_node(node, argnums)
          _mutate_node(node, potential_node)  # mutates node and hence expr too
          suff_stats.add(stat_node)
        else:
          suff_stats.add(node)
      else:
        for p in node.parents:
          if p not in visited and p in desc:
            collect_suff_stats(p)

    collect_suff_stats(expr.expr_node)
    return suff_stats
  else:
    raise TypeError("Can't handle expression type: {}".format(type(expr)))


_einsum_range = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_einsum_index_set = frozenset(_einsum_range)


def _canonicalize_einsum_formula(formula):
  in_formulas, out_formula = split_einsum_formula(formula)
  i = len(commonprefix({f for f in in_formulas if f} | {out_formula}))
  in_formulas = [in_formula[i:] for in_formula in in_formulas]
  out_formula = out_formula[i:]

  in_formulas = ['...' + in_formula for in_formula in in_formulas]
  out_formula = '...' + out_formula

  # Relabel all index names in canonical order.
  index_map = defaultdict(iter(_einsum_range).next)
  return ''.join(index_map[char] if char in _einsum_index_set else char
                 for char in '{}->{}'.format(','.join(in_formulas),
                                             out_formula))


def make_zeros(stat):
  if stat.__class__ in exp_family_stats:
    return stat.__class__(**{name: make_zeros(v)
                             for name, v in stat._asdict().iteritems()})
  elif isinstance(stat, tuple):
    return tuple([make_zeros(item) for item in stat])
  elif isinstance(stat, dict):
    return {k: make_zeros(v) for k, v in stat.iteritems()}
  else:
    return np.zeros_like(stat)


# TODO(mattjj): revise to use inline_expr and replace_node_with_expr
def marginalize(log_joint_fun, argnum, support, *args):
  new_log_joint_fun, log_normalizers, stats_funs, _ = (
      statistic_representation(log_joint_fun, args, (support,), (argnum,)))
  log_normalizer, stat_fun = log_normalizers[0], stats_funs[0]
  stat_zeros = make_zeros(stat_fun(args[argnum]))

  def marginalized_log_prob(*new_args):
    new_args = new_args[:argnum] + (stat_zeros,) + new_args[argnum:]
    log_joint = new_log_joint_fun(*new_args)
    natural_parameters = grad_namedtuple(new_log_joint_fun, argnum)(*new_args)
    log_normalizer_val = log_normalizer(natural_parameters)
    return log_joint + log_normalizer_val

  return marginalized_log_prob


def complete_conditional(log_joint_fun, argnum, support, *args):
  """Infers tractable complete-conditional distributions from log-joints.

  Args:
    log_joint_fun: A callable that returns the log-joint probability of its
      arguments under some model.
    argnum: Integer position of the argument to log_joint_fun whose
      complete conditional we want. For example, if argnum == 1 and
      log_joint_fun(x, y, z) is the joint log-probability log p(x, y,
      z), then this function will try to return p(y | x, z).
    support:
    *args: Arguments to log_joint_fun. These are needed for the tracer.

  Returns:
    conditional_factory: A callable that takes the same args as
      log_joint_fun() and returns the complete conditional as a frozen
      scipy.stats distribution.
  """
  # TODO(mhoffman): Make it possible to pass multiple argnums and
  # return multiple conditionals.
  new_log_joint_fun, log_normalizers, stats_funs, distbns = (
      statistic_representation(log_joint_fun, args, (support,),  (argnum,)))
  log_normalizer, stat_fun, distbn = (
      log_normalizers[0], stats_funs[0],distbns[0])
  stat_zeros = make_zeros(stat_fun(args[argnum]))

  def conditional_factory(*new_args):
    new_args = new_args[:argnum] + (stat_zeros,) + new_args[argnum:]
    natural_parameters = grad_namedtuple(new_log_joint_fun, argnum)(*new_args)
    return distbn(natural_parameters)
  return conditional_factory


def statistic_representation(log_joint_fun, args, supports, argnums=None):
  if argnums is None:
    argnums = range(len(args))
  # TODO(mattjj): add optimization to not always split the einsum node
  expr = _split_einsum_stats(canonicalize(make_expr(log_joint_fun, *args)))
  names = [expr.free_vars.keys()[argnum] for argnum in argnums]
  stats_nodes = [find_sufficient_statistic_nodes(expr, name) for name in names]
  stats_nodes, log_normalizers, distbns = (
      find_distributions(stats_nodes, supports))

  new_log_joint_fun = _make_stat_log_joint(expr, argnums, stats_nodes, supports)
  make_stat_fun = (lambda name, nodes: lambda arg:
                   eval_node(nodes, expr.free_vars, {name: arg}))
  stats_funs = map(make_stat_fun, names, stats_nodes)
  return new_log_joint_fun, log_normalizers, stats_funs, distbns


def make_initializers(args, neg_energy, normalizers, stats_funs):
  stats_vals = [stat_fun(arg) for stat_fun, arg in zip(stats_funs, args)]
  make_nat_init = (lambda i: lambda scale=1.:
                   make_vjp(neg_energy, i)(*stats_vals)[0](scale))
  natural_initializers = map(make_nat_init, range(len(normalizers)))
  make_mean_init = (lambda (i, normalizer): lambda scale=1.:
                   grad(normalizer)(make_vjp(neg_energy, i)(*stats_vals)[0](scale)))
  mean_initializers = map(make_mean_init, enumerate(normalizers))

  return natural_initializers, mean_initializers


def _split_einsum_stats(expr):
  expr = remake_expr(expr)  # copy to avoid mutating expr
  for name in expr.free_vars:
    find_sufficient_statistic_nodes(expr, name, split_einsums=True)  # mutates
  return remake_expr(expr)  # common-subexpression elimination


def flat_dict(stats):
  stats_dict = {}

  def add_to_dict(item, name_so_far):
    if item.__class__ in exp_family_stats:
      for subname, subitem in item._asdict().iteritems():
        add_to_dict(subitem, name_so_far + '_' + subname)
    elif isinstance(item, tuple) or isinstance(item, list):
      for subname, subitem in enumerate(item):
        add_to_dict(subitem, name_so_far + '_' + str(subname))
    elif isinstance(item, dict):
      for subname, subitem in item.iteritems():
        add_to_dict(subitem, name_so_far + '_' + str(subname))
    elif item is not None:
      stats_dict[name_so_far] = item

  add_to_dict(stats, '')
  return stats_dict


def _make_stat_log_joint(expr, argnums, stats_nodes, supports):
  names = tuple(expr.free_vars.keys())
  g_expr = extract_superexpr(expr, flat_dict(stats_nodes))
  def construct_env(args):
    env = {name: arg for i, (name, arg)
           in enumerate(zip(names, args)) if i not in argnums}
    flat_stats_dict = flat_dict([args[argnum] for argnum in argnums])
    return dict(env, **flat_stats_dict)
  g_raw = lambda *args: eval_expr(g_expr, construct_env(args))
  g = make_fun(g_raw, name='neg_energy', varnames=names)
  return g


def make_fun(fun, **kwargs):
  code = fun.func_code
  attr_names = ['argcount', 'nlocals', 'stacksize', 'flags', 'code',
                'consts', 'names', 'varnames', 'filename', 'name',
                'firstlineno', 'lnotab', 'freevars', 'cellvars']
  new_code = CodeType(*(kwargs.get(name, getattr(code, 'co_' + name))
                        for name in attr_names))
  return FunctionType(new_code, fun.func_globals, closure=fun.func_closure)


def split_einsum_node(node, stat_argnums, canonicalize=True):
  """Pushes part of an einsum computation up a level in the graph.

  Args:
    node: The einsum ExprNode to break up. Must contract to a scalar.
    stat_argnums: Which non-formula arguments to push out of `node`.

  Returns:
    potential_node: A new einsum ExprNode that computes the same
      function as `node`, but only indirectly depends on the arguments
      pointed to by `stat_argnums` through the newly created
      `stat_node`.
    stat_node: A new einsum ExprNode that depends directly on the
      arguments pointed to by `stat_argnums`, and is used as an argument
      to `potential_node`.

  Examples:
  ```
  stat_argnums == [2, 3]:
  einsum('ij,ik,j,k->', X, X, beta, beta)
  =>
  einsum('ij,ik,jk->', X, X, einsum('j,k->jk', beta, beta))

  stat_argnums == [0, 1]:
  einsum('...ab,...ab,,a->', x, x, -0.5, tau)
  =>
  einsum('...ab,,a->', einsum('...ab,...ab->...ab', x, x), -0.5, tau)
  ```
  """
  formula = node.args[0]
  assert isinstance(formula, str), "Must use string-formula form of einsum."
  in_formulas, out_formula = split_einsum_formula(formula)
  in_formulas = [formula.lstrip('...') for formula in in_formulas]
  out_formula = out_formula.lstrip('...')
  assert not out_formula, "Must contract to a scalar."

  param_argnums = [i for i, _ in enumerate(in_formulas)
                     if i not in stat_argnums]

  stat_inputs = ','.join(in_formulas[i] for i in stat_argnums)
  param_inputs = ','.join(in_formulas[i] for i in param_argnums)
  stat_indexes = ''.join(OrderedDict.fromkeys(stat_inputs.replace(',', '')).keys())

  stat_formula = '{}->{}'.format(stat_inputs, stat_indexes)
  if param_argnums:
    pot_formula = '{},{}->'.format(param_inputs, stat_indexes)
  else:
    pot_formula = '{}->'.format(stat_indexes)

  if canonicalize:
    stat_formula = _canonicalize_einsum_formula(stat_formula)
    pot_formula = _canonicalize_einsum_formula(pot_formula)

  stat_node = make_node(np.einsum,
                        [stat_formula] + [node.args[i+1] for i in stat_argnums],
                        node.kwargs)
  pot_node = make_node(np.einsum,
                       [pot_formula] + [node.args[i+1] for i in param_argnums]
                       + [stat_node],
                       node.kwargs)
  return pot_node, stat_node


def grad_namedtuple(fun, argnum=0):
  assert type(argnum) is int
  def gradfun(*args):
    args = list(args)
    args[argnum], unflatten = _flatten_namedtuple(args[argnum])
    flat_fun = lambda *args: fun(*subvals(args, [(argnum, unflatten(args[argnum]))]))
    return unflatten(grad(flat_fun, argnum)(*args))
  return gradfun

def _flatten_namedtuple(x):
  try:
    return tuple(x), lambda tup: type(x)(*tup)
  except AttributeError:
    return x, lambda x: x
