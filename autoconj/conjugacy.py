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
from .exponential_families import (canonical_statistic_strs, suff_stat_to_dist,
                                   suff_stat_to_log_normalizer)
from .tracers import (all_descendants_of, ConstExpr, draw_expr, env_lookup ,
                      eval_expr, eval_node, ExprNode, extract_superexpr,
                      grad_expr, GraphExpr, is_descendant_of, make_expr,
                      make_node, _mutate_node, print_expr, remake_expr)
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
          potential_node, stat_node = split_einsum_node2(node, argnums)
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


def _summarize_node(node, wrt_node=None):
  if not isinstance(node, ExprNode):
    return '{}'.format(node)
  if node.fun is env_lookup:
    if node != wrt_node:
      return 'arg_{}'.format(node.args[0])
    else:
      return 'x'
  arg_summaries = [_summarize_node(arg, wrt_node) for arg in node.args]
  if node.fun.__name__ == 'einsum':
    arg_summaries[0] = _canonicalize_einsum_formula(arg_summaries[0])
  if node.fun.__name__ == 'one_hot':
    arg_summaries = arg_summaries[0]  # Don't record `width` parameter.
  return '{}({})'.format(node.fun.__name__, ', '.join(arg_summaries))


def _zero_out_node(node):
  zeros = node.vs.zeros()
  node.fun = lambda *args: zeros


def _trace_and_analyze(log_joint_fun, argid, *args):
  graph = make_expr(log_joint_fun, *args)
  args = graph.free_vars
  graph = canonicalize(graph)
  sufficient_statistic_nodes, natural_parameter_funs = (
      _extract_conditional_factors(graph, argid))
  return graph, sufficient_statistic_nodes, natural_parameter_funs


# TODO(mattjj): revise to use inline_expr and replace_node_with_expr
def marginalize(log_joint_fun, argnum, support, *args):
  graph, sufficient_statistic_nodes, natural_parameter_funs = (
      _trace_and_analyze(log_joint_fun, argnum, *args))
  natural_parameter_factory = (
      _get_natural_parameter_factory(natural_parameter_funs, support))

  key, _ = natural_parameter_factory(*args)
  log_normalizer = suff_stat_to_log_normalizer[support][key]

  for node in sufficient_statistic_nodes:
    # TODO(mhoffman): Actually remove the node. This should work for now.
    _zero_out_node(node)

  def marginalized_log_prob(*new_args):
    new_args = new_args[:argnum] + (0,) + new_args[argnum:]

    argnames = graph.free_vars.keys()
    env = {argnames[i]: new_args[i] for i in range(len(new_args))}
    log_joint = eval_expr(graph, env)
    natural_parameters = natural_parameter_factory(*new_args)[1]
    log_normalizer_val = log_normalizer(*natural_parameters)
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
  _, _, natural_parameter_funs = (
      _trace_and_analyze(log_joint_fun, argnum, *args))

  natural_parameter_factory = (
      _get_natural_parameter_factory(natural_parameter_funs, support))
  def conditional_factory(*new_args):
    new_args = new_args[:argnum] + (args[argnum],) + new_args[argnum:]
    key, natural_parameters = natural_parameter_factory(*new_args)
    return suff_stat_to_dist[support][key](*natural_parameters)
  return conditional_factory


def multilinear_representation(log_joint_fun, args, supports):
  expr = _split_einsum_stats(canonicalize(make_expr(log_joint_fun, *args)))

  stats_nodes, keys = [], []
  for name, free_node in expr.free_vars.iteritems():
    nodes = find_sufficient_statistic_nodes(expr, name)
    names = (_summarize_node(node, free_node) for node in nodes)
    key, nodes = zip(*sorted(zip(names, nodes)))
    stats_nodes.append(nodes)
    keys.append(key)

  neg_energy = _make_neg_energy(expr, stats_nodes, supports)

  make_normalizer_fun = (lambda key, support: lambda arg:
                         suff_stat_to_log_normalizer[support][key](*arg))
  normalizers = map(make_normalizer_fun, keys, supports)

  make_stat_fun = (lambda name, nodes: lambda arg:
                   tuple(eval_node(node, expr.free_vars, {name: arg})
                         for node in nodes))
  stats_funs = map(make_stat_fun, expr.free_vars, stats_nodes)

  stats_vals = [stat_fun(arg) for stat_fun, arg in zip(stats_funs, args)]
  make_nat_init = (lambda i: lambda scale=1.:
                   make_vjp(neg_energy, i)(*stats_vals)[0](scale))
  natural_initializers = map(make_nat_init, range(len(normalizers)))
  make_mean_init = (lambda (i, normalizer): lambda scale=1.:
                   grad(normalizer)(make_vjp(neg_energy, i)(*stats_vals)[0](scale)))
  mean_initializers = map(make_mean_init, enumerate(normalizers))

  samplers = [suff_stat_to_dist[support][key]
              for key, support in zip(keys, supports)]

  return (neg_energy, normalizers, stats_funs, natural_initializers,
          mean_initializers, samplers)


def _split_einsum_stats(expr):
  expr = remake_expr(expr)  # copy to avoid mutating expr
  for name in expr.free_vars:
    find_sufficient_statistic_nodes(expr, name, split_einsums=True)  # mutates
  return remake_expr(expr)  # common-subexpression elimination


def _make_neg_energy(expr, nodes, supports):
  names = tuple('t_{name}'.format(name=name) for name in expr.free_vars)

  def flat_dict(groups):
    return {'{name}_{idx}'.format(name=name, idx=idx): item
            for name, item_group in zip(names, groups)
            for idx, item in enumerate(item_group)}

  g_expr = extract_superexpr(expr, flat_dict(nodes))
  g_raw = lambda *args: eval_expr(g_expr, flat_dict(args))
  g = make_fun(g_raw, name='neg_energy', varnames=names)

  arg_docs = ("{argname}: {support} with shape {shape}"
              .format(argname=name, shape=tuple(n.vs.shape for n in node_group),
                      support=support_type_to_name[support].title())
              for name, node_group, support in zip(names, nodes, supports))
  g.__doc__ = """\
      Negative energy function on statistics.\n
      Args:
        {arg_docs}\n""".format(arg_docs="\n        ".join(arg_docs))
  return g


def make_fun(fun, **kwargs):
  code = fun.func_code
  attr_names = ['argcount', 'nlocals', 'stacksize', 'flags', 'code',
                'consts', 'names', 'varnames', 'filename', 'name',
                'firstlineno', 'lnotab', 'freevars', 'cellvars']
  new_code = CodeType(*(kwargs.get(name, getattr(code, 'co_' + name))
                        for name in attr_names))
  return FunctionType(new_code, fun.func_globals, closure=fun.func_closure)


def _stat_nodes_to_key(nodes, free_node):
  summaries = (_summarize_node(node, free_node) for node in nodes)
  key, _ = canonical_statistic_strs(summaries)
  return key


def _get_natural_parameter_factory(natural_parameter_funs, support):
  key, order = canonical_statistic_strs(natural_parameter_funs.keys())
  if key not in suff_stat_to_dist[support]:
    raise NotImplementedError('Conditional distribution has sufficient '
                              'statistics {}, but no available '
                              'exponential-family distribution with support '
                              '{} has those sufficient statistics.'.format(
                                  key, support_type_to_name[support]))
  def natural_parameter_factory(*args):
    natural_parameters = [f(*args) for f in natural_parameter_funs.values()]
    natural_parameters = [natural_parameters[i] for i in order]
    return key, natural_parameters
  return natural_parameter_factory


def _extract_conditional_factors(log_joint_node, wrt_argid):
  """Determines a node's natural parameters and sufficient statistics.

  If `log_joint_node` defines a log-joint distribution of some random
  variables (including `wrt_node`), and the conditional distribution
  p(wrt_node | everything else) is in an exponential family, then this
  function will try to determine the sufficient statistics and natural
  parameters of that exponential family distribution.

  Args:
    log_joint_node: An ExprNode that computes a log-joint function.
    wrt_argname: string or int indicating which input to the log-joint is of the
      random variable of interest.

  Returns:
    natural_parameter_funs: A dictionary that maps from a string describing
      a sufficient-statistic function to a function f(*args) that takes the
      same arguments as the log-joint and computes the natural parameter array
      that corresponds to the sufficient statistic key.
  """
  if isinstance(wrt_argid, str):
    wrt_node = log_joint_node.free_vars[wrt_argid]
  else:
    wrt_node = log_joint_node.free_vars.values()[wrt_argid]
  sufficient_statistic_nodes = find_sufficient_statistic_nodes(log_joint_node,
                                                               wrt_argid)
  natural_parameter_funs = defaultdict(lambda: lambda *args: 0.)
  for sufficient_statistic_node in sufficient_statistic_nodes:
    grad_fun = grad_expr(log_joint_node, [sufficient_statistic_node],
                         stop_nodes=sufficient_statistic_nodes)
    if sufficient_statistic_node.fun.__name__ == 'einsum':
      argnums = [i for i, arg in enumerate(sufficient_statistic_node.args[1:])
                 if is_descendant_of(arg, wrt_node)]
      _, stat_node, grad_node = (
          split_einsum_node(sufficient_statistic_node, argnums))

      def make_new_grad_fun(grad_fun, grad_node):
        def new_grad_fun(*args):
          env = {key: arg for key, arg in zip(
              log_joint_node.free_vars.keys(), args)}
          return [grad_fun(*args)[0] *
                  eval_node(grad_node, log_joint_node.free_vars, env)]
        return new_grad_fun
      grad_fun = make_new_grad_fun(grad_fun, grad_node)

      statistic_str = _summarize_node(stat_node, wrt_node)
    else:
      statistic_str = _summarize_node(sufficient_statistic_node, wrt_node)

    def make_new_fun(grad_fun, old_fun):
      def new_fun(*args):
        grads = grad_fun(*args)
        old_params = old_fun(*args)
        return grads[0] + old_params
      return new_fun
    # TODO(mhoffman): This may result in redundant computation. It'd
    # be better to build one function that returns all of the natural
    # parameters.
    natural_parameter_funs[statistic_str] = make_new_fun(
        grad_fun, natural_parameter_funs[statistic_str])

  return sufficient_statistic_nodes, natural_parameter_funs


def split_einsum_node(node, stat_argnums):
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
  assert node.fun is np.einsum
  formula = node.args[0]
  assert isinstance(formula, str)
  in_formulas, out_formula = split_einsum_formula(formula)
  assert not out_formula

  def shape(arg):
    if hasattr(arg, 'vs'):
      return arg.vs.shape
    elif hasattr(arg, 'shape'):
      return arg.shape
    else:
      return (0,)

  dim_lengths = {}
  for formula, arg in zip(in_formulas, node.args[1:]):
    for index, length in zip(formula, shape(arg)):
      dim_lengths[index] = length

  param_argnums = [i for i in range(len(in_formulas)) if i not in stat_argnums]

  stat_args = [arg_i for i, arg_i in enumerate(node.args[1:])
               if i in stat_argnums]
  param_args = [arg_i for i, arg_i in enumerate(node.args[1:])
                if i in param_argnums]

  stat_inputs = ','.join([input_i for i, input_i in enumerate(in_formulas)
                          if i in stat_argnums])
  param_inputs = ','.join([input_i for i, input_i in enumerate(in_formulas)
                           if i in param_argnums])

  # Using an OrderedDict as an OrderedSet to get the unique output indexes.
  output_indexes = ''.join(OrderedDict(zip(stat_inputs.replace(',', ''),
                                           stat_inputs)).keys())
  ones_indexes = [c for c in output_indexes if c not in param_inputs]
  ones_args = [np.ones([dim_lengths[index]]) for index in ones_indexes]
  grad_inputs = ','.join(param_inputs.split(',') + ones_indexes)

  stat_formula = '{}->{}'.format(stat_inputs, output_indexes)
  potential_formula = '{},{}->'.format(param_inputs, output_indexes)
  grad_formula = '{}->{}'.format(grad_inputs, output_indexes)

  stat_node = make_node(np.einsum, [stat_formula] + stat_args, node.kwargs)
  grad_node = make_node(np.einsum, [grad_formula] + param_args + ones_args,
                        node.kwargs)
  return make_node(np.einsum, [potential_formula] + param_args + [stat_node],
                   node.kwargs), stat_node, grad_node


def split_einsum_node2(node, stat_argnums, canonicalize=True):
  """Like split_einsum_node but doesn't produce grad_node."""
  formula = node.args[0]
  assert isinstance(formula, str), "Must use string-formula form of einsum."
  in_formulas, out_formula = split_einsum_formula(formula)
  in_formulas = [formula.lstrip('...') for formula in in_formulas]
  out_formula = out_formula.lstrip('...')
  assert not out_formula, "Must contract to a scalar."

  param_argnums = [i for i, _ in enumerate(in_formulas) if i not in stat_argnums]

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
