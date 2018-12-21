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
"""Tracers to build expressions as computation graphs on free variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import funcsigs
import functools
import hashlib
import inspect
import itertools
from numbers import Number
from types import FunctionType, CodeType
import warnings

# this import has side-effects: it registers new Boxes/VSpaces with Autograd
from . import util

import autograd.core as ag_core
from autograd.core import vspace
from autograd.extend import defvjp
from autograd.extend import defvjp_argnums
from autograd.extend import primitive
from autograd.extend import notrace_primitive
from autograd.extend import register_notrace
from autograd.numpy.numpy_vjps import unbroadcast
import autograd.tracer as tracer
from autograd.tracer import getval
from autograd.tracer import Node
from autograd.util import toposort
from autograd.util import subvals
from autograd import core
from autograd import grad
from autograd import numpy as np


# Expression types
GraphExpr = collections.namedtuple('GraphExpr', ['expr_node', 'free_vars'])
ConstExpr = collections.namedtuple('ConstExpr', ['val', 'free_vars'])


def trace(fun, start_nodes, args):
  with tracer.trace_stack.new_trace() as t:
    start_boxes = [tracer.new_box(x, t, n) for x, n in zip(args, start_nodes)]
    end_box = fun(*start_boxes)
    if tracer.isbox(end_box) and end_box._trace == t:
      return end_box._value, end_box._node
    else:
      warnings.warn("Output seems independent of input.")
      return end_box, None


def make_expr(fun, *args, **kwargs):
  """Trace a function's execution to a representation of its body expression.

  Args:
    fun: a Python callable that only requires positional arguments.
    args: positional argument values on which to trace the evaluation of fun.
    names: optional, a list of names for free variables corresponding to the
      positional arguments of fun (the default is to use variable names
      corresponding to the parameter names of fun, or x0, x1, ... if parameter
      names of fun cannot be determined).

  Returns:
    An expression instance representing the body of fun (with all non-primitive
    function calls inlined) with free variables corresponding to the positional
    arguments that affect its value.
  """
  names = kwargs.pop('names', getargs(fun) or _default_names())
  start_nodes = [ExprNode.new_root(name, arg) for name, arg in zip(names, args)]
  val, end_node = trace(fun, start_nodes, args)
  if end_node:
    used_start_nodes = {n for n in toposort(end_node, lambda n: n.parents)
                        if n in start_nodes}
    free_vars = collections.OrderedDict((n.name, n) for n in start_nodes
                                        if n in used_start_nodes)
    return GraphExpr(end_node, free_vars)
  else:
    return ConstExpr(val, {})


class ExprNode(Node):
  """Node type used in GraphExpr internal representation."""
  __slots__ = ['fun', 'args', 'kwargs', 'vs']

  def __init__(self, ans, fun, args, kwargs, parent_argnums, parents):
    self.fun = fun
    self.args = list(subvals(args, zip(parent_argnums, parents)))
    self.kwargs = kwargs
    self.vs = vspace(ans)

  def initialize_root(self, var_name, val):
    self.fun = env_lookup
    self.args = (var_name,)
    self.kwargs = {}
    self.vs = vspace(val)

  @property
  def parents(self):
    return [arg for arg in self.args if isinstance(arg, ExprNode)]

  @property
  def parent_argnums(self):
    return [i for i, arg in enumerate(self.args) if isinstance(arg, ExprNode)]

  @property
  def name(self):
    if self.fun is not env_lookup:
      raise AttributeError("ExprNode only has 'name' for env_lookup nodes.")
    return self.args[0]

  def __eq__(self, x):
    return (isinstance(x, ExprNode)
            and self.fun == x.fun
            and len(self.args) == len(x.args)
            and all(map(equal, self.args, x.args))
            and set(self.kwargs) == set(x.kwargs)
            and all(equal(self.kwargs[k], x.kwargs[k]) for k in self.kwargs))

  def __repr__(self):
    node_name = self.fun.__name__
    if self.fun is env_lookup:
      node_name += '(' + self.name + ')'
    return '<ExprNode {}({}) {}>'.format(
        node_name, ', '.join(['-'] * len(self.args)), hex(id(self)))


def env_lookup(env, var_name):
  """Function used by 'source' nodes in ExprNode graphs to model var lookup."""
  try:
    return env[var_name]
  except KeyError:
    raise NameError("Name '{}' is not defined in environment with names {}"
                    .format(var_name, env.keys()))


def _value_hash(o):
  try:
    return hash(o)  # NOTE can collide, e.g. hash(-1) == hash(-2)
  except TypeError:
    if isinstance(o, np.ndarray):
      return hash(hashlib.sha1(np.require(o, requirements='C')).hexdigest())
    else:
      return id(o)  # give up, hash object by id


class _CSEValue(object):
  __slots__ = ['fun', 'args', 'kwargs']

  def __init__(self, fun, args, kwargs):
    self.fun = fun
    self.args = args
    self.kwargs = kwargs

  def __hash__(self):
    fun, args, kwargs = self.fun, self.args, self.kwargs
    try:
      bound = funcsigs.signature(fun).bind(*args, **kwargs)
    except (ValueError, TypeError):
      pass  # can't use signature to match kwargs to args, use generic approach
    else:
      args, kwargs = bound.args, bound.kwargs
    arg_hash = (_value_hash(arg) for arg in args)
    kwarg_hash = (hash(k) ^ _value_hash(v) for k, v in kwargs.iteritems())
    return hash((fun,) + tuple(itertools.chain(arg_hash, kwarg_hash)))

  def __eq__(self, x):
    return (type(x) is _CSEValue
            and self.fun == x.fun
            and len(self.args) == len(x.args)
            and all(map(equal, self.args, x.args))
            and set(self.kwargs) == set(x.kwargs)
            and all(equal(self.kwargs[k], x.kwargs[k]) for k in self.kwargs))


def _memoize_apply_node(apply_node):
  memoized_vals = {}

  def memoized_apply_node(node, args):
    node_hash = _CSEValue(node.fun, args, node.kwargs)
    if node_hash not in memoized_vals:
      memoized_vals[node_hash] = apply_node(node, args)
    return memoized_vals[node_hash]

  return memoized_apply_node


def _eval_graph(root_node, eval_args, apply_node, cse=True):
  vals = {}
  apply_node = _memoize_apply_node(apply_node) if cse else apply_node
  for node in reversed(list(toposort(root_node))):
    args = eval_args(node, (vals[p] for p in node.parents))
    vals[node] = apply_node(node, args)
  return vals[root_node]

def node_fmap(f, xs):
  if isinstance(xs, ExprNode):
    return f(xs)
  elif isinstance(xs, (list, tuple)):
    elts = [node_fmap(f, elt) for elt in xs]
    # assume there are no tuple subclasses other than namedtuples
    if type(xs) in (list, tuple):
      return type(xs)(elts)
    else:
      return type(xs)(*elts)  # namedtuple
  elif isinstance(xs, dict):
    return {k: node_fmap(f, v) for k, v in xs.items()}
  else:
    return xs

class ContainerOutput(object):
  def __init__(self, container):
    self.container = container

  def __hash__(self):
    return id(self.container)  # unique value

  @property
  def parents(self):
    nodes = set()
    node_fmap(nodes.add, self.container)
    return nodes

def _eval_graph_container(root_container, eval_args, apply_node, cse=True):
  # This function exists to handle root nodes that are container types.
  graph = list(toposort(ContainerOutput(root_container)))[::-1]

  vals = {}
  apply_node = _memoize_apply_node(apply_node) if cse else apply_node
  for node in graph[:-1]:
    args = eval_args(node, (vals[p] for p in node.parents))
    vals[node] = apply_node(node, args)
  out = node_fmap(vals.get, root_container)
  import pdb; pdb.set_trace()
  return out


def eval_expr(expr, env={}):
  """Evaluate an expression in a given environment.

  Args:
    expr: an expression instance.
    env: a dict of name:value bindings, where keys can be strings corresponding
      to free variable names (mapping to variable values), functions (mapping to
      replacement functions to apply), or ExprNodes (mapping to values).

  Returns:
    The value of the expression given the environment.

  Raises:
    NameError if an env_lookup node is encountered without a corresponding name
    binding in env.
  """
  if isinstance(expr, ConstExpr):
    return expr.val
  elif isinstance(expr, GraphExpr):
    def eval_args(node, partial_args):
      return subvals(node.args, zip(node.parent_argnums, partial_args))
    def apply_node(node, node_args):
      fun = env.get(node.fun, node.fun)
      if node in env:
        return env[node]
      if node.fun is env_lookup:
        return fun(env, *node_args, **node.kwargs)
      else:
        return fun(*node_args, **node.kwargs)
    if isinstance(expr.expr_node, tuple):
      return _eval_graph_container(expr.expr_node, eval_args, apply_node)
    else:
      return _eval_graph(expr.expr_node, eval_args, apply_node)
  else:
    raise TypeError("Can't evaluate expression type: {}".format(type(expr)))


def eval_node(node, free_vars, env):
  return eval_expr(GraphExpr(node, free_vars), env)


def backward_pass(g, start_nodes, end_node):
    outgrads = {end_node : (g, False)}
    for node in ag_core.toposort(end_node):
        outgrad = outgrads[node]
        ingrads = node.vjp(outgrad[0])
        for parent, ingrad in zip(node.parents, ingrads):
            outgrads[parent] = ag_core.add_outgrads(outgrads.get(parent), ingrad)
    return [outgrads.pop(node, (None, None))[0] for node in start_nodes]


def make_dummy(node):
  result = node.vs.ones()
  if len(result.shape) >= 2 and result.shape[-1] == result.shape[-2]:
    result = np.ones(result.shape[:-2] + (1, 1)) * np.eye(result.shape[-1])
  return result


# TODO(mattjj): This function re-evals on dummy values, but that's wasteful.
# If we tweak how eager simplifications work, we can avoid FLOPs here.
def remake_expr(expr, env={}):
  """Convenience wrapper for make_expr/eval_expr to apply eager simplifies."""
  # this function doesn't eliminate unused free_vars in expr, but it could
  names = expr.free_vars.keys()
  args = (make_dummy(node) for node in expr.free_vars.values())
  return make_expr(lambda *args: eval_expr(expr, dict(zip(names, args), **env)),
                   *args, names=names)


def inline_expr(expr, symbolic_env):
  """Evaluates expr in a symbolic environment for substituting subgraphs."""
  if isinstance(expr, ConstExpr):
    return expr
  elif isinstance(expr, GraphExpr):
    def eval_args(node, partial_args):
      return subvals(node.args, zip(node.parent_argnums, partial_args))
    def apply_node(node, node_args):
      if node.fun is env_lookup:
        return node.fun(symbolic_env, *node_args, **node.kwargs)
      else:
        return _make_node_like(node, args=node_args)
    expr_node = _eval_graph(expr.expr_node, eval_args, apply_node)
    return GraphExpr(expr_node, {})
  else:
    raise TypeError("Can't inline expression type: {}".format(type(expr)))


def replace_node_with_expr(node, expr):
  """Replaces an ExprNode (in a GraphExpr) with a given expression."""
  if isinstance(expr, ConstExpr):
    val = expr.val
    temp_node = ExprNode(val, lambda: val, (), {}, (), ())
    _mutate_node(node, temp_node)
  elif isinstance(expr, GraphExpr):
    _mutate_node(node, expr.expr_node)
  else:
    raise TypeError("Can't handle expression type: {}".format(type(expr)))
  return node


def _mutate_node(target_node, source_node, **kwargs):
  for attrname in target_node.__slots__:
    attrval = kwargs.get(attrname, getattr(source_node, attrname))
    setattr(target_node, attrname, attrval)
  return target_node


def _make_node_like(node, **kwargs):
  return _mutate_node(ExprNode.__new__(ExprNode), node, **kwargs)


def _default_names():
  return itertools.imap(lambda i: "x{}".format(i), itertools.count())


# TODO(mattjj): this should allow repeated names... should write a new function
# remake_with_new_free_vars that takes a dict mapping nodes to names
def extract_superexpr(expr, nodes):
  """Extract a super-expression on the given nodes (and other free vars).

  Args:
    expr: a GraphExpr instance.
    nodes: dict mapping name strings to ExprNodes in GraphExpr.

  Returns:
    A new expression, with free variables drawn from the keys of `nodes` (and
    any remaining free variables of `expr`), corresponding to the body of the
    function from `nodes` (and any other free variables) to the value of `expr`.
  """
  names = expr.free_vars.keys()
  args = (node.vs.ones() for node in expr.free_vars.values())
  env_vals = (node.vs.ones() for node in nodes.values())

  def fun(*args_and_env):
    N = len(expr.free_vars)
    args, env_vals = args_and_env[:N], args_and_env[N:]
    env = dict(zip(nodes.values(), env_vals))
    return eval_expr(expr, dict(zip(names, args), **env))

  return make_expr(fun, *itertools.chain(args, env_vals),
                   names=itertools.chain(names, nodes))


## util


def getargs(fun):
  try:
    return inspect.getargspec(fun).args
  except TypeError:
    pass


@notrace_primitive
def equal(a, b):
  """An equality function that compares all elements of ndarrays."""
  try:
    return bool(a == b)
  except ValueError:
    return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            and np.shape(a) == np.shape(b) and (a == b).all())


def make_node(fun, args, kwargs):
  # infer shape data by running fun on dummy arguments
  argvals = [arg.vs.ones() if isinstance(arg, ExprNode) else arg
             for arg in args]
  vs = vspace(fun(*argvals, **kwargs))

  new_node = ExprNode.__new__(ExprNode)
  new_node.fun = fun
  new_node.args = list(args)
  new_node.kwargs = kwargs
  new_node.vs = vs
  return new_node


def is_descendant_of(a, b):
  """Test if the object a is a descendant of the ExprNode b."""
  if not isinstance(b, ExprNode):
    raise TypeError("Second argument must be ExprNode, got {}".format(type(b)))
  if a is b:
    return True
  visited = set()
  def is_descendant_of_b(a):
    visited.add(a)
    return b in a.parents or any(p not in visited and is_descendant_of_b(p)
                                 for p in a.parents)
  return isinstance(a, ExprNode) and is_descendant_of_b(a)


def all_descendants_of(root_node, ancestor):
  """Return a set of all descendants of `ancestor` in the graph `root_node`."""
  visited = set([ancestor])
  descendants = set([ancestor])
  def collect_descendants(node):
    if node not in visited:
      visited.add(node)
      if node is ancestor or any([collect_descendants(p) or p in descendants
                                  for p in node.parents]):
        descendants.add(node)
  collect_descendants(root_node)
  return frozenset(descendants)


@primitive
def one_hot(x, width):
  """Convert int array-like x to a one-hot representation of given width."""
  return (np.expand_dims(x, -1) == np.arange(width)).astype(np.float32)
defvjp(one_hot, None)


@primitive
def logdet(x):
  return np.linalg.slogdet(x)[1]
# transpose by swapping last two dimensions
def _T(x): return np.swapaxes(x, -1, -2)
# add two dimensions to the end of x
def _add2d(x): return np.reshape(x, np.shape(x) + (1, 1))
defvjp(logdet, lambda ans, x: lambda g: _add2d(g) * _T(np.linalg.inv(x)))


@primitive
def add_n(*args):
  return reduce(np.add, args)
def grad_add_n_full(parent_argnums, ans, args, kwargs):
  meta = [np.metadata(args[i]) for i in parent_argnums]
  return lambda g: [unbroadcast(g, m) for m in meta]
defvjp_argnums(add_n, grad_add_n_full)


## debugging


def print_expr(expr, env={}):
  """Return a string with an SSA-like representation of an expression."""
  if isinstance(expr, ConstExpr):
    return str(expr.val)
  elif isinstance(expr, GraphExpr):
    fragment = []
    temp_names = ('temp_{}'.format(i) for i in itertools.count())
    apply_str = '{} = {}({})\n'.format
    def eval_args(node, partial_args):
      args = subvals(node.args, zip(node.parent_argnums, partial_args))
      return [str(a) for a in args]
    def apply_node(node, arg_strs):
      if node.fun is env_lookup:
        name, = arg_strs
        return name if name not in env else env[name]
      else:
        name = next(temp_names)
        fragment.append(apply_str(name, node.fun.__name__, ', '.join(arg_strs)))
        return name
    out_name = _eval_graph(expr.expr_node, eval_args, apply_node, cse=False)
    return ''.join(fragment)
  else:
    raise TypeError("Can't print expression type: {}".format(type(expr)))


dot_edge = '{} -> {} [color=gray30];\n'.format
dot_function_node = (
    '{} [label="{}", shape=box, color=lightblue, style=filled];\n'.format)
dot_variable_node = '{} [label="{}", color=orange, style=filled];\n'.format
dot_graph = 'digraph G {{{}}}'.format


def draw_expr(expr, env={}):
  if isinstance(expr, GraphExpr):
    fragment = ['']
    temp_names = ('temp_{}'.format(i) for i in itertools.count())
    node_names = collections.defaultdict(iter(temp_names).next)
    def eval_args(node, partial_args):
      return subvals(node.args, zip(node.parent_argnums, partial_args))
    # TODO print out einsum nodes, string vs float vs input nodes in different color
    def apply_node(node, arg_strs):
      if node.fun is env_lookup:
        name, = arg_strs
        name = node_names[node] = name if name not in env else env[name]
        fragment[0] += dot_variable_node(name, name)
      else:
        name = node_names[node]
        fragment[0] += dot_function_node(name, node.fun.__name__)
        for argnum, arg in enumerate(node.args):
          if argnum in node.parent_argnums:
            fragment[0] += dot_edge(node_names[node.args[argnum]], name)
          else:
            argname = '{}_arg_{}'.format(name, argnum)
            fragment[0] += dot_edge(argname, name)
            fragment[0] += dot_variable_node(argname, arg)
      return name
    name = _eval_graph(expr.expr_node, eval_args, apply_node, cse=False)
    fragment[0] += dot_variable_node('output', 'output')
    fragment[0] += dot_edge(name, 'output')
    return dot_graph(fragment[0])
  else:
    raise TypeError("Can't draw expression type: {}".format(type(expr)))


notrace_functions = [np.ones_like, np.zeros_like]
for fun in notrace_functions:
  register_notrace(ExprNode, fun)
