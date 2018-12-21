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
"""Pattern definitions for use with matcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import autograd.extend as ag_extend
import autograd.util as ag_util
from autograd.numpy.numpy_vspaces import ArrayVSpace
import autograd.numpy as np
import autograd.numpy.numpy_boxes as np_boxes

from . import tracers

## util

def _point_free_logical(op):
  def make_checker(*predicates):
    def check(x):
      return op(pred(x) for pred in predicates)
    pred_names = (pred.__name__ for pred in predicates)
    check.__name__ = '{}({})'.format(op.__name__, ', '.join(pred_names))
    return check
  return make_checker

_or = _point_free_logical(any)
_and = _point_free_logical(all)

## predicates for testing types of literals and nodes in our graphs

def is_node(x): return isinstance(x, tracers.ExprNode)

def is_array_literal(x): return isinstance(x, np.ndarray)
def is_array_node(x): return is_node(x) and isinstance(x.vs, ArrayVSpace)
is_array = _or(is_array_literal, is_array_node)

def is_scalar_literal(x): return np.isscalar(x)
def is_scalar_node(x):
  return is_node(x) and isinstance(x.vs, ArrayVSpace) and x.vs.shape == ()
is_scalar = _or(is_scalar_literal, is_scalar_node)

def is_tuple_literal(x): return isinstance(x, tuple)
is_tuple = is_tuple_literal

def is_list_literal(x): return isinstance(x, list)
is_list = is_list_literal

def is_dict_literal(x): return isinstance(x, dict)
is_dict = is_dict_literal

def is_string_literal(x): return isinstance(x, str)
is_string = is_string_literal

## patterns

def _make_convenience_pattern(*preds):
  def make_pattern(name=None, extra_pred=None):
    all_preds = preds + (extra_pred,) if extra_pred else preds
    return ('?', name, all_preds)
  return make_pattern

Any = _make_convenience_pattern()
Array = _make_convenience_pattern(is_array)
Node = _make_convenience_pattern(is_array_node)
Str = _make_convenience_pattern(is_string)
Scalar = _make_convenience_pattern(is_scalar)
Val = _make_convenience_pattern(_or(is_array, is_scalar))
Tuple = _make_convenience_pattern(is_tuple)
List = _make_convenience_pattern(is_list)
Dict = _make_convenience_pattern(is_dict)

# generate a pattern for each Autograd primitive

def _make_primitive_checker(name):
  def check_node_name(node):
    return is_node(node) and node.fun.__name__ == name
  return check_node_name

def _make_primitive_pattern(fun, pattern_name):
  def pat_maker(name=None):
    return ('?', name,
            (_make_primitive_checker(fun.__name__),),
            lambda node: node.fun)
  pat_maker.__name__ = pattern_name
  return pat_maker

def _import_primitives_no_clobber(new, old):
  def is_primitive(fun): return callable(fun) and hasattr(fun, 'fun')
  for name, obj in old.items():
    titlecase_name = ''.join(word.title() for word in name.split('_'))
    if is_primitive(obj) and titlecase_name not in new:
      new[titlecase_name] = _make_primitive_pattern(obj, titlecase_name)

_import_primitives_no_clobber(globals(), np.__dict__)
_import_primitives_no_clobber(globals(), np.linalg.__dict__)
_import_primitives_no_clobber(globals(), np_boxes.ArrayBox.__dict__)
_import_primitives_no_clobber(globals(), {'add_n': tracers.add_n})
_import_primitives_no_clobber(globals(), {'logdet': tracers.logdet})
_import_primitives_no_clobber(globals(), {'one_hot': tracers.one_hot})
EnvLookup = _make_primitive_pattern(tracers.env_lookup, 'EnvLookup')


## patterns for Autograd internals

def _is_vspace_add(node):
  return (node.fun is ag_util.func(ag_extend.VSpace.add) or
          node.fun is ag_util.func(ag_extend.VSpace.mut_add))


def VSpaceAdd(name=None):
  return ('?', name, (_is_vspace_add,))

## convenience combinators that operate on patterns

def Choice(*alternatives): return ('?:choice',) + alternatives
def List(*list_elements): return ('List',) + list_elements
def Segment(name=None): return Star(Any, name)
def Star(pat, name=None, accumulate=[]): return ('??', name, pat, accumulate)
def Not(pattern): return ('?:not', pattern)
