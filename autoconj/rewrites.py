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

import collections
import functools
import itertools
import operator

import autograd.extend as ag_extend
import autograd.numpy as np
import autograd.numpy.numpy_vspaces as numpy_vspaces
import autograd.tracer as ag_tracer
import funcsigs

from .patterns import (Subtract, Add, Dot, Multiply, Divide, TrueDivide, Node,
                       Val, Einsum, Str, Choice, Segment, Log, Sum, Tuple,
                       VSpaceAdd, Any, Power, Scalar, OneHot, Transpose, Inv,
                       Logdet, AddN, Star)
from .tracers import add_n
from .tracers import logdet
from .tracers import make_dummy
from .tracers import subvals
from .util import split_einsum_formula
from . import matchers
from . import patterns
from . import tracers


_einsum_range = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_einsum_index_set = frozenset(_einsum_range)


### eager rewrites replace individual functions with constant-folding versions


def is_constant(x):
  return not ag_tracer.isbox(x)


def _is_constant_val(x, val):
  return is_constant(x) and np.all(x == val)


_is_constant_zero = functools.partial(_is_constant_val, val=0.)
_is_constant_one = functools.partial(_is_constant_val, val=1.)


def _multiply_as_einsum(x, y):
  x_arr, y_arr = np.array(x), np.array(y)
  new_shape = np.broadcast(x_arr, y_arr).shape
  out_formula = _einsum_range[:len(new_shape)]
  next_index = iter(_einsum_range[len(new_shape):])

  def _make_broadcast_formula(z):
    offset = len(new_shape) - len(z.shape)
    return ''.join([out_formula[offset + i]
                    if z.shape[i] == new_shape[offset + i]
                    else next_index.next()
                    for i in range(len(z.shape))])
  new_formula = '{},{}->{}'.format(_make_broadcast_formula(x_arr),
                                   _make_broadcast_formula(y_arr),
                                   out_formula)
  return np.einsum(new_formula, x, y)


def maybe_multiply(x, y):
  if _is_constant_zero(x) or _is_constant_zero(y):
    return np.zeros(np.broadcast(x, y).shape, dtype=np.result_type(x, y))
  if _is_constant_one(x) and np.shape(y) == np.broadcast(x, y).shape:
    return y
  if _is_constant_one(y) and np.shape(x) == np.broadcast(x, y).shape:
    return x
  return _multiply_as_einsum(x, y)


def maybe_add(x, y):
  if _is_constant_zero(x) and np.shape(y) == np.broadcast(x, y).shape:
    return y
  if _is_constant_zero(y) and np.shape(x) == np.broadcast(x, y).shape:
    return x
  return add_n(x, y)


def maybe_subtract(x, y):
  if _is_constant_zero(y) and np.shape(x) == np.broadcast(x, y).shape:
    return x
  return add_n(x, _multiply_as_einsum(-1, y))


def maybe_getitem(x, idx):
  if isinstance(idx, slice):
    return list(x)[idx]
  else:
    return x[idx]


def dot_as_einsum(x, y):
  if x.ndim == 0 or y.ndim == 0: return np.einsum(',->', x, y)
  if x.ndim == y.ndim == 1: return np.einsum('i,i->', x, y)
  if x.ndim == 2 and y.ndim == 1: return np.einsum('ij,j->i', x, y)
  if x.ndim == 1 and y.ndim == 2: return np.einsum('i,ij->j', x, y)
  return np.einsum('{}ab,{}bc->{}ac'.format(
      _einsum_range[:x.ndim-2][::-1], _einsum_range[:y.ndim-2][::-1],
      _einsum_range[:max([x.ndim, y.ndim])-2][::-1]), x, y)


def maybe_divide(x, y):
  if _is_constant_one(y) and np.shape(x) == np.broadcast(x, y).shape:
    return x
  elif _is_constant_one(x) and np.shape(y) == np.broadcast(x, y).shape:
    return y ** -1
  return _multiply_as_einsum(x, y ** -1)


# TODO(mhoffman): Consider exponent == 0. E.g., what if base could also be 0?
def maybe_power(base, exponent):
  if exponent == 1:
    return base
  elif exponent == 0:
    return 1
  elif isinstance(exponent, int) and exponent > 0 and exponent < 10:
    formula = ''.join([_einsum_range[i] for i in range(len(base.shape))])
    in_formulas = [formula for _ in range(exponent)]
    out_formula = formula
    formula = _reconstitute_einsum_formula(in_formulas, out_formula)
    args = [base for _ in range(exponent)]
    return np.einsum(formula, *args)
  else:
    return base ** exponent


def _rename_formula_indices(formula):
  """Renames einsum formula indices to be in a canonical order."""
  # First, ensure that indices are packed.
  translation_dict = {index: _einsum_range[i] for i, index in
                      enumerate(np.unique([index for index in formula
                                           if index in _einsum_index_set]))}
  translator = lambda x: translation_dict[x] if x in translation_dict else x
  formula = [translator(i) for i in formula]
  # Next, ensure that they're alphabetical in order of appearance.
  translation_dict = {}
  for index in formula:
    if index not in translation_dict and index in _einsum_index_set:
      translation_dict[index] = _einsum_range[len(translation_dict)]
  return ''.join([translator(i) for i in formula])


def debroadcast_formula(formula, *arg_ndims):
  """Given an einsum's formula string and the dimensions of the arguments
  provided to the einsum, converts any broadcasting ellipses into appropriate
  letters.
  """
  formula = _rename_formula_indices(formula)
  num_chars = len(_einsum_index_set.intersection(set(formula)))
  remaining_letters = _einsum_range[num_chars:]
  in_formulas, out_formula = split_einsum_formula(formula)

  max_ellipsis_dims = -float('inf')
  for i, in_formula in enumerate(in_formulas):
    in_formula = decompose_formula(in_formula)
    if '...' in in_formula:
      num_ellipsis_dims = arg_ndims[i]-len(in_formula)+1
      max_ellipsis_dims = max(max_ellipsis_dims, num_ellipsis_dims)
      ellipsis_idx = in_formula.index('...')
      in_formula[ellipsis_idx] = remaining_letters[:num_ellipsis_dims][::-1]
    in_formulas[i] = ''.join(in_formula)

  if '...' in out_formula:
    out_formula = out_formula.replace(
        '...', remaining_letters[:max_ellipsis_dims][::-1])

  new_formula = _reconstitute_einsum_formula(in_formulas, out_formula)
  return _rename_formula_indices(new_formula)


def _zeros_like_einsum(formula, args1, args2):
  args = args1 + args2
  input_formulas, output_formula = split_einsum_formula(formula)
  output_formula = decompose_formula(output_formula)
  input_formulas = input_formulas[:len(args1)] + input_formulas[len(args1)+1:]
  input_formulas = [decompose_formula(input_formula) for
                    input_formula in input_formulas]

  out_shape = []
  for output_index in output_formula:
    for i, input_formula in enumerate(input_formulas):
      position = input_formula.index(output_index)
      if position != -1 and output_index != '...':
        out_shape.append(args[i].shape[position])
        break
      elif position != -1 and output_index == '...':
        for offset in range(args[i].ndim-len(input_formula)+1):
          out_shape.append(args[i].shape[position+offset])
  return np.zeros(out_shape, dtype=np.result_type(*args))


def maybe_einsum(formula, *args):
  formula = debroadcast_formula(formula, *[np.ndim(arg) for arg in args])

  if any(_is_constant_zero(arg) for arg in args):
    return _zeros_like_einsum(formula, args, ())
  if len(args) == 1:
    input_formulas, output_formula = split_einsum_formula(formula)
    if input_formulas[0] == output_formula:
      return args[0]
  return constant_folding_einsum(formula, *args)


def maybe_vspace_add(vs, x_prev, x_new):
  if x_prev is None:
    return x_new
  if isinstance(vs, numpy_vspaces.ArrayVSpace):
    return maybe_add(x_prev, x_new)
  return vs.add(x_prev, x_new)


def swapaxes(x, axis1, axis2):
  """Implements np.swapaxes as an np.einsum."""
  in_formula = _einsum_range[:len(x.shape)]
  out_formula = list(in_formula)
  out_formula[axis1] = in_formula[axis2]
  out_formula[axis2] = in_formula[axis1]
  return np.einsum('{}->{}'.format(in_formula, ''.join(out_formula)), x)


### rewrite rules replace whole subgraphs with other subgraphs


class Rule(collections.namedtuple('BasicRule',
                                  ['pattern', 'rewriter', 'preds'])):

  def __new__(cls, pattern, rewriter, preds=()):
    return super(Rule, cls).__new__(cls, pattern, rewriter, preds)


_add_pattern = Choice((Add, Val('x'), (Add, Val('y'), Val('z'))),
                      (Add, (Add, Val('x'), Val('y')), Val('z')))
replace_add = Rule(_add_pattern, lambda x, y, z: add_n(x, y, z))

_add_addn_pattern = Choice((Add, Val('x'), (AddN, Segment('args'))),
                           (Add, (AddN, Segment('args')), Val('x')))
replace_add_addn = Rule(_add_addn_pattern,
                        lambda x, args: add_n(x, *args))

_addn_addn_pattern = (AddN,
                      Segment('args1'),
                      (AddN, Segment('parent_args')),
                      Segment('args2'))
replace_addn_addn = Rule(
    _addn_addn_pattern,
    lambda args1, parent_args, args2: add_n(*(parent_args + args1 + args2)))


def _duplicated_addn(x, args1, args2, args3):
  return add_n(2 * x, *(args1 + args2 + args3))
_duplicated_addn_pattern = (AddN,
                            Segment('args1'),
                            Val('x'),
                            Segment('args2'),
                            Val('x'),
                            Segment('args3'))
replace_duplicated_addn = Rule(_duplicated_addn_pattern, _duplicated_addn)

# TODO(mattjj): figure out why we want sums as einsums, since not multiplies
_sum_pat = Choice((Sum, Node('x'), Choice(Val('axis'), Tuple('axis'), None)),
                  (Sum, Node('x')))


def _sum_as_einsum(x, axis=None):
  if axis is None:
    return np.einsum('{}->'.format(_einsum_range[:x.ndim]), x)
  axis = axis if isinstance(axis, (tuple, list)) else [axis]
  input_formula = _einsum_range[:x.ndim]
  axis = [i % x.ndim for i in axis]
  output_formula = ''.join([input_formula[i] for i in range(x.ndim)
                            if i not in axis])
  return np.einsum('{}->{}'.format(input_formula, output_formula), x)


replace_sum = Rule(_sum_pat, _sum_as_einsum)

## move log behind an einsum if the other argument is a onehot
_log_oneh_einsum_pat = (Log,
                        (Einsum, Str('formula'),
                         (OneHot, Node('x'), Scalar('depth')),
                         Val('y')))


def _log_behind_onehot_einsum_pred(formula, x, depth, y):
  """Confirms sum is only over index added by one_hot."""
  # TODO(matthewjmackay): broadcasting support might be needed here
  if '...' in formula:
    return False
  in_formulas, out_formula = split_einsum_formula(formula)
  oneh_index = in_formulas[0][-1]
  other_indices = set([ch for in_formula in in_formulas
                       for ch in in_formula])
  other_indices.remove(oneh_index)
  out_indices = set(out_formula)
  return other_indices == out_indices


def _log_behind_onehot_einsum(formula, x, depth, y):
  return np.einsum(formula, tracers.one_hot(x, depth), np.log(y))

log_behind_onehot_einsum = Rule(_log_oneh_einsum_pat, _log_behind_onehot_einsum,
                                (_log_behind_onehot_einsum_pred,))

## move log-add behind an einsum if the other argument is a onehot
_log_addn_oneh_einsum_pat = (Log,
                             (AddN, Val('x'),
                              (Einsum, Str('formula'), Scalar('scale'),
                               (OneHot, Node('y'), Scalar('depth')),
                               Val('z'))))


def _log_addn_behind_onehot_einsum_pred(x, formula, scale, y, depth, z):
  """Confirms sum is only over index added by one_hot"""
  # TODO(matthewjmackay): broadcasting support might be needed here
  if '...' in formula:
    return False
  in_formulas, out_formula = split_einsum_formula(formula)
  oneh_index = in_formulas[1][-1]
  other_indices = set([ch for in_formula in in_formulas
                       for ch in in_formula])
  other_indices.remove(oneh_index)
  out_indices = set(out_formula)
  return other_indices == out_indices


def _log_addn_behind_onehot_einsum(x, formula, scale, y, depth, z):
  in_formulas, out_formula = split_einsum_formula(formula)
  in_formulas = in_formulas[1:]
  formula = _reconstitute_einsum_formula(in_formulas, out_formula)
  return np.einsum(formula,
                   tracers.one_hot(y, depth),
                   np.log(add_n(x, scale*z)))

log_addn_behind_onehot_einsum = Rule(_log_addn_oneh_einsum_pat,
                                     _log_addn_behind_onehot_einsum,
                                     (_log_addn_behind_onehot_einsum_pred,))

## canonicalizing einsums


_einsum_distribute_pat = \
    (Einsum, Str('formula'),
     Segment('args1'),
     (AddN('op'), Segment('add_args')),
     Segment('args2'))


def _distribute_einsum(formula, op, add_args, args1, args2):
  # Make sure any implicit broadcasting isn't lost.
  broadcast_shape = np.broadcast(*add_args).shape
  dtype = np.result_type(*add_args)
  add_args = [arg * np.ones(broadcast_shape, dtype=dtype)
              if not hasattr(arg, 'shape') or broadcast_shape != arg.shape
              else arg
              for arg in add_args]
  return op(*[np.einsum(formula, *(args1 + (arg,) + args2))
              for arg in add_args])


distribute_einsum = Rule(_einsum_distribute_pat, _distribute_einsum)


_einsum_transpose_pat = \
    (Einsum, Str('formula'),
     Segment('args1'),
     (Transpose, Val('x')),
     Segment('args2'))


def _transpose_inside_einsum(formula, args1, x, args2):
  in_formulas, out_formula = split_einsum_formula(formula)
  i = len(args1)
  new_formula = _reconstitute_einsum_formula(
      in_formulas[:i] + [in_formulas[i][::-1]] + in_formulas[i+1:],
      out_formula)
  new_args = args1 + (x,) + args2
  return np.einsum(new_formula, *new_args)


transpose_inside_einsum = Rule(_einsum_transpose_pat, _transpose_inside_einsum)


def _remove_list_elements(list_to_thin, indices_to_remove):
  return [item for i, item in enumerate(list_to_thin)
          if i not in indices_to_remove]


def _remove_einsum_arg(formula, args1, args2):
  in_formulas, out_formula = split_einsum_formula(formula)
  new_formula = _reconstitute_einsum_formula(
      _remove_list_elements(in_formulas, [len(args1)]), out_formula)
  return np.einsum(new_formula, *(args1 + args2))


# Matches things like add_n(x*a, x*b) that can be rewritten as x * add_n(a, b).
_gatherable_add_n_einsum_pat = (
    AddN, Star((Einsum, Str('formula'),
                Segment('args1'), Scalar('x'), Segment('args2')),
               accumulate=['formula', 'args1', 'args2']))


def _add_n_remaining_einsums(formula, args1, args2):
  return add_n(*[_remove_einsum_arg(formula_i, args1_i, args2_i)
                 for formula_i, args1_i, args2_i in zip(formula, args1, args2)])


def _gather_log_add_n_einsum(x, formula, args1, args2):
  return add_n(np.log(x), np.log(_add_n_remaining_einsums(formula, args1, args2)))


gather_log_add_einsum = Rule((Log, _gatherable_add_n_einsum_pat),
                             _gather_log_add_n_einsum)


def _gather_pow_add_n_einsum(x, formula, args1, args2, exponent):
  return (np.power(x, exponent) *
          np.power(_add_n_remaining_einsums(formula, args1, args2), exponent))


gather_pow_add_einsum = Rule(
    (Power, _gatherable_add_n_einsum_pat, Scalar('exponent')),
    _gather_pow_add_n_einsum)


def _gather_inv_add_einsum(x, formula, args1, args2):
  return np.power(x, -1) * np.linalg.inv(_add_n_remaining_einsums(formula, args1, args2))


gather_inv_add_einsum = Rule((Inv, _gatherable_add_n_einsum_pat),
                             _gather_inv_add_einsum)


def _gather_logdet_add_einsum(x, formula, args1, args2):
  new_sum = _add_n_remaining_einsums(formula, args1, args2)
  return new_sum.shape[-1] * np.log(x) + logdet(new_sum)


gather_logdet_add_einsum = Rule((Logdet, _gatherable_add_n_einsum_pat),
                                _gather_logdet_add_einsum)


def _add_powers_within_einsum(formula, x, args1, args2, args3, exponent1,
                              exponent2):
  in_formulas, out_formula = split_einsum_formula(formula)
  new_formula = _reconstitute_einsum_formula(
      _remove_list_elements(in_formulas, [len(args1) + 1 + len(args2)]),
      out_formula)
  return np.einsum(new_formula, *(args1 + (x ** (exponent1 + exponent2),)
                                  + args2 + args3))


def _add_powers_within_einsum_pred(formula, x, args1, args2, args3, exponent1=1,
                                   exponent2=1):
  in_formulas, out_formula = split_einsum_formula(formula)
  x_indices = [len(args1), len(args1) + 1 + len(args2)]
  if in_formulas[x_indices[0]] != in_formulas[x_indices[1]]:
    return False
  x_index_names = frozenset(in_formulas[x_indices[0]] +
                            in_formulas[x_indices[1]])
  if any([not frozenset(in_formula).isdisjoint(x_index_names)
          for i, in_formula in enumerate(in_formulas) if i not in x_indices]):
    return False
  return True


add_powers_within_einsum = Rule((Einsum, Str('formula'), Segment('args1'),
                                 (Power, Val('x'), Scalar('exponent1')),
                                 Segment('args2'),
                                 (Power, Val('x'), Scalar('exponent2')),
                                 Segment('args3')),
                                _add_powers_within_einsum,
                                (_add_powers_within_einsum_pred,))


def _increment_negative_power_in_einsum_r(formula, x, exponent,
                                          args1, args2, args3):
  in_formulas, out_formula = split_einsum_formula(formula)
  new_formula = _reconstitute_einsum_formula(
      in_formulas[:len(args1) + 1 + len(args2)] +
      in_formulas[len(args1) + 2 + len(args2):], out_formula)
  return np.einsum(new_formula,
                   *(args1 + (x ** (exponent + 1),) + args2 + args3))


# TODO(mhoffman): Add predicates that make sure formulas match.
increment_negative_power_in_einsum_r = Rule(
    (Einsum, Str('formula'), Segment('args1'),
     (Power, Node('x'), Scalar('exponent', lambda exponent: exponent < 0)),
     Segment('args2'), Node('x'), Segment('args3')),
    _increment_negative_power_in_einsum_r)


# TODO(mhoffman): Figure out cleaner way of dealing with commuting args.
def _increment_negative_power_in_einsum_l(formula, x, exponent,
                                          args1, args2, args3):
  in_formulas, out_formula = split_einsum_formula(formula)
  new_formula = _reconstitute_einsum_formula(
      in_formulas[:len(args1)] + in_formulas[len(args1) + 1:], out_formula)
  return np.einsum(new_formula,
                   *(args1 + args2 + (x ** (exponent + 1),) + args3))


# TODO(mhoffman): Add predicates that make sure formulas match.
increment_negative_power_in_einsum_l = Rule(
    (Einsum, Str('formula'), Segment('args1'),
     Node('x'), Segment('args2'),
     (Power, Node('x'), Scalar('exponent', lambda exponent: exponent < 0)),
     Segment('args3')),
    _increment_negative_power_in_einsum_l)


_einsum_composition_pat = \
    (Einsum, Str('formula'),
     Segment('args1'),
     (Einsum, Str('parent_formula'), Segment('parent_args')),
     Segment('args2'))


def decompose_formula(formula):
  """Given a string of indices for an argument to an einsum, returns a list
  of the letters used, with '...' treated as an atomic letter.
  """
  formula = formula.replace('...', '.')
  decomposed = []
  for idx in formula:
    if idx == '.':
      decomposed.append('...')
    else:
      decomposed.append(idx)
  return decomposed


def _compose_einsums(formula, args1, args2, parent_formula, parent_args):
  parent_formula = debroadcast_formula(parent_formula,
                                       *[np.ndim(arg) for arg in parent_args])
  parent_in_formulas, parent_out_formula = split_einsum_formula(parent_formula)

  parent_ndim = len(parent_out_formula)
  arg_ndims = ([np.ndim(arg) for arg in args1] +
              [parent_ndim] +
              [np.ndim(arg) for arg in args2])
  formula = debroadcast_formula(formula, *arg_ndims)
  in_formulas, out_formula = split_einsum_formula(formula)

  i = len(args1)
  if len(parent_out_formula) != len(in_formulas[i]):
    raise ValueError('Input formula {} and parent formula {} have'
                     ' inconsistent numbers of indexes, broadcasting'
                     'problem?'.format(in_formulas[i], parent_out_formula))

  subs_map = collections.defaultdict(iter(_einsum_range).next)

  # splice out the old input formula
  old_in_formula = in_formulas[i]
  in_formulas = in_formulas[:i] + in_formulas[i+1:]

  # canonicalize input and output formulas (optional, for cleanliness)
  in_formulas = [''.join(subs_map[idx] for idx in subs) for subs in in_formulas]
  out_formula = ''.join(subs_map[idx] for idx in out_formula)

  # identify parent output indices with corresponding input indices
  subs_map.update((pidx + '_parent', subs_map[idx])
                  for pidx, idx in zip(parent_out_formula, old_in_formula))

  # update the parent input formulas
  parent_in_formulas = [''.join(subs_map[idx + '_parent'] for idx in subs)
                        for subs in parent_in_formulas]

  # splice the formula lists and arguments
  new_in_formulas = in_formulas[:i] + parent_in_formulas + in_formulas[i:]
  new_args = args1 + parent_args + args2

  new_formula = _reconstitute_einsum_formula(new_in_formulas, out_formula)
  return np.einsum(new_formula, *new_args)


combine_einsum_compositions = Rule(_einsum_composition_pat, _compose_einsums)


def _einsum_repeated_one_hot(formula, x, depth, args1, args2, args3):
  in_formulas, out_formula = split_einsum_formula(formula)
  new_letter = in_formulas[len(args1)][-1]
  old_letter = in_formulas[len(args1) + 1 + len(args2)][-1]
  if old_letter in out_formula:
    old_letter, new_letter = new_letter, old_letter
    in_formulas = in_formulas[:len(args1)] + in_formulas[len(args1) + 1:]
  else:
    in_formulas = (in_formulas[:len(args1) + 1 + len(args2)] +
                   in_formulas[len(args1) + 1 + len(args2) + 1:])
  for i in range(len(in_formulas)):
    in_formulas[i] = in_formulas[i].replace(old_letter, new_letter)
  one_hot_x = tracers.one_hot(x, depth)
  return np.einsum(_reconstitute_einsum_formula(in_formulas, out_formula),
                   *(args1 + (one_hot_x,) + args2 + args3))


def _einsum_repeated_one_hot_pred(formula, x, depth, args1, args2, args3):
  in_formulas, out_formula = split_einsum_formula(formula)
  x_letter_1 = in_formulas[len(args1)][-1]
  x_letter_2 = in_formulas[len(args1) + 1 + len(args2)][-1]
  return (x_letter_1 != x_letter_2 and
          not (x_letter_1 in out_formula and x_letter_2 in out_formula))


einsum_repeated_one_hot = Rule((Einsum, Str('formula'), Segment('args1'),
                                (OneHot, Val('x'), Scalar('depth')),
                                Segment('args2'),
                                (OneHot, Val('x'), Scalar('depth')),
                                Segment('args3')),
                               _einsum_repeated_one_hot,
                               (_einsum_repeated_one_hot_pred,))


def _reconstitute_einsum_formula(input_formulas, output_formula):
  return '{}->{}'.format(','.join(input_formulas), output_formula)


## Miscellaneous expansions


def _log_einsum_expand(formula, args):
  assert _check_log_einsum(formula)
  result = np.log(args[0])
  for arg in args[1:]:
    result += np.log(arg)
  return result


def _check_log_einsum(formula):
  input_formulas, output_formula = split_einsum_formula(formula)
  unique_input_indexes = set(list(''.join(input_formulas)))
  return unique_input_indexes == set(list(output_formula))


replace_log_einsum = Rule((Log, (Einsum, Str('formula', _check_log_einsum),
                                 Segment('args'))),
                          _log_einsum_expand)


## replacing autograd internal ops


replace_vspace_add = Rule((VSpaceAdd, Any('vs'), Val('x_prev'), Val('x_new')),
                          lambda vs, x_prev, x_new: x_prev + x_new)


## Miscellaneous simplifications


def constant_folding_einsum(formula, *args):
  in_formulas, out_formula = split_einsum_formula(formula)
  const_indices = []
  node_indices = []
  const_letters = set()
  node_letters = set()
  for i, (in_formula, arg) in enumerate(zip(in_formulas, args)):
    if is_constant(arg):
      const_indices.append(i)
      const_letters.update(in_formula)
    else:
      node_indices.append(i)
      node_letters.update(in_formula)
  const_args = []
  const_in_formulas = []
  indices_to_remove = []
  for i in const_indices:
    if not node_letters.intersection(in_formulas[i]):
      const_args.append(args[i])
      const_in_formulas.append(in_formulas[i])
      indices_to_remove.append(i)
    elif node_letters.issuperset(in_formulas[i]) and np.all(args[i] == 1):
      indices_to_remove.append(i)
  if not indices_to_remove:
    return np.einsum(formula, *args)

  folded_constant = 1
  if const_args:
    const_letters = frozenset(''.join(const_in_formulas))
    const_out_formula = ''.join([i for i in out_formula if i in const_letters])
    folded_constant = np.einsum('{}->{}'.format(','.join(const_in_formulas),
                                                const_out_formula), *const_args)
    if len(indices_to_remove) == len(in_formulas):
      return folded_constant

  retained_in_formulas = ','.join([in_formulas[i]
                                   for i in range(len(in_formulas))
                                   if i not in indices_to_remove])
  retained_args = [arg for i, arg in enumerate(args)
                   if i not in indices_to_remove]
  if np.isscalar(folded_constant) and folded_constant == 0:
    return 0.
  elif np.isscalar(folded_constant) and folded_constant == 1:
    return np.einsum('{}->{}'.format(retained_in_formulas, out_formula),
                     *retained_args)
  else:
    return np.einsum('{},{}->{}'.format(const_out_formula,
                                        retained_in_formulas, out_formula),
                     *([folded_constant] + retained_args))


# TODO(mhoffman): This isn't 100% kosher for negative inputs.
#                 e.g., (-1 ** 2) ** 1.5 == 1, -1 ** 3 == -1.
fold_power = Rule(
    (Power, (Power, Val('base'), Scalar('power1')), Scalar('power2')),
    lambda base, power1, power2: maybe_power(base, power1 * power2))


### rewriter functions


def make_rewriter(rule):
  """Given a rewrite Rule, produces an attempt_rewrite function."""
  pattern, rewriter, preds = rule
  match = matchers.matcher(pattern)
  def attempt_rewrite(node):
    """Given a node, attempt to pattern-match it and apply an in-place rewrite.

    Args:
      node: an ExprNode against which to match the Rule's pattern and, given a
        match, apply an in-place rewrite.

    Returns:
      If the rewrite could not be applied, returns a falsey value. If the
      rewrite was successful, return the node (which gets in-place modified).

    Side-effects:
      If a rewrite was successful then the returned node is modified in-place,
      and in particular its parents are changed.
    """
    bindings = match(node)
    if bindings is not False:
      rewriter_env = dict(node.kwargs, **bindings)
      if all(pred(**rewriter_env) for pred in preds):
        new_expr = run_rewriter(rewriter, rewriter_env)
        tracers.replace_node_with_expr(node, new_expr)  # modifies node in-place
        return node
    return False
  return attempt_rewrite


def run_rewriter(rewriter, symbolic_env):
  """Runs rewriter on a symbolic environment and returns resulting expression.

  Args:
    rewriter: a rewriter function to be traced into a new expression.
    symbolic_env: a dict of bindings that contains the rewriters' arguments as
      keys and can have literals or ExprNodes as values.

  Returns:
    A new expression built on top of the ExprNodes in env.
  """
  # include default argument values in the environment
  sig = funcsigs.signature(rewriter)
  defaults = {name: param.default for name, param in sig.parameters.items()
              if param.default is not param.empty}
  symbolic_env = dict(defaults, **symbolic_env)

  # trace the rewriter function on dummy values to produce a new subexpression
  args = [symbolic_env[name] for name in sig.parameters.keys()]
  flat_args, unflatten = _flatten(args)
  symbolic_args = ((i, arg) for i, arg in enumerate(flat_args)
                   if isinstance(arg, tracers.ExprNode))
  argnums, argnodes = zip(*symbolic_args)

  def _rewriter(*node_vals):
    return rewriter(*unflatten(subvals(flat_args, zip(argnums, node_vals))))

  node_vals = [tracers.make_dummy(argnode) for argnode in argnodes]
  subexpr = tracers.make_expr(_rewriter, *node_vals)

  # return the new subexpression evaluated in the symbolic environment
  return tracers.inline_expr(subexpr, dict(zip(subexpr.free_vars, argnodes)))


def _flatten(obj):
  """Flatten a potentially-nested list/tuple data structure into a flat list."""
  if not isinstance(obj, (list, tuple)):
    return [obj], lambda lst: lst[0]

  constructor = type(obj)
  if not obj: return [], lambda lst: constructor()

  sublists, unflattens = zip(*map(_flatten, obj))
  lengths = list(map(len, sublists))
  starts = np.subtract(np.cumsum(lengths), lengths)
  flat_list = [elt for sublist in sublists for elt in sublist]
  def unflatten(lst):
    sublists = (lst[start:start+l] for start, l in zip(starts, lengths))
    return constructor(unflatten(sublist)
                        for sublist, unflatten in zip(sublists, unflattens))
  return flat_list, unflatten
