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
"""Pattern matcher for computation graphs.

See //learning/brain/contrib/kfac/.../tensormatch/graph_matcher.py for more.

The grammar for the pattern language implemented in this file is:

  pattern ::= element | choice | list | internal_node | negated_pattern
  patterns ::= pattern, patterns | ()

  element ::= ('?', name, restrictions) | ('?', name, restrictions, binding)
  name ::= PYTHON_STRING
  restrictions ::= PYTHON_FUNCTION, restrictions | ()
  binding ::= PYTHON_FUNCTION

  choice ::= ('?:choice', patterns)

  list ::= ('List', list_elements)
  list_elements ::= list_element, list_elements | ()
  list_element ::= pattern | star
  star ::= ('??', name, pattern)

  internal_node ::= (pattern, input_constraints)
  input_constraints ::= list_elements

  negated_pattern ::= ('?:not', pattern)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import itertools
import operator

## graph interface (otherwise the logic is generic to any graph)

parents = operator.attrgetter('args')

## utilities

def identity(x): return x

def _any(itr):
  for val in itr:
    if val: return val
  return False

def _all(itr):
  any_iterations = False
  for val in itr:
    any_iterations = True
    if not val: return val
  return val if any_iterations else True

def is_seq(x): return isinstance(x, (tuple, list))
def is_empty_seq(x): return is_seq(x) and not bool(x)
def is_pair(x): return is_seq(x) and len(x) > 0
def is_thunk(x):
  if callable(x):
    spec = inspect.getargspec(x)
    num_free_args = len(set(spec.args)) - len(set(spec.defaults or {}))
    return num_free_args == 0
  return False

Literal = collections.namedtuple('Literal', ['val'])
Literal.__hash__ = lambda self: id(self.val)

def _singleton(elt):
  try:
    return {elt}
  except TypeError:
    return {Literal(elt)}

def _set(*elts): return reduce(operator.or_, map(_singleton, elts))

## define the syntax of the pattern language

is_pat = is_pair

def is_element_pattern(pat): return is_pair(pat) and pat[0] == '?'
def element_name(pat): return pat[1]
def element_restrictions(pat): return pat[2]
def element_binding(pat): return pat[3] if pat[3:] else identity

def is_choice_pattern(pat): return is_pair(pat) and pat[0] == '?:choice'
def choice_alternatives(pat): return pat[1:]

def is_list_pattern(pat): return is_pair(pat) and pat[0] == 'List'
def list_elements(pat): return pat[1:]

# star matchers are a special form that only occurrs within list patterns,
# and their matching is handled inside match_list
def is_star_matcher(matcher):
  return is_pair(matcher) and matcher[0] == '??' and len(matcher) == 4

def is_not_pattern(pat): return is_pair(pat) and pat[0] == '?:not'
def negated_pattern(pat): return pat[1]

def is_negated_pattern(pat): return pat[1]

def is_noconsume_pattern(pat): return is_pat(pat) and pat[0] == '?:noconsume'

def is_internal_node_pattern(pat): return is_pair(pat) and is_pair(pat[0])

## constructors for pattern-matching combinators

def match_eqv(pattern):
  def eqv_match(data, bindings, consumed, succeed):
    return data == pattern and succeed(bindings, consumed | _singleton(data))
  return eqv_match

def match_noconsume(data, bindings, consumed, succeed):  # pylint: disable=unused-argument
  return succeed(bindings, consumed)

def match_element(name, restrictions, binding):
  def element_match(data, bindings, consumed, succeed):
    consumed |= _singleton(data)
    if _all(restriction(data) for restriction in restrictions):
      if not name:
        return succeed(bindings, consumed)
      elif name in bindings:
        return bindings[name] == binding(data) and succeed(bindings, consumed)
      return succeed(dict(bindings, **{name: binding(data)}), consumed)
    return False
  return element_match

def match_choice(*match_combinators):
  def choice_match(data, bindings, consumed, succeed):
    return _any(matcher(data, bindings, consumed, succeed)
                for matcher in match_combinators)
  return choice_match

def match_list(*match_combinators):
  def list_match(data, bindings, consumed, succeed):
    return _list_match(data, match_combinators, bindings, consumed, succeed)

  def _list_match(data, matchers, bindings, consumed, succeed, carried=()):
    def match_first_then_rest(combinator, datum):
      return combinator(datum, bindings, consumed, match_subsequent_elements)

    def match_subsequent_elements(bindings, consumed):
      return _list_match(data[1:], matchers[1:], bindings, consumed, succeed)

    def try_star(star_matcher):
      _, var_name, submatcher, accumulate = star_matcher
      bind = lambda val: dict(bindings, **({var_name: val} if var_name else {}))

      # if the name is already bound, check that we have a segment here that's
      # consistent with it
      if var_name in bindings:
        n = len(bindings[var_name])
        return (tuple(bindings[var_name]) == tuple(data[:n])
                and _list_match(data[n:], matchers[1:], bindings, consumed,
                                succeed))

      def accumulate_back(new_bindings, bindings):
        accumulated = {k:bindings.get(k, ()) + (new_bindings[k],) for k in accumulate}
        return dict(new_bindings, **accumulated)

      def alternatives():
        # if the data list is empty and there are no other matchers, we match
        if not data and not matchers:
          yield succeed(bind(data), consumed | _set(data))

        # try matching nothing more, proceed with the rest of the non-empty list
        yield _list_match(data[0:], matchers[1:],
                          bind(carried), consumed | _set(carried), succeed)

        # if the data is not empty, try consuming one element *without* using up
        # this star matcher
        if data:
          subbindings = {k:v for k, v in bindings.iteritems() if k not in accumulate}
          yield submatcher(data[0], subbindings, consumed,
                           lambda new_bindings, consumed: _list_match(
                               data[1:], matchers, accumulate_back(new_bindings, bindings),
                               consumed, succeed, carried = carried + (data[0],)))

      return _any(alternatives())

    if is_empty_seq(matchers) and is_empty_seq(data):
      return succeed(bindings, consumed)
    if is_pair(matchers):
      if is_star_matcher(matchers[0]):
        return try_star(matchers[0])
      else:
        return is_pair(data) and match_first_then_rest(matchers[0], data[0])
    return False
  return list_match

def match_not(match_combinator):
  def not_match(data, bindings, consumed, succeed):
    return (not match_combinator(data, bindings, set(),
                                 lambda bindings, _: True)
            and succeed(bindings, consumed))
  return not_match

def match_internal(*match_combinators):
  expanded_matcher = match_list(*match_combinators)
  def internal_match(data, bindings, consumed, succeed):
    try:
      expanded = tuple(itertools.chain([data], parents(data)))
    except:
      return False
    return expanded_matcher(expanded, bindings, consumed, succeed)
  return internal_match

## parsing the pattern language into compositions of combinators

class PatternEvaluator(object):

  def __init__(self, default_operation=None):
    self.default_operation = default_operation
    self.handlers = []

  def defhandler(self, predicate, handler):
    self.handlers.append((predicate, handler))

  def __call__(self, pat):
    for predicate, handler in self.handlers:
      if predicate(pat):
        return handler(pat)
    if self.default_operation:
      return self.default_operation(pat)
    raise ValueError

make_combinators = PatternEvaluator(match_eqv)
make_combinators.defhandler(
    is_element_pattern,
    lambda pat: match_element(element_name(pat), element_restrictions(pat),
                              element_binding(pat)))
make_combinators.defhandler(
    is_list_pattern,
    lambda pat: match_list(*map(make_combinators, list_patterns(pat))))
make_combinators.defhandler(
    is_star_matcher,
    lambda pat: (pat[0], pat[1], make_combinators(pat[2]), pat[3]))
make_combinators.defhandler(
    is_choice_pattern,
    lambda pat: match_choice(*map(make_combinators, choice_alternatives(pat))))
make_combinators.defhandler(
    is_not_pattern,
    lambda pat: match_not(make_combinators(negated_pattern(pat))))
make_combinators.defhandler(
    is_noconsume_pattern,
    lambda pat: match_noconsume)
make_combinators.defhandler(
    is_internal_node_pattern,
    lambda pat: match_internal(*map(make_combinators, pat)))

## utility function so the patterns require fewer parentheses

def expand_syntax(pat):
  def is_thunk(x):
    if callable(x):
      spec = inspect.getargspec(x)
      num_free_args = len(spec.args) - len(spec.defaults or {})
      return num_free_args == 0
    return False
  while is_thunk(pat):
    pat = pat()
  if isinstance(pat, (tuple, list)):
    return type(pat)(map(expand_syntax, pat))
  return pat

## main matcher interface functions

def matcher(pattern):
  combinators = make_combinators(expand_syntax(pattern))
  def match(node):
    return combinators(node, {}, set(), lambda bindings, _: bindings or True)
  return match

def all_matcher(pattern):
  combinators = make_combinators(expand_syntax(pattern))
  results = []

  def all_matches(node):
    combinators(node, {}, set(),
                lambda bindings, _: results.append(bindings or True))
    return results

  return all_matches

def matcher_with_consumed(pattern):
  combinators = make_combinators(expand_syntax(pattern))
  def match(node):
    return combinators(node, {}, set(),
                       lambda bindings, consumed: (bindings, consumed))
  return match
