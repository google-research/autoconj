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
"""PPLHam2, a probabilistic programming language in style of Edward2.

This module provides two members:

1. Lightly wrapped distributions from scipy.stats. They enable tracing over any
   calls to `rvs`;
2. A `make_log_joint_fn` factory function. It takes a PPLHam program and returns
   its log joint probability function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import functools
import inspect
import threading
import numpy as np
from scipy import stats
import six

from . import log_probs as _log_probs


def make_log_joint_fn(model):
  """Takes PPLHam probabilistic program and returns its log joint function.

  Args:
    model: Python callable which executes the generative process of a
      computable probability distribution using PPLham random variables.

  Returns:
    A log-joint probability function. Its inputs are `model`'s original inputs
    and random variables which appear during the program execution. Its output
    is a scalar `np.ndarray`.

  #### Examples

  Below we define Bayesian logistic regression as an PPLHam program, which
  represents the model's generative process. We apply `make_log_joint_fn` in
  order to alternatively represent the model in terms of its joint probability
  function.

  ```python
  import pplham as ph

  def model(X):
    beta = ph.norm.rvs(loc=0., scale=0.1, size=X.shape[1])
    loc = np.einsum('ij,j->i', X, beta)
    y = ph.norm.rvs(loc=loc, scale=1.)
    return y

  log_joint = ph.make_log_joint_fn(model)

  X = np.random.normal(size=[3, 2])
  beta = np.random.normal(size=[2])
  y = np.random.normal(size=[3])
  out = log_joint(X, beta, y)
  ```

  One can use kwargs in `log_joint` if `rvs` are given `name` kwargs.

  ```python
  def model(X):
    beta = ph.norm.rvs(loc=0., scale=0.1, size=X.shape[1], name="beta")
    loc = np.einsum('ij,j->i', X, beta)
    y = ph.norm.rvs(loc=loc, scale=1., name="y")
    return y

  log_joint = ph.make_log_joint_fn(model)
  out = log_joint(X, y=y, beta=beta)
  ```

  #### Notes

  For implementation, we make several requirements:

  1. The `log_probs` module has a supported `log_prob` function for each
     random variable choice.
  2. A random variable's `rvs` method has the same kwargs as scipy.stats'
    `logpmf`/`logpdf` up to `size` and `random_state`.
  3. The event outcome is the first argument of the `log_prob` function in the
     `log_probs` module.
  4. User must use explicit kwargs (no positional arguments) when specifying
     `size` and `random_state` in the `rvs` method.
     TODO(trandustin): Relax this requirement.
  """
  def log_joint_fn(*args, **kwargs):
    """Log-probability of inputs according to a joint probability distribution.

    Args:
      *args: Positional arguments. They are the model's original inputs and can
        alternatively be specified as part of `kwargs`.
      **kwargs: Keyword arguments, where for each key-value pair `k` and `v`,
        `v` is passed as a `value` to the random variable(s) whose keyword
        argument `name` during construction is equal to `k`.

    Returns:
      Scalar `np.ndarray`, which represents the model's log-probability summed
      over all PPLHam random variables and their dimensions.

    Raises:
      TypeError: If a random variable in the model has no specified value in
        `**kwargs`.
    """
    log_probs = []
    args_counter = []

    def interceptor(rv_call, *rv_args, **rv_kwargs):
      """Overrides a random variable's `value` and accumulates its log-prob."""
      if len(args) - len(args_counter) > 0:
        value = args[len(args_counter)]
        args_counter.append(0)
      else:
        # Set value to keyword argument indexed by `name` (an input tensor).
        rv_name = rv_kwargs.get("name")
        if rv_name is None:
          raise KeyError("Random variable call {} has no name in its arguments."
                         .format(rv_call.im_class.__name__))
        value = kwargs.get(rv_name)
        if value is None:
          raise LookupError("Keyword argument specifying value for {} is "
                            "missing.".format(rv_name))
      log_prob_fn = getattr(_log_probs, rv_call.im_class.__name__ + "_log_prob")
      rv_kwargs.pop("size", None)
      rv_kwargs.pop("random_state", None)
      rv_kwargs.pop("name", None)
      log_prob = log_prob_fn(value, *rv_args, **rv_kwargs)
      log_probs.append(log_prob)
      return value

    args, model_args, model_kwargs = _get_function_inputs(
        model, *args, **kwargs)
    with interception(interceptor):
      model(*model_args, **model_kwargs)
    log_prob = sum(log_probs)
    return log_prob
  return log_joint_fn


def _get_function_inputs(f, *args, **kwargs):
  """Filters inputs to be compatible with function `f`'s signature.

  Args:
    f: Function according to whose input signature we filter arguments.
    *args: Keyword arguments to filter according to `f`.
    **kwargs: Keyword arguments to filter according to `f`.

  Returns:
    New original args, args of f, kwargs of f.
  """
  if hasattr(f, "_func"):  # functions returned by tf.make_template
    argspec = inspect.getargspec(f._func)  # pylint: disable=protected-access
  else:
    argspec = inspect.getargspec(f)

  fkwargs = {}
  for k, v in six.iteritems(kwargs):
    if k in argspec.args:
      fkwargs[k] = v
      kwargs.pop(k)
  num_args = len(argspec.args) - len(fkwargs)
  fargs = args[:num_args]
  new_args = args[num_args:]
  return new_args, fargs, fkwargs


class _InterceptorStack(threading.local):
  """A thread-local stack of interceptors."""

  def __init__(self):
    super(_InterceptorStack, self).__init__()
    self.stack = [lambda f, *args, **kwargs: f(*args, **kwargs)]


_interceptor_stack = _InterceptorStack()


@contextmanager
def interception(interceptor):
  """Python context manager for interception.

  Upon entry, an interception context manager pushes an interceptor onto a
  thread-local stack. Upon exiting, it pops the interceptor from the stack.

  Args:
    interceptor: Function which takes a callable `f` and inputs `*args`,
      `**kwargs`.

  Yields:
    None.
  """
  try:
    _interceptor_stack.stack.append(interceptor)
    yield
  finally:
    _interceptor_stack.stack.pop()


def get_interceptor():
  """Returns the top-most (last) interceptor on the thread's stack.

  The bottom-most (first) interceptor in the stack is a function which takes
  `f, *args, **kwargs` as input and returns `f(*args, **kwargs)`. It is the
  default if no `interception` contexts have been entered.
  """
  return _interceptor_stack.stack[-1]


def interceptable(func):
  """Decorator that wraps `func` so that its execution is intercepted.

  The wrapper passes `func` to the interceptor for the current thread.

  Args:
    func: Function to wrap.

  Returns:
    The decorated function.
  """
  @functools.wraps(func)
  def func_wrapped(*args, **kwargs):
    return get_interceptor()(func, *args, **kwargs)
  return func_wrapped


# Automatically generate random variables from scipy.stats. We wrap all
# distributions by registering their `rvs` method as `interceptable`.
#
# A vanilla Edward 2.0-like PPL in SciPy would introduce a RandomVariable
# abstraction: it wraps SciPy frozen distributions and calls `rvs` to associate
# the RandomVariable with a sampled value. SciPy distributions already enable
# parameters as input to `rvs`. Therefore instead of introducing a new
# abstraction, we just wrap `rvs`. This enables the same manipulations.
_globals = globals()
for _name in sorted(dir(stats)):
  _candidate = getattr(stats, _name)
  if isinstance(_candidate, (stats._multivariate.multi_rv_generic,  # pylint: disable=protected-access
                             stats.rv_continuous,
                             stats.rv_discrete,
                             stats.rv_histogram)):
    _candidate.rvs = interceptable(_candidate.rvs)
    _globals[_name] = _candidate
    del _candidate


class categorical_gen(stats._multivariate.multi_rv_generic):  # pylint: disable=invalid-name,protected-access
  """Categorical distribution.

  Implementation follows `scipy.stats.multinomial_gen`. We build this manually
  as scipy.stats does not support a categorical distribution.
  """

  def __init__(self, seed=None):
    super(categorical_gen, self).__init__(seed)

  def __call__(self, p, seed=None):
    return categorical_frozen(p, seed)

  def _process_parameters(self, p):
    p = np.array(p, dtype=np.float64, copy=True)
    p[..., -1] = 1. - p[..., :-1].sum(axis=-1)
    return p

  def rvs(self, p, size=None, random_state=None):
    if size != 1:
      raise NotImplementedError()
    p = self._process_parameters(p)
    random_state = self._get_random_state(random_state)
    scores = (random_state.uniform(size=p.shape[:-1] + (1,)) -
              np.cumsum(p, axis=-1))
    scores[scores < 0] = 0
    return np.argmin(scores, axis=-1)


categorical = categorical_gen()
categorical.rvs = interceptable(categorical.rvs)  # register `rvs` for PPLHam


class categorical_frozen(stats._multivariate.multi_rv_frozen):  # pylint: disable=invalid-name,protected-access

  def __init__(self, p, seed=None):
    self._dist = categorical_gen(seed)
    self.p = self._dist._process_parameters(p)  # pylint: disable=protected-access
    self._dist._process_parameters = lambda p: self.p  # pylint: disable=protected-access

  def rvs(self, size=1, random_state=None):
    return self._dist.rvs(self.p, size, random_state)
