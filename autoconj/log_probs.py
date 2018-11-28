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
"""Log probability functions.

It complements the collection of log-normalizers in `exponential_families.py`.

The API follows scipy.stats. Each log probability function's name is the name
of its associated distribution class followed by "_log_prob". All functions
have the same arguments (and order of arguments) as their associated SciPy
`logpdf` and `logpmf`.

Unlike SciPy log probs, these functions return a scalar value, that is, they
sum across all dimensions. We make this simplification for easier
canonicalization as we don't need to deal with a downstream reduce sum.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import autograd.numpy as np

from autograd.scipy import special
from .tracers import one_hot


def categorical_gen_log_prob(x, p):
  """Log-probability of `categorical_gen` in PPLHam."""
  x_one_hot = one_hot(x, len(p))
  log_prob = np.sum(x_one_hot * np.log(p))
  return log_prob


def dirichlet_gen_log_prob(x, alpha):
  """Log-probability of `dirichlet_gen` in scipy.stats."""
  log_prob = np.sum((alpha - 1) * np.log(x))
  log_prob -= np.sum(special.gammaln(alpha))
  log_prob += np.sum(special.gammaln(np.sum(alpha, -1)))
  return log_prob


def multinomial_gen_log_prob(x, n, p):
  """Log-probability of `multinomial_gen` in scipy.stats."""
  if n != 1:
    raise NotImplementedError()
  log_prob = np.sum(x * np.log(p))
  return log_prob
  # TODO(trandustin): need rewrite rules to handle n > 1
  # log_prob = np.sum(x * np.log(p))
  # log_prob -= np.sum(special.gammaln(x + 1))
  # log_prob += np.sum(special.gammaln(n + 1))
  # return log_prob


def norm_gen_log_prob(x, loc, scale):
  """Log-probability of `norm_gen` in scipy.stats."""
  get_dim = lambda x: np.prod(x.shape) if hasattr(x, "shape") else 1
  precision = 1.0 / scale ** 2
  errors = x - loc
  log_prob = -0.5 * get_dim(errors) * np.log(2.0 * np.pi)
  log_prob += 0.5 * get_dim(errors) * np.sum(np.log(precision))
  log_prob += -0.5 * np.sum(precision * errors * errors)
  return log_prob


# TODO(trandustin): Change signature to follow `scipy.stats.gamma`'s
# (`x, a, loc=0, scale=1`). Using `make_log_joint_fn` with `ph.gamma` will fail
# unless specifying `gamma.rvs(a, b)`. This lets b implicitly act as the rate
# parameter as in here, even though the `gamma.rvs` call will incorrectly set
# `b` as `loc`.
def gamma_gen_log_prob(x, a, b):
  """Log-probability of `gamma_gen` in scipy.stats (via shape/rate)."""
  return (a - 1) * np.log(x) - b * x + a * np.log(b) - special.gammaln(a)
