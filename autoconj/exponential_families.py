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
"""This file encodes knowledge about exponential families.

Each exponential family (normal, gamma, Dirichlet, etc.) is completely
characterized by:
* Support
* Base Measure (not yet implemented---mostly unimportant)
* Sufficient Statistics

The functions and data structures in this file map from the above
information to:
* Log-normalizer function: Maps natural parameters to a scalar such that
  \int_x \exp(natural_parameters^T sufficient_statistics(x)
              - log_normalizer(natural_parameters)) dx = 1.
* scipy.stats distribution classes.
* Standard parameters for those classes as a function of natural parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from scipy import stats

from autograd import numpy as np
from autograd.scipy import misc
from autograd.scipy import special

from . import pplham as ph
from .tracers import logdet
from .util import SupportTypes


suff_stat_to_dist = defaultdict(lambda: defaultdict(dict))
suff_stat_to_log_normalizer = defaultdict(lambda: defaultdict(dict))


T = lambda X: np.swapaxes(X, -1, -2)
sym = lambda X: 0.5 * (X + T(X))


def canonical_statistic_strs(x):
  x = list(x)
  order = np.argsort(x)
  return tuple([x[i] for i in order]), order


def _add_distribution(support, statistic_strs, parameter_translator,
                      distribution_factory, log_normalizer_fun):
  key, _ = canonical_statistic_strs(statistic_strs)
  suff_stat_to_dist[support][key] = (
      lambda *args: distribution_factory(*parameter_translator(*args)))
  suff_stat_to_log_normalizer[support][key] = log_normalizer_fun


# Functions implementing natural parameter mappers and log-normalizers.
def _multivariate_normal_from_natural_parameters(J, h):
  covariance = sym(np.linalg.inv(-2 * sym(J)))
  mean = np.einsum('...ij,...j->...i', covariance, h)
  return mean, covariance


def _add_diag(tau, J):
  return J + np.einsum('...i,j,ij->...ij', tau, np.ones(tau.shape[-1]),
                       np.eye(tau.shape[-1]))


def _multivariate_normal_from_natural_parameters_plus_diag(tau, J, h):
  return _multivariate_normal_from_natural_parameters(_add_diag(tau, J), h)


def _multivariate_normal_plus_diag_log_normalizer(tau, J, h):
  return _multivariate_normal_log_normalizer(_add_diag(tau, J), h)


def _multivariate_normal_log_normalizer(J, h):
  precision = -2 * J
  log_det_term = -0.5 * logdet(sym(precision)).sum()
  pi_term = 0.5 * J.shape[-1] * np.log(2. * np.pi)
  quadratic_term = np.einsum(',...ij,...i,...j->...', 0.5,
                             sym(np.linalg.inv(sym(precision))), h, h).sum()
  return quadratic_term + log_det_term + pi_term


def _zero_mean_multivariate_normal_from_natural_parameters(J):
  return np.zeros(J.shape[-1]), sym(np.linalg.inv(-2 * sym(J)))


def _zero_mean_multivariate_normal_log_normalizer(J):
  precision = -2 * J
  log_det_term = -0.5 * logdet(sym(precision)).sum()
  pi_term = 0.5 * J.shape[-1] * np.log(2. * np.pi)
  return log_det_term + pi_term


def _zero_mean_diagonal_normal_from_natural_parameters(half_minus_tau):
  return np.zeros_like(half_minus_tau), 1. / np.sqrt(-2 * half_minus_tau)


def _zero_mean_diagonal_normal_log_normalizer(half_minus_tau):
  return 0.5 * (-np.log(-2 * half_minus_tau) + np.log(2. * np.pi)).sum()


def _diagonal_normal_from_natural_parameters(half_minus_tau, h):
  tau = -2 * half_minus_tau
  return h / tau, 1. / np.sqrt(tau)


def _diagonal_normal_log_normalizer(half_minus_tau, h):
  tau = -2 * half_minus_tau
  mu = h / tau
  return np.sum(-0.5 * np.log(tau)
                + 0.5 * tau * mu * mu
                + 0.5 * np.log(2. * np.pi))


def _gamma_from_natural_parameters(a_minus_1, minus_b):
  return a_minus_1 + 1, 0., -1. / minus_b


def _gamma_log_normalizer(a_minus_1, minus_b):
  a = a_minus_1 + 1
  b = -minus_b
  return np.sum(-a * np.log(b) + special.gammaln(a))


def _beta_from_natural_parameters(a_minus_1, b_minus_1):
  return a_minus_1 + 1., b_minus_1 + 1.


def _beta_log_normalizer(a_minus_1, b_minus_1):
  a = a_minus_1 + 1.
  b = b_minus_1 + 1.
  return np.sum(special.gammaln(a) + special.gammaln(b)
                - special.gammaln(a + b))


def _dirichlet_log_normalizer(alpha_minus_1):
  alpha = alpha_minus_1 + 1.
  alpha_sum = np.sum(alpha, -1)
  return np.sum(special.gammaln(alpha)) - np.sum(special.gammaln(alpha_sum))


# A couple of multivariate distributions that need batch support.


def batch_dirichlet(alpha):
  """Batched `np.ndarray` of Dirichlet frozen distributions.

  To get each frozen distribution, index the returned `np.ndarray` followed by
  `item(0)`.
  """
  if alpha.ndim == 1:
    return stats.dirichlet(alpha)
  return np.array(
      [stats.dirichlet(vec) for vec in alpha.reshape([-1, alpha.shape[-1]])]
  ).reshape(alpha.shape[:-1])


class BatchMultivariateNormal(object):
  def __init__(self, mean, cov):
    self.mean = mean
    self.cov = cov
    self._chol = None

  def __getitem__(self, i):
    return BatchMultivariateNormal(self.mean[i], self.cov[i])

  @property
  def chol(self):
    if self._chol is None:
      self._chol = np.linalg.cholesky(self.cov)
    return self._chol

  def rvs(self):
    return self.mean + self.chol.dot(np.random.randn(self.mean.shape[-1]))


# Register all known exponential-family distributions.


# Multivariate normal with dense precision matrix.
_add_distribution(SupportTypes.REAL, ['x', 'einsum(...a,...b->...ab, x, x)'],
                  _multivariate_normal_from_natural_parameters,
                  BatchMultivariateNormal,
                  _multivariate_normal_log_normalizer)
# Multivariate normal with a diagonal component.
_add_distribution(SupportTypes.REAL, ['x', 'einsum(...a,...b->...ab, x, x)',
                                      'einsum(...,...->..., x, x)'],
                  _multivariate_normal_from_natural_parameters_plus_diag,
                  BatchMultivariateNormal,
                  _multivariate_normal_plus_diag_log_normalizer)
# Zero-mean specialization of the multivariate normal.
_add_distribution(SupportTypes.REAL, ['einsum(...a,...b->...ab, x, x)'],
                  _zero_mean_multivariate_normal_from_natural_parameters,
                  BatchMultivariateNormal,
                  _zero_mean_multivariate_normal_log_normalizer)

# TODO(mhoffman): Zero-mean specialization of the multivariate normal
# with a diagonal component.

# Zero-mean specialization of the diagonal-covariance multivariate normal.
_add_distribution(SupportTypes.REAL, ['einsum(...,...->..., x, x)'],
                  _zero_mean_diagonal_normal_from_natural_parameters,
                  stats.norm,
                  _zero_mean_diagonal_normal_log_normalizer)
# Diagonal-covariance specialization of the multivariate normal.
_add_distribution(SupportTypes.REAL, ['einsum(...,...->..., x, x)', 'x'],
                  _diagonal_normal_from_natural_parameters,
                  stats.norm,
                  _diagonal_normal_log_normalizer)
# Gamma.
_add_distribution(SupportTypes.NONNEGATIVE, ['x', 'log(x)'],
                  _gamma_from_natural_parameters,
                  stats.gamma,
                  _gamma_log_normalizer)
# Beta.
# TODO(mhoffman): log1p(negative(1)) yields a divide-by-zero.
# TODO(mhoffman): Write rule to transform log(1 - x) into log1p(negative(x)).
_add_distribution(SupportTypes.UNIT_INTERVAL, ['log(x)', 'log1p(negative(x))'],
                  _beta_from_natural_parameters,
                  stats.beta,
                  _beta_log_normalizer)
# Dirichlet.
_add_distribution(SupportTypes.SIMPLEX, ['log(x)'],
                  lambda alpha_minus_1: (alpha_minus_1 + 1,),
                  batch_dirichlet,
                  _dirichlet_log_normalizer)
# Bernoulli.
# TODO(mhoffman): A more numerically stable softplus would be preferable.
_add_distribution(SupportTypes.BINARY, ['x'],
                  lambda logit_prob: (special.expit(logit_prob),),
                  stats.bernoulli,
                  lambda logit_prob: np.sum(np.log1p(np.exp(logit_prob))))
# Categorical.
def _softmax(x):
  safe_x = x - x.max(-1, keepdims=True)
  p = np.exp(safe_x)
  return p / p.sum(-1, keepdims=True)
_add_distribution(SupportTypes.INTEGER, ['one_hot(x)'],
                  lambda logit_probs: (_softmax(logit_probs),),
                  ph.categorical,
                  lambda logit_probs: np.sum(misc.logsumexp(logit_probs, -1)))
# Multinoulli.
_add_distribution(SupportTypes.ONE_HOT, ['x'],
                  lambda logit_probs: (_softmax(logit_probs),),
                  lambda p: stats.multinomial(n=1, p=p),
                  lambda logit_probs: np.sum(misc.logsumexp(logit_probs, -1)))
