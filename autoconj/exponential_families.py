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

from collections import namedtuple
from autograd import numpy as np
from autograd.scipy import misc
from autograd.scipy import special
from scipy import stats

from . import graph_util
from . import matchers
from . import pgm
from .patterns import (
    Subtract, Add, Dot, Multiply, Divide, TrueDivide, Node, Val, Einsum, Str,
    Choice, Segment, Log, Log1P, Sum, Tuple, VSpaceAdd, Any, Power, Scalar,
    OneHot, Transpose, EnvLookup, Getitem, Negative, Star)
from . import pplham as ph
from .tracers import logdet
from .util import SupportTypes


T = lambda X: np.swapaxes(X, -1, -2)
sym = lambda X: 0.5 * (X + T(X))

exp_family_stats = []
distbn_defns = []

StatsMatcher = namedtuple('StatsMatcher', ['matcher', 'preds',
                                           'update_suffstat'])
DistributionDefinition = namedtuple('DistributionDefinition',
                                    ['matchers', 'support', 'check',
                                     'suffstat_cls', 'make_log_normalizer',
                                     'distribution'])

def make_matcher(pattern, preds, update_suffstat):
  return StatsMatcher(matchers.matcher(pattern), preds, update_suffstat)


### Logic for matching sufficient statistic nodes to a distribution
def find_distributions(all_stats_nodes, supports):
  nodenames_lognormalizers_distbns = []
  for stats_nodes, support in zip(all_stats_nodes, supports):
    nodenames_lognormalizers_distbns.append(find_distribution(
        stats_nodes, support))
  return zip(*nodenames_lognormalizers_distbns)


def find_distribution(stats_nodes, support):
  for distbn_defn in distbn_defns:
    if distbn_defn.support != support:
      continue
    suffstat_nodes = match_nodes_with_distbn(stats_nodes, distbn_defn)
    if suffstat_nodes:
      return (suffstat_nodes,
              distbn_defn.make_log_normalizer(suffstat_nodes),
              distbn_defn.distribution)
  suffstat_nodes = NotFoundSuffStat(params=tuple(stats_nodes))
  return suffstat_nodes, not_found_normalizer, not_found_distbn


# TODO(matthewjmackay): change so suffstat fields are initialized to None
def init_suffstat(suffstat_cls):
  return suffstat_cls(**{name:{} for name in suffstat_cls._fields})


def match_nodes_with_distbn(stats_nodes, distbn_defn):
  suffstat = init_suffstat(distbn_defn.suffstat_cls)

  def match_node(node, stats_matchers):
    for stats_matcher in stats_matchers:
      bindings = stats_matcher.matcher(node)
      if bindings and all(pred(**bindings) for pred in stats_matcher.preds):
        return stats_matcher.update_suffstat(suffstat, bindings, node)
    return False

  for node in stats_nodes:
    suffstat = match_node(node, distbn_defn.matchers)
    if not suffstat:
      return False

  # Check if all the required statistics are present and return.
  if distbn_defn.check(suffstat):
    return suffstat
  else:
    return False

### Not found distribution
NotFoundSuffStat = namedtuple('NotFoundSuffStat', ['params'])
def asserts_false(*args, **kwargs):
  assert False
not_found_distbn = lambda *args, **kwargs: asserts_false
not_found_normalizer = asserts_false

### Bernoulli distribution
BernoulliSuffStat = namedtuple('BernoulliSuffStat', ['x'])
x_matcher = make_matcher(
    pattern=EnvLookup('x'), preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x': node})))
bernoulli_matchers = frozenset([x_matcher])
bernoulli_check = lambda suffstat: not isinstance(suffstat.x, dict)

bernoulli_log_normalizer = (lambda natparam:
                            np.sum(np.log1p(np.exp(natparam.x))))
bernoulli_distbn = (lambda natparam:
                    stats.bernoulli(special.expit(natparam.x)))

bernoulli_defn = DistributionDefinition(
    matchers=bernoulli_matchers, support=SupportTypes.BINARY,
    check=bernoulli_check, suffstat_cls=BernoulliSuffStat,
    make_log_normalizer=lambda *args: bernoulli_log_normalizer,
    distribution=bernoulli_distbn)

exp_family_stats.append(BernoulliSuffStat)
distbn_defns.append(bernoulli_defn)

### Gamma distribution
GammaSuffStat = namedtuple('GammaSuffStat',
                           ['log_x', 'x'])
log_x_matcher = make_matcher(
    pattern=(Log, EnvLookup('x')), preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'log_x': node})))
x_matcher = make_matcher(
    pattern=EnvLookup('x'), preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x': node})))
gamma_matchers = frozenset([log_x_matcher, x_matcher])
gamma_check = lambda suffstat: (not isinstance(suffstat.x, dict)
                                and not isinstance(suffstat.log_x, dict))


def gamma_log_normalizer(natparam):
  alpha = natparam.log_x + 1.
  beta = -natparam.x
  return np.sum(-alpha * np.log(beta) + special.gammaln(alpha))


def gamma_distbn(natparam):
  args = [natparam.log_x + 1., 0., -1. / natparam.x]
  return stats.gamma(*args)

gamma_defn = DistributionDefinition(
    matchers=gamma_matchers, support=SupportTypes.NONNEGATIVE,
    check=gamma_check, suffstat_cls=GammaSuffStat,
    make_log_normalizer=lambda *args: gamma_log_normalizer,
    distribution=gamma_distbn)

exp_family_stats.append(GammaSuffStat)
distbn_defns.append(gamma_defn)

### Dirichlet distribution
DirichletSuffStat = namedtuple('DirichletSuffStat',
                               ['log_x'])
log_x_matcher = make_matcher(
    pattern=(Log, EnvLookup('x')), preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'log_x': node})))
dirichlet_matchers = frozenset([log_x_matcher])
dirichlet_check = lambda suffstat: not isinstance(suffstat.log_x, dict)


def dirichlet_log_normalizer(natparam):
  alpha = natparam.log_x + 1.
  alpha_sum = np.sum(alpha, -1)
  return np.sum(special.gammaln(alpha)) - np.sum(special.gammaln(alpha_sum))


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


def dirichlet_distbn(natparam):
  return batch_dirichlet(natparam.log_x + 1.)


dirichlet_defn = DistributionDefinition(
    matchers=dirichlet_matchers, support=SupportTypes.SIMPLEX,
    check=dirichlet_check, suffstat_cls=DirichletSuffStat,
    make_log_normalizer=lambda *args: dirichlet_log_normalizer,
    distribution=dirichlet_distbn)

exp_family_stats.append(DirichletSuffStat)
distbn_defns.append(dirichlet_defn)

### Beta distribution
BetaSuffStat = namedtuple('BetaSuffStat',
                          ['log_x', 'log_one_minus_x'])
log_x_matcher = make_matcher(
    pattern=(Log, EnvLookup('x')), preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'log_x': node})))
log_one_minus_x_matcher = make_matcher(
    pattern=(Log1P, (Negative, EnvLookup('x'))), preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'log_one_minus_x': node})))
beta_matchers = frozenset([log_x_matcher, log_one_minus_x_matcher])
beta_check = lambda suffstat: (not isinstance(suffstat.log_x, dict) and
                              not isinstance(suffstat.log_one_minus_x, dict))


def beta_log_normalizer(natparam):
  alpha = natparam.log_x + 1.
  beta = natparam.log_one_minus_x + 1.
  return np.sum(special.gammaln(alpha) + special.gammaln(beta) -
                special.gammaln(alpha+beta))


def beta_distbn(natparam):
  return stats.beta(natparam.log_x+1., natparam.log_one_minus_x+1.)

beta_defn = DistributionDefinition(
    matchers=beta_matchers, support=SupportTypes.UNIT_INTERVAL,
    check=beta_check, suffstat_cls=BetaSuffStat,
    make_log_normalizer=lambda *args: beta_log_normalizer,
    distribution=beta_distbn)

exp_family_stats.append(BetaSuffStat)
distbn_defns.append(beta_defn)

### Categorical distribution
CategoricalSuffStat = namedtuple('CategoricalSuffStat',
                                 ['onehot_x'])
onehot_x_matcher = make_matcher(
    pattern=(OneHot, EnvLookup('x'), Val),
    preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'onehot_x': node})))
categorical_matchers = frozenset([onehot_x_matcher])
categorical_check = lambda suffstat: not isinstance(suffstat.onehot_x, dict)


def categorical_log_normalizer(natparam):
  return np.sum(misc.logsumexp(natparam.onehot_x, -1))


def _softmax(x):
  safe_x = x - x.max(-1, keepdims=True)
  p = np.exp(safe_x)
  return p / p.sum(-1, keepdims=True)


def categorical_distbn(natparam):
  return ph.categorical(_softmax(natparam.onehot_x))

categorical_defn = DistributionDefinition(
    matchers=categorical_matchers, support=SupportTypes.INTEGER,
    check=categorical_check, suffstat_cls=CategoricalSuffStat,
    make_log_normalizer=lambda *args: categorical_log_normalizer,
    distribution=categorical_distbn)

exp_family_stats.append(CategoricalSuffStat)
distbn_defns.append(categorical_defn)

### Multinoulli distribution
MultinoulliSuffStat = namedtuple('MultinoulliSuffStat', ['x'])
x_matcher = make_matcher(
    pattern=EnvLookup('x'),
    preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x': node})))
multinoulli_matchers = frozenset([x_matcher])
multinoulli_check = lambda suffstat: not isinstance(suffstat.x, dict)


def multinoulli_log_normalizer(natparam):
  return np.sum(misc.logsumexp(natparam.x, -1))


def multinoulli_distbn(natparam):
  return stats.multinomial(n=1, p=_softmax(natparam.x))

multinoulli_defn = DistributionDefinition(
    matchers=multinoulli_matchers, support=SupportTypes.ONE_HOT,
    check=multinoulli_check, suffstat_cls=MultinoulliSuffStat,
    make_log_normalizer=lambda *args: multinoulli_log_normalizer,
    distribution=multinoulli_distbn)

exp_family_stats.append(MultinoulliSuffStat)
distbn_defns.append(multinoulli_defn)

### Multivariate normal with dense precision matrix
MultivariateNormalSuffStat = namedtuple('MultivariateNormalSuffStat',
                                        ['x_xtr', 'x', 'x_squared'])


def diagonal_einsum(**kwargs):
  return kwargs['formula'] == '...,...->...'


def quadratic_einsum(**kwargs):
  return kwargs['formula'] == '...a,...b->...ab'

x_xtr_matcher = make_matcher(
    pattern=(Einsum, Str('formula'), EnvLookup('x'), EnvLookup('x')),
    preds=(quadratic_einsum,),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x_xtr': node})))
x_matcher = make_matcher(
    pattern=EnvLookup('x'),
    preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x': node})))
x_squared_matcher = make_matcher(
    pattern=(Einsum, Str('formula'), EnvLookup('x'), EnvLookup('x')),
    preds=(diagonal_einsum,),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x_squared': node})))
multivariate_normal_check = (lambda suffstat:
                             not isinstance(suffstat.x_xtr, dict))
multivariate_normal_matchers = frozenset([x_xtr_matcher, x_matcher,
                                          x_squared_matcher])


def _add_diag(tau, J):
  return J + np.einsum('...i,j,ij->...ij', tau, np.ones(tau.shape[-1]),
                       np.eye(tau.shape[-1]))


def multivariate_normal_log_normalizer(natparam):
  if isinstance(natparam.x_squared, dict):
    tau = np.zeros(natparam.x_xtr.shape[-1])
  else:
    tau = natparam.x_squared
  J = _add_diag(tau, natparam.x_xtr)
  precision = -2 * J
  log_det_term = -0.5 * logdet(sym(precision)).sum()
  pi_term = 0.5 * J.shape[-1] * np.log(2. * np.pi)
  if not isinstance(natparam.x, dict):
    quadratic_term = np.einsum(',...ij,...i,...j->...', 0.5,
                               sym(np.linalg.inv(sym(precision))),
                               natparam.x, natparam.x).sum()
  else:
    quadratic_term = 0
  return quadratic_term + log_det_term + pi_term


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


def multivariate_normal_from_natural_parameters(J, h):
  covariance = sym(np.linalg.inv(-2 * sym(J)))
  mean = np.einsum('...ij,...j->...i', covariance, h)
  return mean, covariance


def multivariate_normal_distbn(natparam):
  if isinstance(natparam.x_squared, dict):
    tau = np.zeros(natparam.x_xtr.shape[-1])
  else:
    tau = natparam.x_squared
  J = _add_diag(tau, natparam.x_xtr)
  if isinstance(natparam.x, dict):
    h = np.zeros(natparam.x_xtr.shape[-1])
  else:
    h = natparam.x
  return BatchMultivariateNormal(
      *multivariate_normal_from_natural_parameters(J, h))

multivariate_normal_defn = DistributionDefinition(
    matchers=multivariate_normal_matchers, support=SupportTypes.REAL,
    check=multivariate_normal_check,
    suffstat_cls=MultivariateNormalSuffStat,
    make_log_normalizer=lambda *args: multivariate_normal_log_normalizer,
    distribution=multivariate_normal_distbn)

exp_family_stats.append(MultivariateNormalSuffStat)
distbn_defns.append(multivariate_normal_defn)

### Diagonal-covariance normal
DiagonalNormalSuffStat = namedtuple('DiagonalNormalSuffStat',
                                    ['x_squared', 'x'])
x_squared_matcher = make_matcher(
    pattern=(Einsum, Str('formula'), EnvLookup('x'), EnvLookup('x')),
    preds=(diagonal_einsum,),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x_squared': node})))
x_matcher = make_matcher(
    pattern=EnvLookup('x'),
    preds=(),
    update_suffstat=(lambda suffstat, bindings, node:
                     suffstat._replace(**{'x': node})))
diagonal_normal_matchers = frozenset([x_squared_matcher, x_matcher])
diagonal_normal_check = lambda suffstat: not isinstance(suffstat.x_squared, dict)


def diagonal_normal_log_normalizer(natparam):
  if isinstance(natparam.x, dict):
    h = np.zeros_like(natparam.x_squared)
  else:
    h = natparam.x
  tau = -2 * natparam.x_squared
  mu = h / tau
  return np.sum(-0.5*np.log(tau) + 0.5*tau*mu*mu + 0.5*np.log(2.*np.pi))


def diagonal_normal_from_natural_parameters(half_minus_tau, h):
  tau = -2 * half_minus_tau
  return h / tau, 1. / np.sqrt(tau)


def diagonal_normal_distbn(natparam):
  if isinstance(natparam.x, dict):
    h = np.zeros_like(natparam.x_squared)
  else:
    h = natparam.x
  return stats.norm(*diagonal_normal_from_natural_parameters(
      natparam.x_squared, h))

diagonal_normal_defn = DistributionDefinition(
    matchers=diagonal_normal_matchers, support=SupportTypes.REAL,
    check=diagonal_normal_check, suffstat_cls=DiagonalNormalSuffStat,
    make_log_normalizer=lambda *args: diagonal_normal_log_normalizer,
    distribution=diagonal_normal_distbn)

exp_family_stats.append(DiagonalNormalSuffStat)
distbn_defns.append(diagonal_normal_defn)

### Structured normal distribution
StructuredNormalSuffStat = namedtuple('StructuredNormalSuffStat',
                                      ['xi_xjtrs', 'xi_times_xjs',
                                       'xi_xitrs', 'xi_squareds',
                                       'xis'])
def different_indices(formula, x, idx):
  return len(idx) == len(set(idx))
def single_index(formula, x, idx):
  return len(set(idx)) == 1
def two_indices(formula, x, idx):
  return len(idx) == 2


def make_joint_updater(name):
  def joint_updater(suffstat, bindings, node):
    factor_dict = getattr(suffstat, name)
    factor_dict[bindings['idx']] = node
    return suffstat
  return joint_updater


def make_single_updater(name):
  def single_updater(suffstat, bindings, node):
    factor_dict = getattr(suffstat, name)
    idx = bindings['idx']
    if isinstance(idx, tuple):
      idx = idx[0]
    factor_dict[(idx,)] = node
    return suffstat
  return single_updater


xi_xjtr_matcher = make_matcher(
    pattern=(Einsum, Str('formula'),
             Star((Getitem, EnvLookup('x'), Val('idx')), accumulate=['idx'])),
    preds=(different_indices, quadratic_einsum, two_indices),
    update_suffstat=make_joint_updater('xi_xjtrs'))
xi_times_xj_matcher = make_matcher(
    pattern=(Einsum, Str('formula'),
             Star((Getitem, EnvLookup('x'), Val('idx')), accumulate=['idx'])),
    preds=(different_indices, diagonal_einsum, two_indices),
    update_suffstat=make_joint_updater('xi_times_xjs'))
xi_xitr_matcher = make_matcher(
    pattern=(Einsum, Str('formula'),
             Star((Getitem, EnvLookup('x'), Val('idx')), accumulate=['idx'])),
    preds=(quadratic_einsum, single_index),
    update_suffstat=make_single_updater('xi_xitrs'))
xi_squared_matcher = make_matcher(
    pattern=(Einsum, Str('formula'),
             Star((Getitem, EnvLookup('x'), Val('idx')), accumulate=['idx'])),
    preds=(diagonal_einsum, single_index),
    update_suffstat=make_single_updater('xi_squareds'))
xi_matcher = make_matcher(
    pattern=(Getitem, EnvLookup('x'), Val('idx')),
    preds=(),
    update_suffstat=make_single_updater('xis'))
struct_normal_matchers = frozenset([xi_xjtr_matcher, xi_times_xj_matcher,
                                    xi_xitr_matcher, xi_squared_matcher,
                                    xi_matcher])


def struct_normal_check(suffstat):
  factors = (suffstat.xi_xjtrs.keys() + suffstat.xi_times_xjs.keys() +
      suffstat.xi_xitrs.keys() + suffstat.xi_squareds.keys() +
      suffstat.xis.keys())
  nodes = {node for factor in factors for node in factor}

  for node in nodes:
    single_factor = (node,)
    if (single_factor not in suffstat.xi_xitrs and
        single_factor not in suffstat.xi_squareds):
      return False
  return True


def make_struct_normal_log_normalizer(suffstat):
  factors = {frozenset(factor) for factor in suffstat.xi_xjtrs.keys() +
             suffstat.xi_times_xjs.keys() + suffstat.xi_xitrs.keys() +
             suffstat.xi_squareds.keys() + suffstat.xis.keys()}
  factor_graph = graph_util.make_factor_graph(factors)
  tree_order = graph_util.find_tree(factor_graph)
  if tree_order:
    elim_order = [node for node in tree_order if isinstance(node, int)]
    return pgm.make_tree_normal_log_normalizer(elim_order)
  return not_found_normalizer

struct_normal_distbn = not_found_distbn

struct_normal_defn = DistributionDefinition(
    matchers=struct_normal_matchers, support=SupportTypes.REAL,
    check=struct_normal_check, suffstat_cls=StructuredNormalSuffStat,
    make_log_normalizer=make_struct_normal_log_normalizer,
    distribution=struct_normal_distbn)

exp_family_stats.append(StructuredNormalSuffStat)
distbn_defns.append(struct_normal_defn)

### Structured categorical distribution
StructuredCategoricalSuffStat = namedtuple('StructuredCategoricalSuffStat',
                                           ['joint_onehot_xis',
                                            'single_onehot_xis'])


# TODO(matthewjmackay): we probably need a predicate on the einsum formula here
joint_onehot_xis_matcher = make_matcher(
    pattern=(Einsum, Str('formula'),
             Star((OneHot, (Getitem, EnvLookup('x'), Val('idx')), Val),
                  accumulate=['idx'])),
    preds=(different_indices,),
    update_suffstat=make_joint_updater('joint_onehot_xis'))
single_onehot_xi_matcher = make_matcher(
    pattern=(OneHot, (Getitem, EnvLookup('x'), Val('idx')), Val),
    preds=(),
    update_suffstat=make_single_updater('single_onehot_xis'))
struct_categorical_matchers = frozenset([joint_onehot_xis_matcher,
                                         single_onehot_xi_matcher])


def struct_categorical_check(suffstat):
  return True  # TODO(matthewjmackay): think about whether this is correct


def make_struct_categorical_log_normalizer(suffstat):
  factors = suffstat.joint_onehot_xis.keys() + suffstat.single_onehot_xis.keys()
  factor_graph = graph_util.make_factor_graph(factors)
  tree_order = graph_util.find_tree(factor_graph)
  if tree_order:
    elim_order = [node for node in tree_order if not isinstance(node, tuple)]
    return pgm.make_tree_categorical_log_normalizer(elim_order)
  return not_found_normalizer

struct_categorical_distbn = not_found_distbn

struct_categorical_defn = DistributionDefinition(
    matchers=struct_categorical_matchers,
    support=SupportTypes.INTEGER, check=struct_categorical_check,
    suffstat_cls=StructuredCategoricalSuffStat,
    make_log_normalizer=make_struct_categorical_log_normalizer,
    distribution=struct_categorical_distbn)

exp_family_stats.append(StructuredCategoricalSuffStat)
distbn_defns.append(struct_categorical_defn)

if __name__ == '__main__':
  app.run(main)
