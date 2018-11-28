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
"""Mixture of Gaussians with PPLHam."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import autograd.numpy as np

from autoconj import conjugacy
from autoconj import pplham as ph
from autoconj.tracers import one_hot


def model(sigma_sq, alpha, sigma_sq_mu):
  pi = ph.dirichlet.rvs(alpha)
  mu = ph.norm.rvs(0., np.sqrt(sigma_sq_mu), size=[5, 2])
  z = ph.categorical.rvs(pi, size=200)
  z_one_hot = one_hot(z, len(pi))
  mu_z = np.dot(z_one_hot, mu)
  x = ph.norm.rvs(mu_z, np.sqrt(sigma_sq))
  return x


def remove_arg(argnum, args):
  return args[:argnum] + args[argnum + 1:]


def main(argv):
  del argv

  n_clusters = 5
  n_dimensions = 2
  n_observations = 200

  alpha = 3.3 * np.ones(n_clusters)
  sigma_sq_mu = 1.5 ** 2
  sigma_sq = 0.5 ** 2

  np.random.seed(10001)

  pi = np.random.gamma(alpha)
  pi /= pi.sum()
  mu = np.random.normal(0, np.sqrt(sigma_sq_mu), [n_clusters, n_dimensions])
  z = np.random.choice(np.arange(n_clusters), size=n_observations, p=pi)
  x = np.random.normal(mu[z, :], sigma_sq)

  pi_est = np.ones(n_clusters) / n_clusters
  z_est = np.random.choice(np.arange(n_clusters), size=n_observations, p=pi_est)
  mu_est = np.random.normal(0., 0.01, [n_clusters, n_dimensions])

  all_args = [sigma_sq, alpha, sigma_sq_mu, pi_est, mu_est, z_est, x]

  log_joint = ph.make_log_joint_fn(model)
  pi_posterior = conjugacy.complete_conditional(
      log_joint, 3, conjugacy.SupportTypes.SIMPLEX, *all_args)
  z_posterior = conjugacy.complete_conditional(
      log_joint, 5, conjugacy.SupportTypes.INTEGER, *all_args)
  mu_posterior = conjugacy.complete_conditional(
      log_joint, 4, conjugacy.SupportTypes.REAL, *all_args)

  print('iteration\tlog_joint')
  for iteration in range(100):
    z_est[:] = z_posterior(*remove_arg(5, all_args)).rvs()
    pi_est[:] = pi_posterior(*remove_arg(3, all_args)).rvs()
    mu_est[:] = mu_posterior(*remove_arg(4, all_args)).rvs()
    print('{}\t\t{}'.format(iteration, log_joint(*all_args)))


if __name__ == '__main__':
  app.run(main)
