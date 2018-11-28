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
"""Mixture of Gaussians with variational inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import time

from autoconj import log_probs
from autoconj.tracers import one_hot
from autoconj.util import SupportTypes
from autoconj.meanfield import cavi

flags.DEFINE_integer(
    'num_clusters',
    default=10,
    help='Number of clusters.')
flags.DEFINE_integer(
    'num_dimensions',
    default=20,
    help='Number of data dimensions.')
flags.DEFINE_integer(
    'num_observations',
    default=1000,
    help='Number of observations.')
flags.DEFINE_integer(
    'num_iterations',
    default=500,
    help='Number of iterations to run training.')
FLAGS = flags.FLAGS

REAL = SupportTypes.REAL
INTEGER = SupportTypes.INTEGER
SIMPLEX = SupportTypes.SIMPLEX
NONNEGATIVE = SupportTypes.NONNEGATIVE


def make_log_joint(x, alpha, a, b, kappa):
  def log_joint(pi, z, mu, tau):
    log_p_pi = log_probs.dirichlet_gen_log_prob(pi, alpha)
    log_p_mu = log_probs.norm_gen_log_prob(mu, 0., 1. / np.sqrt(kappa * tau))
    log_p_z = log_probs.categorical_gen_log_prob(z, pi)
    log_p_tau = log_probs.gamma_gen_log_prob(tau, a, b)

    z_one_hot = one_hot(z, len(pi))
    mu_z = np.dot(z_one_hot, mu)
    log_p_x = log_probs.norm_gen_log_prob(x, mu_z, 1. / np.sqrt(tau))
    return log_p_pi + log_p_z + log_p_mu + log_p_x
  return log_joint


def plot(mu, data):
  fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

  ax.plot(data[:,0], data[:,1], 'k.')

  (log_weights,), _, (Exx, Ex), _ = mu
  for weight, second_moment, mean in zip(np.exp(log_weights), Exx, Ex):
    Sigma = np.diag(second_moment - mean**2)
    plot_ellipse(ax, weight, mean, Sigma)

  return fig


def plot_ellipse(ax, alpha, mean, cov):
  t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
  circle = np.vstack((np.sin(t), np.cos(t)))
  ellipse = np.dot(np.linalg.cholesky(cov), circle) + mean[:,None]
  ax.plot(ellipse[0], ellipse[1], alpha=1., linestyle='-', linewidth=2)


def main(argv):
  del argv

  n_clusters = FLAGS.num_clusters
  n_dimensions = FLAGS.num_dimensions
  n_observations = FLAGS.num_observations

  alpha = 3.3 * np.ones(n_clusters)
  a = 1.
  b = 1.
  kappa = 0.1

  npr.seed(10001)

  # generate true latents and data
  pi = npr.gamma(alpha)
  pi /= pi.sum()
  mu = npr.normal(0, 1.5, [n_clusters, n_dimensions])
  z = npr.choice(np.arange(n_clusters), size=n_observations, p=pi)
  x = npr.normal(mu[z, :], 0.5 ** 2)

  # points used for initialization
  pi_est = np.ones(n_clusters) / n_clusters
  z_est = npr.choice(np.arange(n_clusters), size=n_observations, p=pi_est)
  mu_est = npr.normal(0., 0.01, [n_clusters, n_dimensions])
  tau_est = 1.
  init_vals = pi_est, z_est, mu_est, tau_est

  # instantiate the model log joint
  log_joint = make_log_joint(x, alpha, a, b, kappa)

  # run mean field on variational mean parameters
  def callback(meanparams):
    fig = plot(meanparams, x)
    plt.savefig('/tmp/gmm_{:04d}.png'.format(itr))
    plt.close(fig.number)

  start = time.time()
  cavi(log_joint, init_vals, (SIMPLEX, INTEGER, REAL, NONNEGATIVE),
       FLAGS.num_iterations, callback=lambda *args: None)
  runtime = time.time() - start
  print("CAVI Runtime (s): ", runtime)


if __name__ == '__main__':
  app.run(main)
