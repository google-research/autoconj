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
"""Normal-Gamma model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import autograd.numpy as np

from autoconj import conjugacy, log_probs


def log_joint(tau, mu, x, alpha, beta, kappa, mu0):
  log_p_tau = log_probs.gamma_gen_log_prob(tau, alpha, beta)
  log_p_mu = log_probs.norm_gen_log_prob(mu, mu0, 1. / np.sqrt(kappa * tau))
  log_p_x = log_probs.norm_gen_log_prob(x, mu, 1. / np.sqrt(tau))
  return log_p_tau + log_p_mu + log_p_x


def main(argv):
  del argv  # Unused.

  n_examples = 10
  a = 1.3
  b = 2.2
  kappa = 1.5
  mu0 = 0.3
  tau = np.random.gamma(a, 1. / b)
  mu = np.random.normal(mu0, 1. / np.sqrt(tau * kappa))
  x = np.random.normal(mu, 1. / np.sqrt(tau), n_examples)
  all_args = [tau, mu, x, a, b, kappa, mu0]
  all_args_ex_mu = [tau, x, a, b, kappa, mu0]

  mu_conditional_factory = conjugacy.complete_conditional(
      log_joint, 1, conjugacy.SupportTypes.REAL, *all_args)
  mu_conditional = mu_conditional_factory(*all_args_ex_mu)

  log_p_tau = conjugacy.marginalize(log_joint, 1, conjugacy.SupportTypes.REAL,
                                    *all_args)
  tau_conditional_factory = conjugacy.complete_conditional(
      log_p_tau, 0, conjugacy.SupportTypes.NONNEGATIVE, *all_args_ex_mu)
  tau_conditional = tau_conditional_factory(*all_args_ex_mu[1:])

  print('True tau: {}'.format(tau))
  print('tau posterior is gamma({}, {}). Mean is {}, std. dev. is {}.'.format(
      tau_conditional.args[0], 1. / tau_conditional.args[2],
      tau_conditional.args[0] * tau_conditional.args[2],
      np.sqrt(tau_conditional.args[0]) * tau_conditional.args[2]))
  print()
  print('True mu: {}'.format(mu))
  print('mu posterior given tau is normal({}, {})'.format(
      mu_conditional.args[0], mu_conditional.args[1]))

if __name__ == '__main__':
  app.run(main)
