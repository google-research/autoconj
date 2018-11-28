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
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr

from autoconj import pplham as ph
from autoconj.util import SupportTypes
from autoconj.meanfield import cavi

REAL = SupportTypes.REAL
NONNEGATIVE = SupportTypes.NONNEGATIVE


def main(unused_argv):
  npr.seed(10001)

  def make_model(alpha, beta):
    """Generates matrix of shape [num_examples, num_features]."""
    def sample_model():
      epsilon = ph.norm.rvs(0, 1, size=[num_examples, num_latents])
      w = ph.norm.rvs(0, 1, size=[num_features, num_latents])
      tau = ph.gamma.rvs(alpha, beta)
      x = ph.norm.rvs(np.dot(epsilon, w.T), 1. / np.sqrt(tau))
      return [epsilon, w, tau, x]
    return sample_model

  num_examples = 50
  num_features = 10
  num_latents = 5
  alpha = 2.
  beta = 8.
  sampler = make_model(alpha, beta)
  _, _, _, x = sampler()
  epsilon, w, tau, _ = sampler()  # initialization

  log_joint_fn_ = ph.make_log_joint_fn(sampler)
  log_joint_fn = lambda *args: log_joint_fn_(*(args + (x,)))  # crappy partial

  cavi(log_joint_fn, (epsilon, w, tau), (REAL, REAL, NONNEGATIVE), 50)


if __name__ == '__main__':
  app.run(main)
