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

from absl.testing import absltest

import autograd.numpy as np

from autoconj import log_probs
from autoconj import pplham as ph


class PPLHamTest(absltest.TestCase):

  def testMakeLogJointUnconditional(self):
    """Test `make_log_joint` works on unconditional model."""
    def model():
      loc = ph.norm.rvs(loc=0.0, scale=1.0, name="loc")
      x = ph.norm.rvs(loc=loc, scale=0.5, size=5, name="x")
      return x

    log_joint = ph.make_log_joint_fn(model)

    x = np.random.normal(size=5)
    loc = 0.3

    value = log_joint(loc=loc, x=x)
    true_value = log_probs.norm_gen_log_prob(loc, loc=0.0, scale=1.0)
    true_value += log_probs.norm_gen_log_prob(x, loc=loc, scale=0.5)
    self.assertAlmostEqual(value, true_value)

  def testMakeLogJointConditional(self):
    """Test `make_log_joint` works on conditional model."""
    def model(X, prior_precision):
      beta = ph.norm.rvs(loc=0.0,
                         scale=1.0 / np.sqrt(prior_precision),
                         size=X.shape[1],
                         name="beta")
      loc = np.einsum('ij,j->i', X, beta)
      y = ph.norm.rvs(loc=loc, scale=1.0, name="y")
      return y

    log_joint = ph.make_log_joint_fn(model)

    X = np.random.normal(size=[3, 2])
    prior_precision = 0.5
    beta = np.random.normal(size=[2])
    y = np.random.normal(size=[3])

    true_value = log_probs.norm_gen_log_prob(
        beta, loc=0.0, scale=1.0 / np.sqrt(prior_precision))
    loc = np.einsum('ij,j->i', X, beta)
    true_value += log_probs.norm_gen_log_prob(y, loc=loc, scale=1.0)

    # Test args as input.
    value = log_joint(X, prior_precision, beta, y)
    self.assertAlmostEqual(value, true_value)

    # Test kwargs as input.
    value = log_joint(X, prior_precision, y=y, beta=beta)
    self.assertAlmostEqual(value, true_value)

if __name__ == '__main__':
  np.random.seed(8327)
  absltest.main()
