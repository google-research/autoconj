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
from scipy import stats

from autoconj import log_probs


class LogProbTest(absltest.TestCase):

  def testCategoricalGenLogProb(self):
    x = np.array(2)
    x_one_hot = np.array([0, 0, 1])
    p = np.array([0.4, 0.25, 0.35])
    value = log_probs.categorical_gen_log_prob(x, p=p)
    true_value = stats.multinomial.logpmf(x_one_hot, n=1, p=p)
    self.assertAlmostEqual(value, true_value)

    xs_one_hot = stats.multinomial.rvs(n=1, p=p, size=[10])
    xs = np.argmax(xs_one_hot, axis=1)
    value = sum([log_probs.categorical_gen_log_prob(xs[i], p=p)
                 for i in range(xs.shape[0])])
    true_value = sum([stats.multinomial.logpmf(xs_one_hot[i], n=1, p=p)
                      for i in range(xs_one_hot.shape[0])])
    self.assertAlmostEqual(value, true_value)

  def testDirichletGenLogProb(self):
    x = np.array([0.4, 0.25, 0.35])
    alpha = np.array([2.12, 0.54, 1.6])
    value = log_probs.dirichlet_gen_log_prob(x, alpha=alpha)
    true_value = stats.dirichlet.logpdf(x, alpha=alpha)
    self.assertAlmostEqual(value, true_value)

    xs = stats.dirichlet.rvs(alpha=alpha, size=[10])
    value = sum([log_probs.dirichlet_gen_log_prob(xs[i], alpha=alpha)
                 for i in range(xs.shape[0])])
    true_value = sum([stats.dirichlet.logpdf(xs[i], alpha=alpha)
                      for i in range(xs.shape[0])])
    self.assertAlmostEqual(value, true_value)

  def testMultinomialGenLogProb(self):
    x = np.array([0, 0, 1])
    n = 1
    p = np.array([0.4, 0.25, 0.35])
    value = log_probs.multinomial_gen_log_prob(x, n=n, p=p)
    true_value = stats.multinomial.logpmf(x, n=n, p=p)
    self.assertAlmostEqual(value, true_value)

    xs = stats.multinomial.rvs(n=n, p=p, size=[10])
    value = sum([log_probs.multinomial_gen_log_prob(xs[i], n=n, p=p)
                 for i in range(xs.shape[0])])
    true_value = sum([stats.multinomial.logpmf(xs[i], n=n, p=p)
                      for i in range(xs.shape[0])])
    self.assertAlmostEqual(value, true_value)

  def testNormGenLogProb(self):
    x = 2.3
    loc = 0.3
    scale = 1.0
    value = log_probs.norm_gen_log_prob(x, loc=loc, scale=scale)
    true_value = stats.norm.logpdf(x, loc=loc, scale=scale)
    self.assertAlmostEqual(value, true_value)

    x = stats.norm.rvs(loc=loc, scale=scale, size=[10])
    value = log_probs.norm_gen_log_prob(x, loc=loc, scale=scale)
    true_value = sum(stats.norm.logpdf(x, loc=loc, scale=scale))
    self.assertAlmostEqual(value, true_value)

if __name__ == '__main__':
  np.random.seed(3251)
  absltest.main()
