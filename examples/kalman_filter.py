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

import time

from autograd import grad
import autograd.numpy as np

from absl import app

from autograd import grad
from autoconj.conjugacy import complete_conditional, marginalize, SupportTypes
from autoconj import canonicalize, conjugacy, log_probs, tracers


def log_p_x1_y1(x1, y1, x1_scale, y1_scale):
  log_p_x1 = log_probs.norm_gen_log_prob(x1, 0, x1_scale)
  log_p_y1_given_x1 = log_probs.norm_gen_log_prob(y1, x1, y1_scale)
  return log_p_x1 + log_p_y1_given_x1


def log_p_xt_xtt_ytt(xt, xtt, ytt, xt_prior_mean, xt_prior_scale, x_scale,
                     y_scale):
  log_p_xt = log_probs.norm_gen_log_prob(xt, xt_prior_mean, xt_prior_scale)
  log_p_xtt = log_probs.norm_gen_log_prob(xtt, xt, x_scale)
  log_p_ytt = log_probs.norm_gen_log_prob(ytt, xtt, y_scale)
  return log_p_xt + log_p_xtt + log_p_ytt


def make_marginal_fn():
  x1_given_y1_factory = complete_conditional(
      log_p_x1_y1, 0, SupportTypes.REAL, *([1.] * 4))
  log_p_y1 = marginalize(log_p_x1_y1, 0, SupportTypes.REAL, *([1.] * 4))

  log_p_xtt_ytt = marginalize(
      log_p_xt_xtt_ytt, 0, SupportTypes.REAL, *([1.] * 7))
  log_p_ytt = marginalize(
      log_p_xtt_ytt, 0, SupportTypes.REAL, *([1.] * 6))
  xt_conditional_factory = complete_conditional(
      log_p_xtt_ytt, 0, SupportTypes.REAL, *([1.] * 6))

  def marginal(y_list, x_scale, y_scale):
    log_p_y = log_p_y1(y_list[0], x_scale, y_scale)
    xt_conditional = x1_given_y1_factory(y_list[0], x_scale, y_scale)

    for t in range(1, len(y_list)):
      log_p_y += log_p_ytt(y_list[t], xt_conditional.args[0],
                           xt_conditional.args[1], x_scale, y_scale)
      xt_conditional = xt_conditional_factory(
          y_list[t], xt_conditional.args[0], xt_conditional.args[1], x_scale,
          y_scale)
    return log_p_y
  return marginal

def main(argv):
  del argv  # Unused.

  x_scale = 0.1
  y_scale = 1.
  T = 50

  x_list = np.cumsum(x_scale * np.random.randn(T))
  y_list = np.array([x_list[t] + y_scale * np.random.randn() for t in range(T)])

  marginal = make_marginal_fn()
  marginal_grad = grad(lambda y_list, scales: marginal(y_list, *scales), 1)

  x_scale_est = 0.1
  y_scale_est = 1.
  step_size = 0.5 / T
  for i in range(100):
    t0 = time.time()
    x_scale_grad, y_scale_grad = marginal_grad(
        y_list, (x_scale_est, y_scale_est))
    x_scale_est *= np.exp(step_size * x_scale_est * x_scale_grad)
    y_scale_est *= np.exp(step_size * y_scale_est * y_scale_grad)
    print('{}\t{}\t{}\t{}\t{}'.format(
        time.time() - t0, i, marginal(y_list, x_scale_est, y_scale_est),
        x_scale_est, y_scale_est))


if __name__ == '__main__':
  app.run(main)
