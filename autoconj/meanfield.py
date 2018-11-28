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
"""Conjugate meanfield functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad
from autograd import fmap_util

from . import conjugacy


def elbo(neg_energy, normalizers, eta, return_lp=False):
  # TODO(trandustin): more efficiently return various parts of elbo
  logZ = total_normalizer(normalizers)
  val, mu = value_and_grad(logZ)(eta)
  lp = neg_energy(*mu)
  elbo_val = lp - (dot(eta, mu) - val)
  if return_lp:
    return elbo_val, lp
  return elbo_val


def cavi(log_joint, init_vals, supports, num_iters, callback=None):
  if not callback:
    callback = lambda t, neg_energy, normalizers, natparams: print(elbo(neg_energy, normalizers, natparams))
  neg_energy, normalizers, _, initializers, _, _ = \
      conjugacy.multilinear_representation(log_joint, init_vals, supports)

  natparams = [initializer(10.) for initializer in initializers]
  meanparams = [grad(normalizer)(natparam)
                for normalizer, natparam in zip(normalizers, natparams)]

  callback(-1, neg_energy, normalizers, natparams)

  for t in range(num_iters):
    for i, normalizer in reversed(list(enumerate(normalizers))):
      natparams[i] = grad(neg_energy, i)(*meanparams)
      meanparams[i] = grad(normalizer)(natparams[i])

    callback(t, neg_energy, normalizers, natparams)

  return natparams, meanparams


### util


def dot(a, b):
  tot = [0.]
  def _dot(a, b):
    tot[0] += np.dot(np.ravel(a), np.ravel(b))
  fmap_util.container_fmap(_dot, a, b)
  return tot[0]


def total_normalizer(normalizers):
  def logZ(eta):
    return sum([norm(eta[i]) for i, norm in enumerate(normalizers)])
  return logZ
