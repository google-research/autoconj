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
"""Functions for exploiting graphical model structure in random variables,
once it has been identified. This includes efficient log-normalizers for
tree-structured Gaussian and Categorical distributions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import autograd.numpy as np
from autograd.scipy.misc import logsumexp

from .tracers import logdet


### Tree normal log normalizer
def diag(value):
  return np.einsum('...i,j,ij->...ij', value, np.ones(value.shape[-1]),
                   np.eye(value.shape[-1]))


def convert_to_info_form(natparam):
  info_natparam = natparam.__class__(**{k: defaultdict(int)
                                        for k in natparam._fields})
  for factor, val in natparam.xi_xjtrs.iteritems():
    info_natparam.xi_xjtrs[factor] += -val
  for factor, val in natparam.xi_times_xjs.iteritems():
    info_natparam.xi_xjtrs[factor] += -diag(val)
  for factor, val in natparam.xi_xitrs.iteritems():
    info_natparam.xi_xitrs[factor] += -2*val
  for factor, val in natparam.xi_squareds.iteritems():
    info_natparam.xi_xitrs[factor] += -2*diag(val)
  for factor, val in natparam.xis.iteritems():
    info_natparam.xis[factor] += val
  return info_natparam


def make_tree_normal_log_normalizer(elim_order):
  def tree_normal_log_normalizer(natparam):
    log_normalizer = 0
    natparam = convert_to_info_form(natparam)
    for node in elim_order:
      # Find joint factor node participates in, if it exists.
      joint_onehot_xis = [factor for factor in natparam.xi_xjtrs.keys()
                          if node in factor]
      joint_factor = joint_onehot_xis[0] if joint_onehot_xis else None

      # Retrieve natural parameters (in information form) of factors node
      # participates in.
      single_J = natparam.xi_xitrs.pop((node,), 0)
      single_h = natparam.xis.pop((node,), np.zeros(single_J.shape[-1]))
      joint_J = natparam.xi_xjtrs.pop(joint_factor, 0)

      # After marginalizing, accumulate the result into log normalizer.
      inv_single_J = np.linalg.inv(single_J)
      dim_node = single_J.shape[0]
      log_normalizer += 0.5*np.dot(single_h, np.dot(inv_single_J, single_h))
      log_normalizer -= 0.5*logdet(single_J) + 0.5*dim_node*np.log(2*np.pi)

      # Compute message to other node in joint factor, if it exists.
      if joint_factor:
        node_pos = joint_factor.index(node)
        other_pos = (node_pos + 1) % 2
        joint_J = joint_J.transpose((node_pos, other_pos))

        msg_h = -np.dot(joint_J.T, np.dot(inv_single_J, single_h))
        msg_J = -np.dot(joint_J.T, np.dot(inv_single_J, joint_J))

        other_node = joint_factor[other_pos]
        natparam.xis[(other_node,)] += msg_h
        natparam.xi_xitrs[(other_node,)] += msg_J

    return log_normalizer

  return tree_normal_log_normalizer


### Tree categorical
def make_tree_categorical_collapser(elim_order, collapse_fun):
  def tree_categorical_collapser(natparam_original):
    # Make a copy of the natural parameters so we can mutate without outside
    # effects.
    natparam = natparam_original.__class__(
        **{k: defaultdict(int, v) for k, v
           in natparam_original._asdict().iteritems()})
    # Eliminate nodes.
    for node in elim_order:
      # Find single and joint factors node participates in, if they exist.
      joint_onehot_xis = [factor for factor in natparam.joint_onehot_xis.keys()
                          if node in factor]
      joint_factor = joint_onehot_xis[0] if joint_onehot_xis else ()

      # Retrieve natural parameters of factors node participates in.
      log_single_param = natparam.single_onehot_xis.pop((node,), 0)
      log_joint_param = natparam.joint_onehot_xis.pop(joint_factor, 0)

      # Rearrange log_joint_param for broadcasting.
      node_pos = joint_factor.index(node) if joint_factor else 0
      if joint_factor:
        old_axes = tuple(range(log_joint_param.ndim))
        new_axes = old_axes[:node_pos] + old_axes[node_pos+1:] + (node_pos,)
        log_joint_param = np.transpose(log_joint_param, new_axes)

      # Construct new collapsed factor.
      collapsed_factor = joint_factor[:node_pos] + joint_factor[node_pos+1:]
      log_collapsed_param = collapse_fun(log_single_param + log_joint_param,
                                         axis=-1)
      if len(collapsed_factor) <= 1:
        original_param = natparam.single_onehot_xis[collapsed_factor]
        natparam.single_onehot_xis[collapsed_factor] = (log_collapsed_param +
                                                        original_param)
      else:
        original_param = natparam.joint_onehot_xis[collapsed_factor]
        natparam.joint_onehot_xis[collapsed_factor] = (log_collapsed_param +
                                                       original_param)

    return natparam.single_onehot_xis[()]
  return tree_categorical_collapser

make_tree_categorical_log_normalizer =(
    lambda elim_order: make_tree_categorical_collapser(elim_order,
                                                       collapse_fun=logsumexp))
make_tree_categorical_maximum = (
    lambda elim_order: make_tree_categorical_collapser(elim_order,
                                                       collapse_fun=np.max))
