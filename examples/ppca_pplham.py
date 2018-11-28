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
"""Probabilistic principal components analysis with PPLHam."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time
from absl import app
from absl import flags
from autograd import grad
from autograd.misc.optimizers import adam
import autograd.numpy as np
import matplotlib
from matplotlib import figure
from matplotlib.backends import backend_agg
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('fivethirtyeight')
plt.rc('axes', labelsize='15')
import six  # pylint: disable=g-import-not-at-top

from autoconj import conjugacy, log_probs
from autoconj import pplham as ph
from autoconj.meanfield import cavi
from autoconj.meanfield import elbo as elbo_fn

flags.DEFINE_list(
    'inference',
    default=['gibbs', 'advi', 'map', 'cavi'],
    help='Comma-separated list of algorithms to perform and compare across. '
         'Choices are gibbs, advi, map, cavi.')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'ppca/'),
    help='Directory to put the model\'s fit.')
flags.DEFINE_list(
    'num_iterations',
    default=['500', '7500', '15000', '1500'],
    help='Comma-separated list of training steps. Aligns with each algorithm.')
flags.DEFINE_integer('num_print',
                     default=25,
                     help='Print progress every many of these steps.')
flags.DEFINE_bool('plot_only',
                  default=None,
                  help='If True, only does plotting. Defaults to running.')
FLAGS = flags.FLAGS


def run_gibbs(log_joint_fn, all_args, num_iterations):
  """Train model with Gibbs sampling."""
  alpha, beta, epsilon, w, tau, x = all_args
  # Form complete conditionals for Gibbs sampling.
  epsilon_conditional_factory = conjugacy.complete_conditional(
      log_joint_fn,
      2,
      conjugacy.SupportTypes.REAL,
      *all_args)
  w_conditional_factory = conjugacy.complete_conditional(
      log_joint_fn,
      3,
      conjugacy.SupportTypes.REAL,
      *all_args)
  tau_conditional_factory = conjugacy.complete_conditional(
      log_joint_fn,
      4,
      conjugacy.SupportTypes.NONNEGATIVE,
      *all_args)
  epsilon_conditional = lambda w, tau: epsilon_conditional_factory(  # pylint: disable=g-long-lambda
      alpha, beta, w, tau, x)
  w_conditional = lambda epsilon, tau: w_conditional_factory(  # pylint: disable=g-long-lambda
      alpha, beta, epsilon, tau, x)
  tau_conditional = lambda epsilon, w: tau_conditional_factory(  # pylint: disable=g-long-lambda
      alpha, beta, epsilon, w, x)
  log_posterior = lambda epsilon, w, tau: log_joint_fn(  # pylint: disable=g-long-lambda
      alpha, beta, epsilon, w, tau, x)

  # Run training loop. Track expected log joint probability, i.e.,
  # E [ log p(xnew, params | xtrain) ]. It is estimated with 1 posterior sample.
  print('Running Gibbs...')
  epsilon = ph.norm.rvs(0, 1, size=epsilon.shape)
  w = ph.norm.rvs(0, 1, size=w.shape)
  tau = ph.gamma.rvs(alpha, scale=1./beta)
  log_joints = []
  runtimes = []
  start = time.time()
  for t in range(num_iterations):
    epsilon = epsilon_conditional(w, tau).rvs()
    w = w_conditional(epsilon, tau).rvs()
    tau = tau_conditional(epsilon, w).rvs()
    if t % FLAGS.num_print == 0 or (t + 1) == num_iterations:
      log_joint = log_posterior(epsilon, w, tau)
      runtime = time.time() - start
      print('Iteration: {:>3d} Log Joint: {:.3f} '
            'Runtime (s): {:.3f}'.format(t, log_joint, runtime))
      log_joints.append(log_joint)
      runtimes.append(runtime)
  return log_joints, runtimes


def run_advi(log_joint_fn, all_args, num_iterations, run_map=False):
  """Train model with automatic differentiation variational inference.

  Args:
    run_map: If True, runs ADVI with `E_q [ log p(data, params) ]` as loss
      function.
  """
  alpha, beta, epsilon, w, tau, x = all_args
  log_posterior = lambda epsilon, w, tau: log_joint_fn(  # pylint: disable=g-long-lambda
      alpha, beta, epsilon, w, tau, x)

  def unpack_params(params):
    """Unpacks `np.ndarray` into list of variational parameters."""
    param_shapes = [epsilon.shape,  # loc for q(epsilon)
                    epsilon.shape,  # log scale for q(epsilon)
                    w.shape,        # loc for q(w)
                    w.shape,        # log scale for q(w)
                    tau.shape,      # loc for q(tau)
                    tau.shape]      # log scale for q(tau)
    begin = 0
    end = 0
    unpacked_params = []
    for param_shape in param_shapes:
      end += int(np.prod(param_shape))  # accumulate by number of parameters
      param = params[begin:end].reshape(param_shape)
      begin = end
      unpacked_params.append(param)
    return unpacked_params

  def loss(params, t, return_marginal=False):
    """Reparameterization-based Monte Carlo estimate of negative ELBO."""
    del t  # unused
    unpacked_params = unpack_params(params)
    zs = []
    log_q = 0.
    # TODO(trandustin): Learn gamma latent with log transform. Currently, it is
    # fixed at its true value.
    # for t in range(3):
    for t in range(2):
      loc = unpacked_params[2 * t]  # 0, 2, 4
      log_scale = unpacked_params[2 * t + 1]  # 1, 3, 5
      z = loc + np.exp(log_scale) * np.random.normal(0, 1, size=log_scale.shape)
      zs.append(z)
      log_q += log_probs.norm_gen_log_prob(z, loc, np.exp(log_scale))

    zs.append(tau)
    log_p = log_posterior(*zs)  # pylint: disable=no-value-for-parameter
    if return_marginal:
      return log_p
    elif run_map:
      return -log_p
    return log_q - log_p

  def callback(params, t, g):
    """Callback for use in Autograd's optimizer routine."""
    del g  # unused
    if t % FLAGS.num_print == 0 or (t + 1) == num_iterations:
      log_joint = loss(params, t, return_marginal=True)
      elbo = -loss(params, t)
      runtime = time.time() - start
      print('Iteration: {:>3d} Log Joint: {:.3f} ELBO: {:.3f} '
            'Runtime (s): {:.3f}'.format(t, log_joint, elbo, runtime))
      log_joints.append(log_joint)
      elbos.append(elbo)
      runtimes.append(runtime)
    return

  grad_loss = grad(loss)

  # TODO(trandustin): why is the ELBO positive?
  # Run training loop. Track expected log joint probability, i.e.,
  # E [ log p(xnew, params | xtrain) ]. It is estimated with 1 posterior sample.
  if run_map:
    print('Running MAP...')
  else:
    print('Running ADVI...')
  num_params = int(2 * np.prod(epsilon.shape) +
                   2 * np.prod(w.shape) +
                   2 * np.prod(tau.shape))
  print('Number of parameters: ', num_params)
  # TODO(trandustin): use lists of params
  # Initialize randomly near 0 for means and largely negative for log stddev.
  params = np.concatenate([
      np.random.normal(0, 1, size=int(np.prod(epsilon.shape))),
      np.random.normal(-3, 1e-3, size=int(np.prod(epsilon.shape))),
      np.random.normal(0, 1, size=int(np.prod(w.shape))),
      np.random.normal(-3, 1e-3, size=int(np.prod(w.shape))),
      np.random.normal(0, 1, size=int(np.prod(tau.shape))),
      np.random.normal(-3, 1e-3, size=int(np.prod(tau.shape)))], 0)
  log_joints = []
  elbos = []
  runtimes = []
  start = time.time()
  params = adam(grad_loss,
                params,
                callback=callback,
                num_iters=num_iterations,
                step_size=1e-2)
  return log_joints, runtimes, elbos


def run_cavi(log_joint_fn, all_args, num_iterations):
  """Train model with coordinate-ascent variational inference."""
  alpha, beta, epsilon, w, tau, x = all_args
  log_posterior = lambda epsilon, w, tau: log_joint_fn(  # pylint: disable=g-long-lambda
      alpha, beta, epsilon, w, tau, x)

  def callback(t, neg_energy, normalizers, natparams):
    """Callback for use in CAVI routine."""
    if t % FLAGS.num_print == 0 or (t + 1) == num_iterations:
      elbo, log_joint = elbo_fn(neg_energy, normalizers, natparams,
                                return_lp=True)
      runtime = time.time() - start
      print('Iteration: {:>3d} Log Joint: {:.3f} ELBO: {:.3f} '
            'Runtime (s): {:.3f}'.format(t, log_joint, elbo, runtime))
      log_joints.append(log_joint)
      elbos.append(elbo)
      runtimes.append(runtime)
    return

  # Run training loop. Track expected log joint probability, i.e.,
  # E [ log p(xnew, params | xtrain) ]. It is estimated with 1 posterior sample.
  print('Running CAVI...')
  epsilon = ph.norm.rvs(0, 1, size=epsilon.shape)
  w = ph.norm.rvs(0, 1, size=w.shape)
  tau = ph.gamma.rvs(alpha, scale=1./beta)
  log_joints = []
  elbos = []
  runtimes = []
  start = time.time()
  _ = cavi(log_posterior,
           init_vals=(epsilon, w, tau),
           supports=(conjugacy.SupportTypes.REAL,
                     conjugacy.SupportTypes.REAL,
                     conjugacy.SupportTypes.NONNEGATIVE),
           num_iters=num_iterations,
           callback=callback)
  return log_joints, runtimes, elbos


def main(argv):
  del argv  # Unused.
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
  FLAGS.num_iterations = [int(i) for i in FLAGS.num_iterations]

  def model(alpha, beta):
    """Generates matrix of shape [num_examples, num_features]."""
    epsilon = ph.norm.rvs(0, 1, size=[num_examples, num_latents])
    w = ph.norm.rvs(0, 1, size=[num_features, num_latents])
    tau = ph.gamma.rvs(alpha, beta)
    # TODO(trandustin): try that this works
    # x = ph.norm.rvs(np.dot(epsilon, w.T), 1. / np.sqrt(tau))
    x = ph.norm.rvs(np.einsum('ik,jk->ij', epsilon, w), 1. / np.sqrt(tau))
    return [epsilon, w, tau, x]

  if FLAGS.plot_only:
    # Load results from CSV.
    # TODO(trandustin): refactor data structures. this is messy..
    inference_algs = []
    xs = []
    ys = []
    fname = os.path.join(FLAGS.model_dir, 'results_lp.csv')
    print('Loading {}'.format(fname))
    with open(fname, 'rb') as f:
      reader = csv.reader(f)
      for i, row in enumerate(reader):
        if i % 3 == 0:
          inference_algs.append(row)
        elif i % 3 == 1:
          xs.append(row)
        else:
          ys.append(row)
    results_lp = {inference_alg[0]: [x, y]
                  for inference_alg, x, y in zip(inference_algs, xs, ys)}

    inference_algs = []
    xs = []
    ys = []
    fname = os.path.join(FLAGS.model_dir, 'results_elbo.csv')
    print('Loading {}'.format(fname))
    with open(fname, 'rb') as f:
      reader = csv.reader(f)
      for i, row in enumerate(reader):
        if i % 3 == 0:
          inference_algs.append(row)
        elif i % 3 == 1:
          xs.append(row)
        else:
          ys.append(row)
    results_elbo = {inference_alg[0]: [x, y]
                    for inference_alg, x, y in zip(inference_algs, xs, ys)}
  else:
    # Use synthetic data generated from model.
    num_examples = 100
    num_features = 20
    num_latents = 5
    alpha = 2.
    beta = 8.
    epsilon, w, tau, x = model(alpha, beta)
    all_args = [alpha, beta, epsilon, w, tau, x]

    log_joint_fn = ph.make_log_joint_fn(model)
    results_lp = {}
    results_elbo = {}
    for inference_alg, num_iters in zip(FLAGS.inference, FLAGS.num_iterations):
      if inference_alg == 'gibbs':
        log_joints, runtimes = run_gibbs(log_joint_fn, all_args, num_iters)
      elif inference_alg == 'advi':
        log_joints, runtimes, elbos = run_advi(log_joint_fn, all_args, num_iters)
        results_elbo[inference_alg] = [runtimes, elbos]
      elif inference_alg == 'map':
        log_joints, runtimes, _ = run_advi(log_joint_fn, all_args, num_iters,
                                           run_map=True)
      elif inference_alg == 'cavi':
        log_joints, runtimes, elbos = run_cavi(log_joint_fn, all_args, num_iters)
        results_elbo[inference_alg] = [runtimes, elbos]
      else:
        raise NotImplementedError("Only 'gibbs', 'advi', 'map', 'cavi' is "
                                  "implemented.")
      results_lp[inference_alg] = [runtimes, log_joints]

    # Write results to CSV to easily tweak plots and not have to rerun training.
    fname = os.path.join(FLAGS.model_dir, 'results_lp.csv')
    with open(fname, 'wb') as f:
      writer = csv.writer(f, quoting=csv.QUOTE_ALL)
      for inference_alg, (x, y) in six.iteritems(results_lp):
        writer.writerow([inference_alg])
        writer.writerow(x)
        writer.writerow(y)
    print('Saved {}'.format(fname))

    fname = os.path.join(FLAGS.model_dir, 'results_elbo.csv')
    with open(fname, 'wb') as f:
      writer = csv.writer(f, quoting=csv.QUOTE_ALL)
      for inference_alg, (x, y) in six.iteritems(results_elbo):
        writer.writerow([inference_alg])
        writer.writerow(x)
        writer.writerow(y)
    print('Saved {}'.format(fname))

  labels = {'gibbs': 'Gibbs', 'advi': 'ADVI', 'map': 'MAP', 'cavi': 'CAVI'}

  # Plot ELBO by runtime (s).
  figsize = (10, 5)

  fig = figure.Figure(figsize=figsize)
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax = fig.add_subplot(1, 1, 1)
  for inference_alg, (x, y) in six.iteritems(results_elbo):
    ax.plot(x, y, label=labels[inference_alg])
  ax.set_xlabel('Runtime (s)')
  ax.set_ylabel('ELBO')
  ax.legend(loc='lower right')
  fname = os.path.join(FLAGS.model_dir, 'elbo-over-runtime.png')
  canvas.print_figure(fname, format='png')
  print('Saved {}'.format(fname))
  fname = os.path.join(FLAGS.model_dir, 'elbo-over-runtime.pdf')
  canvas.print_figure(fname, format='pdf')
  print('Saved {}'.format(fname))

  # Plot expected log joint density by runtime (s).
  # TODO(trandustin): calculate log posterior predictive (expected log
  # likelihood), not expected log joint.
  fig = figure.Figure(figsize=figsize)
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax = fig.add_subplot(1, 1, 1)
  for inference_alg, (x, y) in six.iteritems(results_lp):
    ax.plot(x, y, label=labels[inference_alg])
  ax.set_xlabel('Runtime (s)')
  ax.set_ylabel('Expected Log Joint')
  ax.legend(loc='lower right')
  fname = os.path.join(FLAGS.model_dir, 'log-joint-over-runtime.png')
  canvas.print_figure(fname, format='png', bbox_inches='tight')
  print('Saved {}'.format(fname))
  fname = os.path.join(FLAGS.model_dir, 'log-joint-over-runtime.pdf')
  canvas.print_figure(fname, format='pdf', bbox_inches='tight')
  print('Saved {}'.format(fname))

  fig = figure.Figure(figsize=figsize)
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax = fig.add_subplot(1, 1, 1)
  for inference_alg, (x, y) in six.iteritems(results_lp):
    if inference_alg == 'advi':
      continue
    ax.plot(x, y, label=labels[inference_alg])
  ax.set_xlabel('Runtime (s)')
  ax.set_ylabel('Expected Log Joint')
  ax.set_ylim((-10000.0, -0.0))
  ax.legend(loc='lower right')
  fname = os.path.join(FLAGS.model_dir, 'log-joint-over-runtime-zoom.png')
  canvas.print_figure(fname, format='png', bbox_inches='tight')
  print('Saved {}'.format(fname))
  fname = os.path.join(FLAGS.model_dir, 'log-joint-over-runtime-zoom.pdf')
  canvas.print_figure(fname, format='pdf', bbox_inches='tight')
  print('Saved {}'.format(fname))

if __name__ == '__main__':
  app.run(main)
