# Copyright 2020 Google LLC
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

import numpy as np
from scipy.special import logsumexp, softmax, binom
import functools

from absl.testing import absltest
from absl.testing import parameterized

from sdtw_div.numba_ops import sdtw, sdtw_C, sdtw_grad_C
from sdtw_div.numba_ops import sdtw_value_and_grad_C
from sdtw_div.numba_ops import sdtw_value_and_grad
from sdtw_div.numba_ops import sdtw_directional_derivative_C
from sdtw_div.numba_ops import sdtw_hessian_product_C
from sdtw_div.numba_ops import sdtw_entropy, sdtw_entropy_C
from sdtw_div.numba_ops import sdtw_div
from sdtw_div.numba_ops import sdtw_div_value_and_grad

from sdtw_div.numba_ops import sharp_sdtw, sharp_sdtw_C
from sdtw_div.numba_ops import sharp_sdtw_value_and_grad_C
from sdtw_div.numba_ops import sharp_sdtw_value_and_grad
from sdtw_div.numba_ops import sharp_sdtw_div
from sdtw_div.numba_ops import sharp_sdtw_div_value_and_grad

from sdtw_div.numba_ops import mean_cost, mean_cost_C
from sdtw_div.numba_ops import mean_cost_value_and_grad_C
from sdtw_div.numba_ops import mean_cost_value_and_grad
from sdtw_div.numba_ops import mean_cost_div
from sdtw_div.numba_ops import mean_cost_div_value_and_grad

from sdtw_div.numba_ops import squared_euclidean_cost
from sdtw_div.numba_ops import squared_euclidean_cost_vjp
from sdtw_div.numba_ops import squared_euclidean_cost_jvp
from sdtw_div.numba_ops import squared_euclidean_distance

from sdtw_div.numba_ops import cardinality
from sdtw_div.numba_ops import alignment_matrices

from sdtw_div.numba_ops import barycenter

from nose.tools import assert_equal
from nose.tools import assert_almost_equal
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


def _make_time_series(size_X, size_Y, num_dim):
  rng = np.random.RandomState(0)
  X = rng.randn(size_X, num_dim)
  Y = rng.randn(size_Y, num_dim)
  return X, Y


def _num_gradient(f, C, eps=1e-6):
  G = np.zeros_like(C)
  for i in range(C.shape[0]):
    for j in range(C.shape[1]):
      Z = np.zeros_like(C)
      Z[i, j] = 1
      G[i, j] = (f(C + eps * Z) - f(C - eps * Z)) / (2 * eps)
  return G


def _sdtw_brute_force(C, gamma=1.0):
  scores = []
  for A in alignment_matrices(*C.shape):
    scores.append(np.vdot(A, C))
  scores = np.array(scores)
  return -gamma * logsumexp(-scores / gamma)


def _probas(C, gamma=1.0):
  scores = []
  for A in alignment_matrices(*C.shape):
    scores.append(np.vdot(A, C))
  scores = np.array(scores)
  return softmax(-scores / gamma)


def _expectation_brute_force(C, gamma=1.0):
  p = _probas(C, gamma=gamma)
  E = np.zeros_like(C)
  for i, A in enumerate(alignment_matrices(*C.shape)):
    E += p[i] * A
  return E

class SdtwNumbaTests(parameterized.TestCase):

  def test_sdtw(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    assert_almost_equal(sdtw_C(C), _sdtw_brute_force(C))
    assert_almost_equal(sdtw_C(C), sdtw_value_and_grad_C(C)[0])
    assert_almost_equal(sdtw_C(C), sdtw(X, Y))

    assert_almost_equal(sdtw_C(C, gamma=0.1), _sdtw_brute_force(C, gamma=0.1))
    assert_almost_equal(sdtw_C(C, gamma=0.1), sdtw_value_and_grad_C(C, gamma=0.1)[0])


  def test_sdtw_gamma0(self):
    X = _make_time_series(size_X=3, size_Y=4, num_dim=2)[0]

    # When gamma is small, dtw(C(X,X)) should be close to zero.
    assert_almost_equal(sdtw(X, X, gamma=1e-5), 0)


  def test_sdtw_grad_C(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    V, P = sdtw_C(C, return_all=True)
    G = sdtw_grad_C(P)
    G_num = _num_gradient(sdtw_C, C)
    G_bf = _expectation_brute_force(C)
    assert_array_almost_equal(G, G_num)
    assert_array_almost_equal(G, G_bf)

    # Check value_and_grad.
    for gamma in (0.1, 1.0):
      G = sdtw_value_and_grad_C(C, gamma=gamma)[1]
      G_bf = _expectation_brute_force(C, gamma=gamma)
      assert_array_almost_equal(G, G_bf)


  def test_sdtw_grad(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)

    value, G = sdtw_value_and_grad(X, Y)
    func = functools.partial(sdtw, Y=Y)
    G_num = _num_gradient(func, X)
    assert_almost_equal(value, func(X))
    assert_array_almost_equal(G, G_num)


  def test_sdtw_grad_C_gamma0(self):
    X = _make_time_series(size_X=3, size_Y=4, num_dim=2)[0]

    V, P = sdtw(X, X, gamma=1e-5, return_all=True)
    G = sdtw_grad_C(P)
    # When gamma is small, G should be close to the identity matrix.
    assert_array_almost_equal(G, np.eye(len(X)))


  def test_sdtw_directional_derivative_C(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    M = np.dot(X, Y.T)

    V, P = sdtw(X, Y, return_all=True)
    val = sdtw_directional_derivative_C(P, M)
    G = sdtw_grad_C(P)
    assert_array_almost_equal(np.vdot(G, M), val)

    # Check that wrong inputs raise an exception.
    assert_raises(ValueError, sdtw_directional_derivative_C, P, X)


  def test_sdtw_hessian_product(self):
    def f(C, M):
      V, P = sdtw_C(C, return_all=True)
      return sdtw_directional_derivative_C(P, M)

    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    V, P = sdtw_C(C, return_all=True)
    E = sdtw_grad_C(P, return_all=True)
    V_dot = sdtw_directional_derivative_C(P, C, return_all=True)
    hvp = sdtw_hessian_product_C(P, E, V_dot)
    hvp_num = _num_gradient(functools.partial(f, M=C), C)
    # The Hessian product is equal to the gradient of the directional derivative.
    assert_array_almost_equal(hvp, hvp_num)

    # Check that wrong inputs raise an exception.
    assert_raises(ValueError, sdtw_hessian_product_C, P, E, X)
    assert_raises(ValueError, sdtw_hessian_product_C, P, X, V_dot)


  def test_sdtw_entropy(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    gamma = 1.0
    eps = 1e-6
    p = _probas(C, gamma=gamma)
    H1 = -np.dot(p, np.log(p))
    H2 = sdtw_entropy_C(C, gamma=gamma)
    H3 = -(sdtw_C(C, gamma + eps) - sdtw_C(C, gamma - eps)) / (2 * eps)
    H4 = sdtw_entropy(X, Y, gamma=gamma)
    assert_almost_equal(H1, H2)
    assert_almost_equal(H1, H3)
    assert_almost_equal(H1, H4)


  def test_sharp_sdtw(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    for gamma in (0.1, 1.0):
      E = sdtw_value_and_grad_C(C, gamma=gamma)[1]
      val1 = np.vdot(E, C)
      val2 = sharp_sdtw_C(C, gamma=gamma)
      val3 = sharp_sdtw_value_and_grad_C(C, gamma=gamma)[0]
      val4 = sharp_sdtw_value_and_grad(X, Y, gamma=gamma)[0]
      val5 = sharp_sdtw(X, Y, gamma=gamma)
      assert_almost_equal(val1, val2)
      assert_almost_equal(val1, val3)
      assert_almost_equal(val1, val4)
      assert_almost_equal(val1, val5)


  def test_sharp_sdtw_grad(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    for gamma in (0.1, 1.0):
      G = sharp_sdtw_value_and_grad_C(C, gamma=gamma)[1]
      f = functools.partial(sharp_sdtw_C, gamma=gamma)
      G_num = _num_gradient(f, C)
      assert_array_almost_equal(G, G_num)

      G = sharp_sdtw_value_and_grad(X, Y, gamma=gamma)[1]
      f = functools.partial(sharp_sdtw, Y=Y, gamma=gamma)
      G_num = _num_gradient(f, X)
      assert_array_almost_equal(G, G_num)


  def test_cardinality(self):
    def delannoy(m, n):
      s = 0
      for k in range(min(m, n) + 1):
        s += binom(m, k) * binom(n, k) * 2 ** k
      return s

    for size_X in (1, 2, 3):
      for size_Y in (1, 2, 3, 4):
        card = cardinality(size_X, size_Y)
        card2 = len(list(alignment_matrices(size_X, size_Y)))
        assert_equal(card, card2)
        assert_equal(card, delannoy(size_X - 1, size_Y - 1))


  def test_mean_cost(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    scores = []
    for A in alignment_matrices(*C.shape):
      scores.append(np.vdot(A, C))
    val1 = np.mean(scores)
    val2 = mean_cost_C(C)
    val3 = mean_cost_value_and_grad_C(C)[0]
    val4 = mean_cost(X, Y)
    val5 = mean_cost_value_and_grad(X, Y)[0]
    assert_almost_equal(val1, val2)
    assert_almost_equal(val1, val3)
    assert_almost_equal(val1, val4)
    assert_almost_equal(val1, val5)


  def test_mean_cost_grad(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    A_sum = np.zeros_like(C)
    n = 0
    for A in alignment_matrices(*C.shape):
      A_sum += A
      n += 1
    A_mean1 = A_sum / n
    A_mean2 = mean_cost_value_and_grad_C(C)[1]

    assert_array_almost_equal(A_mean1, A_mean2)

    G = _num_gradient(mean_cost_C, A_sum)  # Any matrix can be used.
    assert_array_almost_equal(A_mean1, G)


  def test_squared_euclidean_cost(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C_XY = squared_euclidean_cost(X, Y)
    C = np.zeros_like(C_XY)

    for i in range(len(X)):
      for j in range(len(Y)):
        C[i, j] = 0.5 * np.sum((X[i] - Y[j]) ** 2)

    assert_array_almost_equal(C_XY, C)

    C_XX = squared_euclidean_cost(X, X)
    C_YY = squared_euclidean_cost(Y, Y)
    C_XY2, C_XX2, C_YY2 = squared_euclidean_cost(X, Y, return_all=True)
    assert_array_almost_equal(C_XY, C_XY2)
    assert_array_almost_equal(C_XX, C_XX2)
    assert_array_almost_equal(C_YY, C_YY2)


  def test_squared_euclidean_cost_vjp(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y)

    vjp = squared_euclidean_cost_vjp(X, Y, C)
    vjp2 = np.zeros_like(vjp)
    for i in range(len(X)):
      for j in range(len(Y)):
        for k in range(X.shape[1]):
          vjp2[i, k] += C[i, j] * (X[i, k] - Y[j, k])
    assert_array_almost_equal(vjp, vjp2)


  def test_squared_euclidean_cost_log_vjp(self):
    def f(X, Y):
      C = squared_euclidean_cost(X, Y, log=True)
      return sdtw_C(C)

    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    C = squared_euclidean_cost(X, Y, log=True)
    val, grad_C = sdtw_value_and_grad_C(C)
    grad_X = squared_euclidean_cost_vjp(X, Y, grad_C, log=True)
    grad_num = _num_gradient(functools.partial(f, Y=Y), X)
    assert_array_almost_equal(grad_X, grad_num)


  def test_squared_euclidean_cost_jvp(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)

    jvp = squared_euclidean_cost_jvp(X, Y, X)
    jvp2 = np.zeros_like(jvp)
    for i in range(len(X)):
      for j in range(len(Y)):
        for k in range(X.shape[1]):
          jvp2[i, j] += X[i, k] * (X[i, k] - Y[j, k])
    assert_array_almost_equal(jvp, jvp2)


  def test_squared_euclidean_distances(self):
    X, Y = _make_time_series(size_X=3, size_Y=3, num_dim=2)
    val = squared_euclidean_distance(X, Y)
    val2 = 0
    for i in range(len(X)):
      val2 += 0.5 * np.sum((X[i] - Y[i]) ** 2)
    assert_almost_equal(val, val2)

    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)
    assert_raises(ValueError, squared_euclidean_distance, X, Y)


  def test_divergences(self):
    X, Y = _make_time_series(size_X=3, size_Y=4, num_dim=2)

    for div, div_vg in ((sdtw_div, sdtw_div_value_and_grad),
                        (sharp_sdtw_div, sharp_sdtw_div_value_and_grad),
                        (mean_cost_div, mean_cost_div_value_and_grad)):

      for gamma in (0.1, 1.0):
        if str(div.__name__) == "mean_cost_div":
          value, G = div_vg(X, Y)
          value2 = div(X, Y)
          f = functools.partial(div, Y=Y)
          G_zero = div_vg(X, X)[1]
        else:
          value, G = div_vg(X, Y, gamma=gamma)
          value2 = div(X, Y, gamma=gamma)
          f = functools.partial(div, Y=Y, gamma=gamma)
          G_zero = div_vg(X, X, gamma=gamma)[1]

        G_num = _num_gradient(f, X)
        assert_almost_equal(value, value2)
        assert_array_almost_equal(G, G_num)
        assert_array_almost_equal(G_zero, np.zeros_like(G_zero))


  def test_barycenter(self):
    rng = np.random.RandomState(0)
    size_X, size_Y, num_dim = 3, 3, 2
    Ys = rng.randn(1, size_Y, num_dim)

    for value_and_grad in (sdtw_div_value_and_grad,
                           sharp_sdtw_div_value_and_grad,
                           mean_cost_div_value_and_grad):

      X = barycenter(Ys, Ys[0], value_and_grad)
      assert_array_almost_equal(X, Ys[0])


if __name__ == "__main__":
  absltest.main()
