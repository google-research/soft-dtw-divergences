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

"""Numpy + Numba implementation of the following paper.

Differentiable Divergences between Time Series
Mathieu Blondel, Arthur Mensch, Jean-Philippe Vert
https://arxiv.org/abs/2010.08354
"""

import functools
import numba
import numpy as np
from scipy.optimize import minimize


@numba.njit
def _soft_min_argmin(a, b, c):
  """Computes the soft min and argmin of (a, b, c).

  Args:
    a: scalar value.
    b: scalar value.
    c: scalar value.
  Returns:
    softmin, softargmin[0], softargmin[1], softargmin[2]
  """
  min_abc = min(a, min(b, c))
  exp_a = np.exp(min_abc - a)
  exp_b = np.exp(min_abc - b)
  exp_c = np.exp(min_abc - c)
  s = exp_a + exp_b + exp_c
  exp_a /= s
  exp_b /= s
  exp_c /= s
  val = min_abc - np.log(s)
  return val, exp_a, exp_b, exp_c


@numba.njit
def _sdtw_C(C, V, P):
  """SDTW dynamic programming recursion.

  Args:
    C: cost matrix (input).
    V: intermediate values (output).
    P: transition probability matrix (output).
  """
  size_X, size_Y = C.shape

  for i in range(1, size_X + 1):
    for j in range(1, size_Y + 1):

      smin, P[i, j, 0], P[i, j, 1], P[i, j, 2] = \
          _soft_min_argmin(V[i, j - 1], V[i - 1, j - 1], V[i - 1, j])

      # The cost matrix C is indexed starting from 0.
      V[i, j] = C[i - 1, j - 1] + smin


def sdtw_C(C, gamma=1.0, return_all=False):
  """Computes the soft-DTW value from a cost matrix C.

  Args:
    C: cost matrix, numpy array of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
    return_all: whether to return intermediate computations.
  Returns:
    sdtw_value if not return_all
    V (intermediate values), P (transition probability matrix) if return_all
  """
  size_X, size_Y = C.shape

  # Handle regularization parameter 'gamma'.
  if gamma != 1.0:
    C = C / gamma

  # Matrix containing the values of sdtw.
  V = np.zeros((size_X + 1, size_Y + 1))
  V[:, 0] = 1e10
  V[0, :] = 1e10
  V[0, 0] = 0

  # Tensor containing the probabilities of transition.
  P = np.zeros((size_X + 2, size_Y + 2, 3))

  _sdtw_C(C, V, P)

  if return_all:
    return gamma * V, P
  else:
    return gamma * V[size_X, size_Y]


def sdtw(X, Y, gamma=1.0, return_all=False):
  """Computes the soft-DTW value from time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
    return_all: whether to return intermediate computations.
  Returns:
    sdtw_value if not return_all
    V (intermediate values), P (transition probability matrix) if return_all
  """
  C = squared_euclidean_cost(X, Y)
  return sdtw_C(C, gamma=gamma, return_all=return_all)


@numba.njit
def _sdtw_grad_C(P, E):
  """Backward dynamic programming recursion.

  Args:
    P: transition probability matrix (input).
    E: expected alignment matrix (output).
  """
  # Equivalent to using reversed (not currently supported by Numba).
  #for j in reversed(range(1, E.shape[1] - 1)):
    #for i in reversed(range(1, E.shape[0] - 1)):
  for j in range(E.shape[1] - 2, 0, -1):
    for i in range(E.shape[0] - 2, 0, -1):

      E[i, j] = P[i, j + 1, 0] * E[i, j + 1] + \
                P[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                P[i + 1, j, 2] * E[i + 1, j]


def sdtw_grad_C(P, return_all=False):
  """Computes the soft-DTW gradient w.r.t. the cost matrix C.

  The gradient is equal to the expected alignment under the Gibbs distribution.

  Args:
    P: transition probability matrix.
    return_all: whether to return intermediate computations.
  Returns:
    E (expected alignment) if not return_all
    E with edges if return_all
  """
  E = np.zeros((P.shape[0], P.shape[1]))
  E[-1, :] = 0
  E[:, -1] = 0
  E[-1, -1] = 1
  P[-1, -1] = 1

  _sdtw_grad_C(P, E)

  if return_all:
    return E
  else:
    return E[1:-1, 1:-1]


def sdtw_value_and_grad_C(C, gamma=1.0):
  """Computes the soft-DTW value *and* gradient w.r.t. the cost matrix C.

  Args:
    C: cost matrix, numpy array of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
  Returns:
    sdtw_value, sdtw_gradient_C
  """
  size_X, size_Y = C.shape
  V, P = sdtw_C(C, gamma=gamma, return_all=True)
  return V[size_X, size_Y], sdtw_grad_C(P)


def sdtw_value_and_grad(X, Y, gamma=1.0):
  """Computes soft-DTW value *and* gradient w.r.t. time series X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    sdtw_value, sdtw_gradient_X
  """
  C = squared_euclidean_cost(X, Y)
  val, grad = sdtw_value_and_grad_C(C, gamma=gamma)
  return val, squared_euclidean_cost_vjp(X, Y, grad)


@numba.njit
def _sdtw_directional_derivative_C(P, Z, V_dot):
  """Recursion for computing the directional derivative in the direction of Z.

  Args:
    P: transition probability matrix (input).
    Z: direction matrix (input).
    V_dot: intermediate computations (output).
  """
  size_X, size_Y = Z.shape

  for i in range(1, size_X + 1):
    for j in range(1, size_Y + 1):
      # The matrix Z is indexed starting from 0.
      V_dot[i, j] = Z[i - 1, j - 1] + \
                    P[i, j, 0] * V_dot[i, j - 1] + \
                    P[i, j, 1] * V_dot[i - 1, j - 1] + \
                    P[i, j, 2] * V_dot[i - 1, j]


def sdtw_directional_derivative_C(P, Z, return_all=False):
  """Computes the soft-DTW directional derivative in the direction of Z.

  Args:
    P: transition probability matrix.
    Z: direction matrix.
    return_all: whether to return intermediate computations.
  Returns:
    sdtw_directional_derivative if not return_all
    V_dot (intermediate values) if return_all
  """
  size_X, size_Y = Z.shape

  if size_X != P.shape[0] - 2 or size_Y != P.shape[1] - 2:
    raise ValueError("Z should have shape " + str((P.shape[0], P.shape[1])))

  V_dot = np.zeros((size_X + 1, size_Y + 1))
  V_dot[0, 0] = 0

  _sdtw_directional_derivative_C(P, Z, V_dot)

  if return_all:
    return V_dot
  else:
    return V_dot[size_X, size_Y]


@numba.njit
def _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot):
  """Recursion for computing the Hessian product with Z.

  Args:
    P: transition probability matrix (input).
    P_dot: intermediate computations (output).
    E: output of sdtw_grad_C (input).
    E_dot: intermediate computations (output).
    V_dot: output of sdtw_directional_derivative_C (input).
  """
  # Equivalent to using reversed (not currently supported by Numba).
  #for j in reversed(range(1, V_dot.shape[1])):
    #for i in reversed(range(1, V_dot.shape[0])):
  for j in range(V_dot.shape[1] - 1, 0, -1):
    for i in range(V_dot.shape[0] - 1, 0, -1):

      inner = P[i, j, 0] * V_dot[i, j - 1]
      inner += P[i, j, 1] * V_dot[i - 1, j - 1]
      inner += P[i, j, 2] * V_dot[i - 1, j]

      P_dot[i, j, 0] = P[i, j, 0] * inner
      P_dot[i, j, 1] = P[i, j, 1] * inner
      P_dot[i, j, 2] = P[i, j, 2] * inner

      P_dot[i, j, 0] -= P[i, j, 0] * V_dot[i, j - 1]
      P_dot[i, j, 1] -= P[i, j, 1] * V_dot[i - 1, j - 1]
      P_dot[i, j, 2] -= P[i, j, 2] * V_dot[i - 1, j]

      E_dot[i, j] = P_dot[i, j + 1, 0] * E[i, j + 1] + \
                    P[i, j + 1, 0] * E_dot[i, j + 1] + \
                    P_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                    P[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                    P_dot[i + 1, j, 2] * E[i + 1, j] + \
                    P[i + 1, j, 2] * E_dot[i + 1, j]


def sdtw_hessian_product_C(P, E, V_dot):
  """Computes the soft-DTW Hessian product.

  Args:
    P: transition probability matrix.
    E: expected alignment matrix (output of sdtw_grad_C).
    V_dot: output of sdtw_directional_derivative_C.
  Returns:
    sdtw_Hessian_product
  """
  E_dot = np.zeros_like(E)
  P_dot = np.zeros((E.shape[0], E.shape[1], 3))

  if P.shape[0] != E.shape[0] or P.shape[1] != E.shape[1]:
    raise ValueError("P and E have incompatible shapes.")

  if P.shape[0] - 1 != V_dot.shape[0] or P.shape[1] - 1 != V_dot.shape[1]:
    raise ValueError("P and V_dot have incompatible shapes.")

  _sdtw_hessian_product_C(P, P_dot, E, E_dot, V_dot)

  return E_dot[1:-1, 1:-1]


def sdtw_entropy_C(C, gamma=1.0):
  """Computes the entropy of the Gibbs distribution associated with soft-DTW.

  Args:
    C: cost matrix, numpy array of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
  Returns:
    entropy_value
  """
  val, E = sdtw_value_and_grad_C(C, gamma=gamma)
  return (np.vdot(E, C) - val) / gamma


def sdtw_entropy(X, Y, gamma=1.0):
  """Computes the entropy of the Gibbs distribution associated with soft-DTW.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    entropy_value
  """
  C = squared_euclidean_cost(X, Y)
  return sdtw_entropy_C(C, gamma=gamma)


def sharp_sdtw_C(C, gamma=1.0):
  """Computes the sharp soft-DTW value from a cost matrix C.

  Args:
    C: cost matrix, numpy array of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
  Returns:
    sharp_sdtw_value
  """
  P = sdtw_C(C, gamma=gamma, return_all=True)[1]
  return sdtw_directional_derivative_C(P, C)


def sharp_sdtw(X, Y, gamma=1.0):
  """Computes the sharp soft-DTW value from time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    sharp_sdtw_value
  """
  C = squared_euclidean_cost(X, Y)
  return sharp_sdtw_C(C, gamma=gamma)


def sharp_sdtw_value_and_grad_C(C, gamma=1.0):
  """Computes the sharp soft-DTW value *and* its gradient w.r.t. C.

  Args:
    C: cost matrix, numpy array of shape (size_X, size_Y).
    gamma: regularization strength (scalar value).
  Returns:
    sharp_sdtw_value, sharp_sdtw_grad_C
  """
  V, P = sdtw_C(C, gamma=gamma, return_all=True)
  E = sdtw_grad_C(P, return_all=True)
  V_dot = sdtw_directional_derivative_C(P, C, return_all=True)
  HC = sdtw_hessian_product_C(P, E, V_dot)
  G = E[1:-1, 1:-1]
  val = V_dot[-1, -1]
  grad = G + HC / gamma
  return val, grad


def sharp_sdtw_value_and_grad(X, Y, gamma=1.0):
  """Computes the sharp soft-DTW value *and* its gradient w.r.t. X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    sharp_sdtw_value, sharp_sdtw_grad_X
  """
  C = squared_euclidean_cost(X, Y)
  val, grad = sharp_sdtw_value_and_grad_C(C, gamma=gamma)
  return val, squared_euclidean_cost_vjp(X, Y, grad)


@numba.njit
def _cardinality(V, P):
  """Recursion for computing the cardinality of the set of alignments."""
  for i in range(1, V.shape[0]):
    for j in range(1, V.shape[1]):
      V[i,j] = V[i, j-1] + V[i-1, j-1] + V[i-1, j]
      P[i, j, 0] = V[i, j - 1] / V[i, j]
      P[i, j, 1] = V[i - 1, j - 1] / V[i, j]
      P[i, j, 2] = V[i - 1, j] / V[i, j]


def cardinality(size_X, size_Y, return_all=False):
  """Computes the cardinality of the set of alignments.

  Args:
    size_X: size of the time series X.
    size_Y: size of the time series Y.
    return_all: whether to return intermediate computations.
  Returns:
    cardinality if not return_all
    V (intermediate values), P (transition probability matrix) if return_all
  """

  # Matrix containing the cardinalities.
  V = np.zeros((size_X + 1, size_Y + 1))
  V[0, 0] = 1

  # Tensor containing the probabilities of transition.
  P = np.zeros((size_X + 2, size_Y + 2, 3))

  _cardinality(V, P)

  if return_all:
    return V, P
  else:
    return V[size_X, size_Y]


def mean_alignment(size_X, size_Y):
  """Computes the mean of all possible alignments.

  Args:
    size_X: size of the time series X.
    size_Y: size of the time series Y.
  Returns:
    mean_alignment
  """
  P = cardinality(size_X, size_Y, return_all=True)[1]
  return sdtw_grad_C(P)


def mean_cost_C(C):
  """Computes the mean cost from a cost matrix C.

  Args:
    C: cost matrix of shape (size_X, size_Y).
  Returns:
    mean_cost_value
  """
  P = cardinality(C.shape[0], C.shape[1], return_all=True)[1]
  return sdtw_directional_derivative_C(P, C)


def mean_cost(X, Y):
  """Computes the mean cost from time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
  Returns:
    mean_cost_value
  """
  C = squared_euclidean_cost(X, Y)
  return mean_cost_C(C)


def mean_cost_value_and_grad_C(C):
  """Computes the mean cost value *and* gradient w.r.t. the cost matrix C.

  Args:
    C: cost matrix of shape (size_X, size_Y).
  Returns:
    mean_cost_value, mean_cost_grad_C
  """
  P = cardinality(C.shape[0], C.shape[1], return_all=True)[1]
  val = sdtw_directional_derivative_C(P, C)
  G = sdtw_grad_C(P)
  return val, G


def mean_cost_value_and_grad(X, Y):
  """Computes the mean cost value *and* gradient w.r.t. X.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
  Returns:
    mean_cost_value, mean_cost_grad_X
  """
  C = squared_euclidean_cost(X, Y)
  val, grad = mean_cost_value_and_grad_C(C)
  return val, squared_euclidean_cost_vjp(X, Y, grad)

def squared_euclidean_cost(X, Y, return_all=False, log=False):
  """Computes the squared Euclidean cost.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    return_all: whether to also return the cost matrices for (X, X) and (Y, Y).
    log: whether to use the log-augmented cost or not (see paper).
  Returns:
    C(X, Y) if not return_all
    C(X, Y), C(X, X), C(Y, Y) if return_all
  """
  def _C(C):
    if log:
      return C + np.log(2 - np.exp(-C))
    else:
      return C

  X_sqnorms = 0.5 * np.sum(X ** 2, axis=1)
  Y_sqnorms = 0.5 * np.sum(Y ** 2, axis=1)
  XY = np.dot(X, Y.T).astype(X_sqnorms.dtype)

  if return_all:
    C_XY = -XY
    C_XY += X_sqnorms[:, np.newaxis]
    C_XY += Y_sqnorms

    C_XX = -np.dot(X, X.T)
    C_XX += X_sqnorms[:, np.newaxis]
    C_XX += X_sqnorms

    C_YY = -np.dot(Y, Y.T)
    C_YY += Y_sqnorms[:, np.newaxis]
    C_YY += Y_sqnorms

    return _C(C_XY), _C(C_XX), _C(C_YY)

  else:
    C = -XY
    C += X_sqnorms[:, np.newaxis]
    C += Y_sqnorms
    return _C(C)


def squared_euclidean_cost_vjp(X, Y, E, log=False):
  """Left-product with the Jacobian of the squared Euclidean cost.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    E: matrix to multiply with, numpy array of shape (size_X, size_Y).
    log: whether to use the log-augmented cost or not (see paper).
  Returns:
    vjp
  """
  if E.shape[0] != len(X) or E.shape[1] != len(Y):
    raise ValueError("E.shape should be equal to (len(X), len(Y)).")

  e = E.sum(axis=1)
  vjp = X * e[:, np.newaxis]
  vjp -= np.dot(E, Y)

  if log:
    C = squared_euclidean_cost(X, Y)
    deriv = np.exp(-C) / (2 - np.exp(-C))
    vjp += squared_euclidean_cost_vjp(X, Y, E * deriv)

  return vjp


def squared_euclidean_cost_jvp(X, Y, Z):
  """Right-product with the Jacobian of the squared Euclidean cost.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    Z: matrix to multiply with, numpy array of shape (size_X, num_dim).
  Returns:
    jvp
  """
  if Z.shape[0] != X.shape[0] or Z.shape[1] != X.shape[1]:
    raise ValueError("Z should be of the same shape as X.")

  if Y.shape[1] != Z.shape[1]:
    raise ValueError("Y.shape[1] should be equal to Z.shape[1].")

  jvp = -np.dot(Z, Y.T)
  jvp += np.sum(X * Z, axis=1)[:, np.newaxis]
  return jvp


def squared_euclidean_distance(X, Y):
  """Computes the squared Euclidean distance between two time series.

  The two time series must have the same length.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
  Returns:
    distance_value
  """
  if len(X) != len(Y) or X.shape[1] != Y.shape[1]:
    raise ValueError("X and Y have incompatible shapes.")

  return 0.5 * np.sum((X - Y) ** 2)


def _divergence(func, X, Y):
  """Converts a value function into a divergence.

  The cost is assumed to be the squared Euclidean one.

  Args:
    func: function to use.
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
  Returns:
    func(C(X,Y)) - 0.5 * func(C(X,X)) - 0.5 * func(C(Y,Y))
  """
  C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)
  value = func(C_XY)
  value -= 0.5 * func(C_XX)
  value -= 0.5 * func(C_YY)
  return value


def _divergence_value_and_grad(func, X, Y):
  """Converts a value and grad function into a divergence.

  The cost is assumed to be the squared Euclidean one.

  Args:
    func: function to use.
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
  Returns:
    div_value, div_grad_X
  """
  C_XY, C_XX, C_YY = squared_euclidean_cost(X, Y, return_all=True)
  value_XY, grad_XY = func(C_XY)
  value_XX, grad_XX = func(C_XX)
  value_YY, grad_YY = func(C_YY)
  value = value_XY - 0.5 * value_XX - 0.5 * value_YY
  grad = squared_euclidean_cost_vjp(X, Y, grad_XY)
  # The 0.5 factor cancels out.
  grad -= squared_euclidean_cost_vjp(X, X, grad_XX)
  return value, grad


def sdtw_div(X, Y, gamma=1.0):
  """Compute the soft-DTW divergence value between time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value
  """
  func = functools.partial(sdtw_C, gamma=gamma)
  return _divergence(func, X, Y)


def sdtw_div_value_and_grad(X, Y, gamma=1.0):
  """Compute the soft-DTW divergence value *and* gradient w.r.t. X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value, divergence_grad
  """
  func = functools.partial(sdtw_value_and_grad_C, gamma=gamma)
  return _divergence_value_and_grad(func, X, Y)


def sharp_sdtw_div(X, Y, gamma=1.0):
  """Compute the sharp soft-DTW divergence value between time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value
  """
  func = functools.partial(sharp_sdtw_C, gamma=gamma)
  return _divergence(func, X, Y)


def sharp_sdtw_div_value_and_grad(X, Y, gamma=1.0):
  """Compute the sharp soft-DTW divergence value *and* gradient w.r.t. X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value, divergence_grad
  """
  func = functools.partial(sharp_sdtw_value_and_grad_C, gamma=gamma)
  return _divergence_value_and_grad(func, X, Y)


def mean_cost_div(X, Y):
  """Compute the mean-cost divergence value between time series X and Y.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value
  """
  return _divergence(mean_cost_C, X, Y)


def mean_cost_div_value_and_grad(X, Y):
  """Compute the mean-cost divergence value *and* gradient w.r.t. X.

  The cost is assumed to be the squared Euclidean one.

  Args:
    X: time series, numpy array of shape (size_X, num_dim).
    Y: time series, numpy array of shape (size_Y, num_dim).
    gamma: regularization strength (scalar value).
  Returns:
    divergence_value, divergence_grad
  """
  return _divergence_value_and_grad(mean_cost_value_and_grad_C, X, Y)


def euclidean_mean(Ys, weights=None):
  """Compute the Euclidean of a list of time series.

  Args:
    Ys: a list of time series, i.e., a list of numpy arrays.
    weights: a list of weights (ones by default).
  Returns:
    mean(Ys)
  """
  if weights is None:
    weights = np.ones(len(Ys))

  X = None
  weight_sum = 0

  for i, Y in enumerate(Ys):
    if X is None:
      X = weights[i] * Y.copy()
    else:
      X += weights[i] * Y
    weight_sum += weights[i]

  X /= weight_sum

  return X


def barycenter(Ys, X_init, value_and_grad=sdtw_div_value_and_grad, weights=None,
               method="L-BFGS-B", tol=1e-3, max_iter=200):
  """Computes the barycenter of a list of time series.

  Args:
    Ys: a list of time series, i.e., a list of numpy arrays.
    X_init: initialization.
    value_and_grad: function returning a value and a gradient.
    weights: a list of weights (ones by default).
    method: method to be used.
    tol: tolerance for the stopping criterion.
    max_iter: max number of iterations.
  Returns:
    barycenter(Ys)
  """

  if weights is None:
    weights = np.ones(len(Ys))

  weights = np.array(weights)

  if len(weights) != len(Ys):
    raise ValueError("Ys and weights should have the same length.")

  if isinstance(X_init, str) and X_init == "euclidean_mean":
    X_init = euclidean_mean(Ys, weights)

  elif isinstance(X_init, str) and X_init == "sdtw":
    X_init = barycenter(Ys, X_init="euclidean_mean",
                        value_and_grad=sdtw_value_and_grad, weights=weights)

  elif isinstance(X_init, str) and X_init == "mean_cost":
    X_init = barycenter(Ys, X_init="euclidean_mean",
                        value_and_grad=mean_cost_value_and_grad, weights=weights)

  def _func(X_flat):
    X = X_flat.reshape(*X_init.shape)
    G = np.zeros_like(X_init)
    obj_value = 0

    for i in range(len(Ys)):
      value, grad = value_and_grad(X, Ys[i])
      G += weights[i] * grad
      obj_value += weights[i] * value

    # 'minimize' cannot handle matrices so we need to flatten the gradient.
    return obj_value, G.ravel()

  res = minimize(_func, X_init.ravel(), method=method, jac=True,
                 tol=tol, options=dict(maxiter=max_iter, disp=False))

  return res.x.reshape(*X_init.shape)


def _alignment_matrices(size_X, size_Y, start=None, M=None):
  """Helper function"""
  if start is None:
    start = [0, 0]
    M = np.zeros((size_X, size_Y))

  i, j = start
  M[i, j] = 1
  ret = []

  if i == size_X - 1 and j == size_Y - 1:
    yield M
  else:
    if i < size_X - 1:
      yield from _alignment_matrices(size_X, size_Y, (i+1, j), M.copy())
    if i < size_X - 1 and j < size_Y - 1:
      yield from _alignment_matrices(size_X, size_Y, (i+1, j+1), M.copy())
    if j < size_Y - 1:
      yield from _alignment_matrices(size_X, size_Y, (i, j+1), M.copy())


def alignment_matrices(size_X, size_Y):
  """Generator of all alignment matrices of shape (size_X, size_Y)."""
  yield from _alignment_matrices(size_X, size_Y)
