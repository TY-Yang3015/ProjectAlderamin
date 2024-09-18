from typing import Optional

import jax.numpy as jnp
import functools


def slogdet(x):
  """Computes sign and log of determinants of matrices.

  This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

  Args:
    x: square matrix.

  Returns:
    sign, (natural) logarithm of the determinant of x.
  """
  if x.shape[-1] == 1:
    if x.dtype == jnp.complex64 or x.dtype == jnp.complex128:
      sign = x[..., 0, 0] / jnp.abs(x[..., 0, 0])
    else:
      sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet


def logdet_matmul(
    xs: jnp.ndarray, w: Optional[jnp.ndarray] = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Combines determinants and takes dot product with weights in log-domain.

  We use the log-sum-exp trick to reduce numerical instabilities.

  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.

  Returns:
    sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
    determinant (or product of the i-th determinant in each spin channel, if
    full_det is not used).
  """
  # 1x1 determinants appear to be numerically sensitive and can become 0
  # (especially when multiple determinants are used with the spin-factored
  # wavefunction). Avoid this by not going into the log domain for 1x1 matrices.
  # Pass initial value to functools so det1d = 1 if all matrices are larger than
  # 1x1.
  det1d = functools.reduce(lambda a, b: a * b,
                           [x.reshape(-1) for x in xs if x.shape[-1] == 1], 1)
  # Pass initial value to functools so sign_in = 1, logdet = 0 if all matrices
  # are 1x1.
  phase_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [slogdet(x) for x in xs if x.shape[-1] > 1], (1, 0))

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = phase_in * det1d * jnp.exp(logdet - maxlogdet)
  if w is None:
    result = jnp.sum(det)
  else:
    result = jnp.matmul(det, w)[0]
  # return phase as a unit-norm complex number, rather than as an angle
  if result.dtype == jnp.complex64 or result.dtype == jnp.complex128:
    phase_out = jnp.angle(result)  # result / jnp.abs(result)
  else:
    phase_out = jnp.sign(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return phase_out, log_out