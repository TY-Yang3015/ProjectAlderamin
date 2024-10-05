import jax.numpy as jnp
import functools


def signed_log_sum_exp(signs, log_dets):
    # find the maximum log determinant for numerical stability
    max_log_det = jnp.max(log_dets)

    # compute the signed exponential sum
    signed_exps = signs * jnp.exp(
        log_dets - max_log_det
    )  # Incorporate the signs directly
    signed_sum = jnp.sum(signed_exps)

    # compute the log of the absolute value of the final sum
    log_abs_sum = max_log_det + jnp.log(jnp.abs(signed_sum))

    # return the log-sum-exp result with the correct sign
    return log_abs_sum
