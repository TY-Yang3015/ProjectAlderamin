import jax.numpy as jnp
import functools


def signed_log_sum_exp(signs, log_dets):
    # Find the maximum log determinant for numerical stability
    max_log_det = jnp.max(log_dets)

    # Compute the signed exponential sum
    signed_exps = signs * jnp.exp(
        log_dets - max_log_det
    )  # Incorporate the signs directly
    signed_sum = jnp.sum(signed_exps)

    # Get the sign of the final result
    result_sign = jnp.sign(signed_sum)

    # Compute the log of the absolute value of the final sum
    log_abs_sum = max_log_det + jnp.log(jnp.abs(signed_sum))

    # Return the log-sum-exp result with the correct sign
    return log_abs_sum
