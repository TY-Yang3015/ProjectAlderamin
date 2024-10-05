import flax.linen as nn
import jax.numpy as jnp

from einops import rearrange, reduce, repeat
from jax import random
import jax

from psiflax.utils import signed_log_sum_exp


def custom_initializer(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype) * 0.1 + 1.


class Envelop(nn.Module):
    """
    the exponential-decaying envelop to satisfy the vanishing condition of the wavefunction
    at infinity, with two sets of learnable parameters.

    :cvar num_of_determinants: the number of determinants for psiformer before multiplying to the
                               jastrow factor.
    :cvar num_of_electrons: the number of electrons in the system.
    :cvar num_of_nucleus: the number of nucleus in the system.

    :cvar computation_dtype: the dtype of the computation.
    :cvar param_dtype: the dtype of the parameters. this is a dummy input for this class.
    """

    num_of_determinants: int
    num_of_electrons: int
    num_of_nucleus: int

    computation_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.pi_kiI = self.param(
            "pi_kiI",
            nn.initializers.ones,
            (1, self.num_of_nucleus, self.num_of_determinants * self.num_of_electrons),
        )
        self.sigma_kiI = self.param(
            "omega_kiI",
            nn.initializers.ones,
            (1, self.num_of_nucleus, self.num_of_determinants * self.num_of_electrons),
        )

    def __call__(
            self, elec_nuc_features: jnp.ndarray, psiformer_pre_det: jnp.ndarray
    ) -> jnp.ndarray:
        """
        :param elec_nuc_features: jnp.ndarray contains electron-nuclear distance with dimension (batch,
                                  num_of_electrons, num_of_nucleus, 1).
        :param psiformer_pre_det: psiformer electron-nuclear channel output with dimension
                                  (batch, num_of_electrons, number_of_determinants x number_of_electrons).
        :return: weighted sum of all determinants with envelop multiplied with shape (batch, 1)
        """

        assert psiformer_pre_det.ndim == 3, "psiformer_pre_det must be 3d tensor."

        #psiformer_pre_det = psiformer_pre_det.reshape(
        #    psiformer_pre_det.shape[0],
        #    self.num_of_electrons,
        #    -1,
        #    self.num_of_determinants,
        #)
        #psiformer_pre_det = rearrange(psiformer_pre_det, "b i j k -> b k i j")

        # k i 1 I, b k i j I -> b k i j
        matrix_element_omega = (self.pi_kiI * jnp.exp(-elec_nuc_features * self.sigma_kiI)).sum(axis=2)

        # b k i j, b k i j -> b k i j
        determinants = psiformer_pre_det * matrix_element_omega

        determinants = jnp.reshape(determinants, (determinants.shape[0],
                                                  determinants.shape[1], -1,
                                                  self.num_of_determinants))
        determinants = rearrange(determinants, "b i j k -> b k j i")
        return determinants


"""
import jax

print(Envelop(num_of_determinants=16,
              num_of_electrons=5,
              num_of_nucleus=2).tabulate(jax.random.PRNGKey(0),
                                jnp.ones((256, 3, 2, 1)), jnp.ones((256, 3, 80)),
                                depth=1, console_kwargs={'width': 150}))
#"""
