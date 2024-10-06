import flax.linen as nn
import jax.numpy as jnp

from einops import rearrange
from functools import partial

class Envelop(nn.Module):
    """
    the exponential-decaying envelop to satisfy the vanishing condition of the wavefunction
    at infinity, with two sets of learnable parameters.

    :cvar num_of_determinants: the number of determinants for psiformer before multiplying to the
                               jastrow factor.
    :cvar num_of_electrons: the number of electrons in the system.
    :cvar num_of_nucleus: the number of nucleus in the system.

    :cvar param_dtype: the dtype of the parameters.
    """

    num_of_determinants: int
    num_of_electrons: int
    num_of_nucleus: int

    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.pi_kiI = self.param(
            "pi_kiI",
            nn.initializers.ones,
            (1, self.num_of_nucleus, self.num_of_determinants * self.num_of_electrons),
            self.param_dtype
        )
        self.sigma_kiI = self.param(
            "omega_kiI",
            nn.initializers.ones,
            (1, self.num_of_nucleus, self.num_of_determinants * self.num_of_electrons),
            self.param_dtype
        )

    def __call__(
            self, elec_nuc_features: jnp.ndarray, psiformer_pre_det: jnp.ndarray
    ) -> jnp.ndarray:
        """
        :param elec_nuc_features: jnp.ndarray contains electron-nuclear distance with dimension (batch,
                                  num_of_spin_electrons, num_of_nucleus, 1).
        :param psiformer_pre_det: psiformer electron-nuclear channel output with dimension
                                  (batch, num_of_spin_electrons, number_of_determinants x number_of_electrons).
        :return: weighted sum of all determinants with envelop multiplied with shape (batch, 1)
        """

        # 1 I k*N, (b s I 1 * 1 I k*N) -> b s I k*N -> b s k*N
        matrix_element_omega = (
                self.pi_kiI * jnp.exp(-elec_nuc_features * self.sigma_kiI)
        ).sum(axis=2)

        # b s k*N, b s k*N -> b s k*N
        determinants = psiformer_pre_det * matrix_element_omega

        # b s k*N -> b s N k
        determinants = jnp.reshape(
            determinants,
            (
                determinants.shape[0],
                determinants.shape[1],
                -1,
                self.num_of_determinants,
            ),
        )

        # b s N k -> b k N s
        determinants = rearrange(determinants, "b s N k -> b k N s")

        return determinants


"""
import jax

print(Envelop(num_of_determinants=16,
              num_of_electrons=5,
              num_of_nucleus=2).tabulate(jax.random.PRNGKey(0),
                                jnp.ones((256, 3, 2, 1)), jnp.ones((256, 3, 80)),
                                depth=1, console_kwargs={'width': 150}))
#"""
