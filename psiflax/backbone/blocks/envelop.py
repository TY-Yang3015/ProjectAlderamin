import flax.linen as nn
import jax.numpy as jnp

from einops import rearrange, reduce, repeat
from jax import random
import jax


def custom_initializer(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype) * 0.1 + 10.0


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
            (self.num_of_determinants, self.num_of_electrons, 1, self.num_of_nucleus),
        )
        self.sigma_kiI = self.param(
            "omega_kiI",
            nn.initializers.ones,
            (self.num_of_determinants, self.num_of_electrons, 1, self.num_of_nucleus),
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

        psiformer_pre_det = psiformer_pre_det.reshape(
            psiformer_pre_det.shape[0],
            self.num_of_electrons,
            self.num_of_electrons,
            self.num_of_determinants,
        )
        psiformer_pre_det = rearrange(psiformer_pre_det, "b i j k -> b k i j")

        elec_nuc_features = rearrange(elec_nuc_features, f"b j I 1 -> b 1 j I ")
        elec_nuc_features = repeat(
            elec_nuc_features, f"b 1 j I -> b {self.num_of_determinants} 1 j I"
        )  # b k 1 j I
        exponent = self.sigma_kiI * elec_nuc_features  # k i 1 I, b k 1 j I -> b k i j I

        # k i 1 I, b k i j I -> b k i j
        matrix_element_omega = (self.pi_kiI * jnp.exp(-exponent)).sum(axis=-1)

        # b k i j -> b k i j
        determinants = psiformer_pre_det * matrix_element_omega

        # b k i j -> b 1
        wavefunction = jnp.linalg.det(determinants).sum(axis=-1).reshape(-1, 1)

        return wavefunction


# import jax

# print(Envelop(6, 5, 3).tabulate(jax.random.PRNGKey(0),
#                                jnp.ones((4096, 5, 3, 1)), jnp.ones((4096, 5, 30)),
#                                depth=1, console_kwargs={'width': 150}))
