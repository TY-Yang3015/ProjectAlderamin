import flax.linen as nn
import jax.numpy as jnp

from einops import rearrange


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

    computation_dtype: jnp.dtype = jnp.float16
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.pi_kiI = self.param(
            "pi_kiI",
            nn.initializers.ones,
            (self.num_of_determinants, self.num_of_electrons, self.num_of_nucleus),
        )
        self.sigma_kiI = self.param(
            "omega_kiI",
            nn.initializers.normal(stddev=0.01),
            (self.num_of_determinants, self.num_of_electrons, self.num_of_nucleus),
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

        psiformer_pre_det = psiformer_pre_det.reshape(psiformer_pre_det.shape[0],
                                                      self.num_of_electrons, self.num_of_electrons,
                                                      self.num_of_determinants)
        psiformer_pre_det = rearrange(psiformer_pre_det, "b i j k -> b k i j")

        exponent = jnp.einsum(
            "kiI,bjI1->kijI",
            self.sigma_kiI,
            elec_nuc_features,
            preferred_element_type=self.computation_dtype,
        )

        matrix_element_omega = jnp.einsum(
            "kiI,kijI->kij",
            self.pi_kiI,
            jnp.exp(-exponent),
            preferred_element_type=self.computation_dtype,
        )

        determinants = jnp.einsum(
            "bkij,kij->bkij",
            psiformer_pre_det,
            matrix_element_omega,
            preferred_element_type=self.computation_dtype,
        )

        determinants = jnp.clip(determinants, -10000, 10000)

        sign, logabsdet = jnp.linalg.slogdet(determinants)

        log_abs_wavefunction = jnp.log(
            jnp.abs(
                jnp.expand_dims((jnp.exp(logabsdet) * sign).sum(axis=-1), -1)
            )
        )

        return log_abs_wavefunction


# import jax

# print(Envelop(6, 5, 3).tabulate(jax.random.PRNGKey(0),
#                                jnp.ones((4096, 5, 3, 1)), jnp.ones((4096, 5, 30)),
#                                depth=1, console_kwargs={'width': 150}))
