import flax.linen as nn
import jax.numpy as jnp


class Envelop(nn.Module):
    """
    the exponential-decaying envelop to satisfy the vanishing condition of the wavefunction
    at infinity, with two sets of learnable parameters. The weight matrix is included in this
    class as well.

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
        self.pi_kiI = self.param("pi_kiI", nn.initializers.ones, (self.num_of_determinants
                                                                  , self.num_of_electrons, self.num_of_nucleus))
        self.sigma_kiI = self.param("omega_kiI", nn.initializers.ones, (self.num_of_determinants,
                                                                        self.num_of_electrons, self.num_of_nucleus))

        self.weights = self.param("weights", nn.initializers.ones, (self.num_of_determinants,
                                                                    self.num_of_determinants * self.num_of_electrons,
                                                                    self.num_of_electrons))

    def __call__(self, elec_nuc_features: jnp.ndarray, psiformer_pre_det: jnp.ndarray) -> jnp.ndarray:
        """
        :param elec_nuc_features: jnp.ndarray contains electron-nuclear distance with dimension (batch,
                                  num_of_electrons, num_of_nucleus, 1).
        :param psiformer_pre_det: psiformer electron-nuclear channel output with dimension
                                  (batch, num_of_electrons, number_of_determinants x number_of_electrons).
        :return: weighted sum of all determinants with envelop multiplied with shape (batch, 1)
        """

        assert psiformer_pre_det.ndim == 3, "psiformer_pre_det must be 3d tensor."

        weighted_feature = jnp.einsum('bip,kpj->bkij', psiformer_pre_det, self.weights,
                                      preferred_element_type=self.computation_dtype)

        exponent = -jnp.einsum('kiI,bjI1->bkijI', self.sigma_kiI, elec_nuc_features,
                               preferred_element_type=self.computation_dtype)
        matrix_element_omega = jnp.einsum("kiI,bkijI->bkij", self.pi_kiI, jnp.exp(exponent),
                                          preferred_element_type=self.computation_dtype)

        determinants = jnp.einsum('bkij,bkij->bk', weighted_feature, matrix_element_omega,
                                  preferred_element_type=self.computation_dtype)

        return jnp.expand_dims(determinants.sum(axis=-1), -1)

# import jax

# print(Envelop(6, 5, 3).tabulate(jax.random.PRNGKey(0),
#                                jnp.ones((4096, 5, 3, 1)), jnp.ones((4096, 5, 30)),
#                                depth=1, console_kwargs={'width': 150}))
