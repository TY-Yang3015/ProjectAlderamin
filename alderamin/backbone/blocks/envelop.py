import flax.linen as nn
import jax.numpy as jnp


class Envelop(nn.Module):
    num_of_determinants: int
    num_of_electrons: int
    num_of_nucleus: int

    computation_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.pi_Ijk = self.param("pi_kiI", nn.initializers.ones, (self.num_of_determinants
                                                                  , self.num_of_electrons, self.num_of_nucleus))
        self.omega_ijk = self.param("omega_kiI", nn.initializers.ones, (self.num_of_determinants,
                                                                        self.num_of_electrons, self.num_of_nucleus))

    def __call__(self, elec_nuc_features: jnp.ndarray, psiformer_pre_det: jnp.ndarray) -> jnp.ndarray:
        """
        :param elec_nuc_features: jnp.ndarray contains electron-nuclear distance with dimension (batch, 1).
        :param psiformer_pre_det: psiformer electron-nuclear channel output with dimension
                                  (batch, num_of_electrons, number_of_determinants x number_of_electrons).
        :return: weighted sum of features with dimension (batch, num_of_det,
                                  number_of_electrons, number_of_electrons).
        """


