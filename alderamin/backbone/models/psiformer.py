import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange

from alderamin.backbone.blocks import PsiFormerBlock, SimpleJastrow, Envelop


class PsiFormer(nn.Module):
    """
    full implementation of PsiFormer, consists of three main pieces: jastrow factor, decaying
    envelop and PsiFormer blocks. see docs for each component for details.

    :cvar num_of_determinants: the number of determinants for psiformer before multiplying to the
                               jastrow factor.
    :cvar num_of_electrons: the number of electrons in the system.
    :cvar num_of_nucleus: the number of nucleus in the system.

    :cvar num_of_blocks: the number of PsiFormer blocks.
    :cvar num_heads: The number of heads in the multi-head attention block.
    :cvar use_memory_efficient_attention: whether to use memory efficient attention. see the
                                        doc for MultiHeadCrossAttention layer for more details.
    :cvar group: set to None for LayerNorm, otherwise GroupNorm will be used. default: None.

    :cvar computation_dtype: the dtype of the computation.
    :cvar param_dtype: the dtype of the parameters.
    """

    num_of_determinants: int
    num_of_electrons: int
    num_of_nucleus: int

    num_of_blocks: int
    num_heads: int
    qkv_size: int
    use_memory_efficient_attention: bool = False
    group: None | int = None

    computation_dtype: jnp.dtype | str = "float32"
    param_dtype: jnp.dtype | str = "float32"

    @nn.compact
    def __call__(
            self,
            electron_nuclear_features: jnp.ndarray,
            single_electron_features: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        :param electron_nuclear_features: the electronic nuclear features tensor, should have the shape
                                           (batch, num_of_electrons, num_of_nucleus, 5)
        :param single_electron_features: the single electron features tensor, should have the shape
                                           (batch, num_of_electrons, 4)
        :return: wavefunction values with shape (batch, 1)
        """

        if electron_nuclear_features.ndim == 3:
            electron_nuclear_features = jnp.expand_dims(electron_nuclear_features, 0)

        x = rearrange(electron_nuclear_features, "b n c f -> b n (c f)")
        x = nn.Dense(
            features=self.num_heads * self.qkv_size,
            use_bias=False,
            dtype=self.computation_dtype,
            param_dtype=self.param_dtype,
        )(x)

        for _ in range(self.num_of_blocks):
            x = PsiFormerBlock(
                num_heads=self.num_heads,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                group=self.group,
                param_dtype=self.param_dtype,
                computation_dtype=self.computation_dtype,
            )(x)

        psiformer_pre_det = nn.Dense(
            features=self.num_of_electrons * self.num_of_determinants,
            use_bias=False,
            dtype=self.computation_dtype,
            param_dtype=self.param_dtype,
        )(x)

        log_abs_wavefunction = Envelop(
            num_of_determinants=self.num_of_determinants,
            num_of_electrons=self.num_of_electrons,
            num_of_nucleus=self.num_of_nucleus,
            param_dtype=self.param_dtype,
            computation_dtype=self.computation_dtype,
        )(jnp.expand_dims(electron_nuclear_features[..., 3], -1), psiformer_pre_det)

        jastrow_factor = SimpleJastrow()(single_electron_features)

        # return jnp.log(jnp.abs(jnp.exp(log_abs_wavefunction) * jnp.exp(jastrow_factor)) + 1e-12)
        return jastrow_factor + log_abs_wavefunction

# import jax

# print(PsiFormer(num_of_determinants=6,
#                num_of_electrons=2,
#                num_of_nucleus=2,
#                num_of_blocks=5,
#                num_heads=8).tabulate(jax.random.PRNGKey(0),
#                                       jnp.ones((512, 2, 2, 5)), jnp.ones((512, 2, 4)),
#                                       depth=1, console_kwargs={'width': 150}))
