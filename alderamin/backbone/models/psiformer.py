import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange

from alderamin.backbone.blocks import PsiFormerBlock, SimpleJastrow, Envelop


class PsiFormer(nn.Module):
    num_of_determinants: int
    num_of_electrons: int
    num_of_nucleus: int

    num_of_blocks: int
    num_heads: int
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
                                           (batch, num_of_electrons, num_of_nucleus, 4)
        :param single_electron_features: the single electron features tensor, should have the shape
                                           (batch, num_of_electrons, 4)
        :return:
        """

        x = rearrange(electron_nuclear_features, "b n c f -> b n (c f)")
        x = nn.Dense(
            features=self.num_heads,
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

        summed_dets = Envelop(
            num_of_determinants=self.num_of_determinants,
            num_of_electrons=self.num_of_electrons,
            num_of_nucleus=self.num_of_nucleus,
            param_dtype=self.param_dtype,
            computation_dtype=self.computation_dtype,
        )(jnp.expand_dims(electron_nuclear_features[..., -1], -1), psiformer_pre_det)

        jastrow_factor = SimpleJastrow()(single_electron_features)

        return summed_dets * jnp.exp(jastrow_factor)


# import jax

# print(PsiFormer(num_of_determinants=6,
#                num_of_electrons=2,
#                num_of_nucleus=2,
#                num_of_blocks=5,
#                num_heads=8).tabulate(jax.random.PRNGKey(0),
#                                       jnp.ones((512, 2, 2, 4)), jnp.ones((512, 2, 4)),
#                                       depth=1, console_kwargs={'width': 150}))
