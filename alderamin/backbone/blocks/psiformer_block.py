import flax.linen as nn
import jax.numpy as jnp

from alderamin.backbone.blocks import MultiHeadCrossAttention


class PsiformerBlock(nn.Module):
    """
    this is the building block for Psiformer electron-nucleus pipeline. contains the
    multi-head attention block and linear projection layer with residual connection.
    see https://arxiv.org/abs/2211.13672 for the paper.

    :cvar num_heads: The number of heads in the multi-head attention block.
    :cvar use_memory_efficient_attention: whether to use memory efficient attention. see the
                                        doc for MultiHeadCrossAttention layer for more details.
    :cvar group: set to None for LayerNorm, otherwise GroupNorm will be used. default: None.

    :cvar param_dtype: the dtype of the parameters. default 'jnp.float32'.
    :cvar computation_dtype: the dtype of the computation. default 'jnp.float32'.

    """

    num_heads: int
    use_memory_efficient_attention: bool
    group: None | int = None

    param_dtype: jnp.dtype = jnp.float32
    computation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        h = MultiHeadCrossAttention(num_heads=self.num_heads,
                                    output_channels=x.shape[-1],
                                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                                    group=self.group,
                                    use_qkv_bias=True,
                                    use_dropout=False,
                                    computation_dtype=self.computation_dtype,
                                    param_dtype=self.param_dtype
                                    )(x, False, None)
        h = nn.Dense(features=x.shape[-1],
                     use_bias=False,
                     dtype=self.computation_dtype,
                     param_dtype=self.param_dtype)(h)

        h += x
        h += nn.tanh(
            nn.Dense(features=x.shape[-1],
                     use_bias=True,
                     dtype=self.computation_dtype,
                     param_dtype=self.param_dtype
                     )(h)
        )

        return h


import jax
print(PsiformerBlock(4, False).tabulate(jax.random.PRNGKey(0),
                                       jnp.ones((4096, 5, 4)),
                                       depth=1, console_kwargs={'width': 150}))
