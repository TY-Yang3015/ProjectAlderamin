import flax.linen as nn
import jax.numpy as jnp

from psiflax.backbone.blocks import MultiHeadAttention


class PsiFormerBlock(nn.Module):
    """
    this is the building block for psiformer electron-nucleus pipeline. contains the
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
    kernel_init: nn.initializers.Initializer
    bias_init: nn.initializers.Initializer
    use_norm: bool
    group: None | int = None

    param_dtype: jnp.dtype = jnp.float32
    computation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        embed_dim = x.shape[-1]

        h = MultiHeadAttention(
            num_heads=self.num_heads,
            output_channels=embed_dim,
            use_norm=self.use_norm,
            group=self.group,
            use_qkv_bias=False,
            kernel_init=self.kernel_init,
            computation_dtype=self.computation_dtype,
            param_dtype=self.param_dtype,
        )(x)

        x += h
        x += nn.tanh(
            nn.Dense(
                features=embed_dim,
                use_bias=True,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.computation_dtype,
                param_dtype=self.param_dtype,
            )(
                nn.tanh(
                    nn.Dense(
                        features=256,
                        use_bias=True,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        dtype=self.computation_dtype,
                        param_dtype=self.param_dtype,
                    )(x)
                )
            )
        )

        if self.use_norm is not False:
            if self.group is not None:
                x = nn.GroupNorm(
                    num_groups=self.group
                    if x.shape[-1] % self.group == 0
                    else x.shape[-1],
                    group_size=None,
                    param_dtype=self.param_dtype,
                )(x)
            else:
                x = nn.LayerNorm(
                    dtype=self.computation_dtype,
                    param_dtype=self.param_dtype,
                    use_scale=True,
                    use_bias=True,
                    scale_init=nn.initializers.ones,
                    bias_init=nn.initializers.zeros,
                    epsilon=1e-5,
                )(x)

        return x


"""
import jax
print(PsiFormerBlock(4, False,
                     kernel_init=nn.initializers.kaiming_normal(),
                     bias_init=nn.initializers.zeros,
                     use_norm=False).tabulate(jax.random.PRNGKey(0),
                                       jnp.ones((4096, 5, 256)),
                                       depth=1, console_kwargs={'width': 150}))
#"""
