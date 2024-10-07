import flax.linen as nn
import jax.numpy as jnp
import jax


class MultiHeadAttention(nn.Module):
    """
    this is an implementation of multi-head attention for PsiFormer.

    :cvar output_channels: number of projected output channels.
    :cvar num_heads: number of attention heads.
    :cvar use_memory_efficient_attention: whether to use memory efficient attention.
    :cvar group: number of groups used for GroupNorm. If None, LayerNorm will be used.
    :cvar use_qkv_bias: whether to use bias in the QKV matrix.
    :cvar use_dropout: whether to use dropout in the attention layer.
    :cvar dropout_rate: dropout rate for the attention dropout, only used if the use_dropout was set to True.

    :cvar param_dtype: parameter dtype for all layers. defaults to `jnp.float32`.
    :cvar computation_dtype: computation dtype for all layers. defaults to `jnp.float32`.

    """

    output_channels: int
    num_heads: int = 4
    use_norm: bool = False
    group: int | None = None
    use_qkv_bias: bool = False
    kernel_init: nn.initializers.Initializer = jax.nn.initializers.variance_scaling(1., mode='fan_in',
                                                                                    distribution='normal')

    param_dtype: jnp.dtype = jnp.float32
    computation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
            self, x: jnp.ndarray) -> jnp.ndarray:

        batch, num_electrons, embed_size = x.shape

        query = nn.Dense(
            features=embed_size,
            use_bias=self.use_qkv_bias,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.normal(stddev=1),
            dtype=self.computation_dtype,
            param_dtype=self.param_dtype,
        )(x)

        key = nn.Dense(
            features=embed_size,
            use_bias=self.use_qkv_bias,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.normal(stddev=1),
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )(x)

        value = nn.Dense(
            features=embed_size,
            use_bias=self.use_qkv_bias,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.normal(stddev=1),
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )(x)

        query = jnp.reshape(query, (batch, num_electrons, self.num_heads, -1))
        key = jnp.reshape(key, (batch, num_electrons, self.num_heads, -1))
        value = jnp.reshape(value, (batch, num_electrons, self.num_heads, -1))

        head_size = query.shape[-1]

        query_key = jnp.einsum('...thd,...Thd->...htT', query, key)
        query_key *= 1. / jnp.sqrt(head_size)
        query_key = nn.softmax(query_key)

        qkv = jnp.einsum('...htT,...Thd->...thd', query_key, value)
        qkv = jnp.reshape(qkv, (batch, num_electrons, embed_size))

        attention_output = nn.Dense(
            features=self.output_channels,
            use_bias=self.use_qkv_bias,
            kernel_init=self.kernel_init,
            bias_init=nn.initializers.normal(stddev=1),
            dtype=self.computation_dtype,
            param_dtype=self.param_dtype,
        )(qkv)

        if self.use_norm is not False:
            if self.group is not None:
                attention_output = nn.GroupNorm(
                    num_groups=self.group
                    if x.shape[-1] % self.group == 0
                    else x.shape[-1],
                    group_size=None,
                    param_dtype=self.param_dtype,
                )(attention_output)
            else:
                attention_output = nn.LayerNorm(
                    dtype=self.computation_dtype,
                    param_dtype=self.param_dtype,
                    use_scale=True,
                    use_bias=True,
                    scale_init=nn.initializers.ones,
                    bias_init=nn.initializers.zeros,
                    epsilon=1e-5,
                )(attention_output)

        return attention_output


"""
print(MultiHeadCrossAttention(256, 4).tabulate(
        jax.random.PRNGKey(0),
        jnp.ones((512, 2, 256), dtype=jnp.float16),
        False,
        depth=3,
        console_kwargs={"width": 150},
))
#"""
