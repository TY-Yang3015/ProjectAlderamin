import flax.linen as nn
import jax.numpy as jnp
import jax
from alderamin.backbone.blocks.memory_efficient_attention import (
    MemoryEfficientAttention,
)
from einops import rearrange


class MultiHeadCrossAttention(nn.Module):
    """
    This is a light wrapper around `flax.linen.MultiHeadDotProductAttention` with a GroupNorm (can be changed to LayerN
    orm) and an output projection. The memory efficient attention mechanism from diffuser by HuggingFace is adapted.
    The softmax activation is applied over the last dimension before output.

    Note: **this is a CROSS-attention layer so context can be passed into it.**

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
    use_memory_efficient_attention: bool = False
    group: int | None = None
    use_qkv_bias: bool = True
    use_dropout: bool = False
    dropout_rate: float = 0.1

    param_dtype: jnp.dtype = jnp.float32
    computation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool, context: jnp.ndarray | None = None
    ) -> jnp.ndarray:

        if self.group is not None:
            x = nn.GroupNorm(
                num_groups=self.group if x.shape[-1] % self.group == 0 else x.shape[-1],
                group_size=None,
                param_dtype=self.param_dtype,
            )(x)
        else:
            x = nn.LayerNorm(
                dtype=self.computation_dtype, param_dtype=self.param_dtype
            )(x)
        shape = x.shape

        if self.use_memory_efficient_attention:
            x = MemoryEfficientAttention(
                query_dim=shape[-1],
                heads=self.num_heads,
                dim_head=self.output_channels,
                dropout=0 if self.use_dropout is False else self.dropout_rate,
                dtype=self.computation_dtype,
            )(x, deterministic=not train, context=context)
        else:
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=shape[-1],
                out_features=self.output_channels,
                dropout_rate=0 if self.use_dropout is False else self.dropout_rate,
                deterministic=not train,
                use_bias=self.use_qkv_bias,
                dtype=self.computation_dtype,
                param_dtype=self.param_dtype,
            )(x, inputs_k=context, inputs_v=context)

        x = x.reshape(shape)
        x = nn.softmax(x, axis=-1)

        return x


print(
    MultiHeadCrossAttention(512, 8).tabulate(
        jax.random.PRNGKey(0),
        jnp.ones((5, 2, 512), dtype=jnp.float16),
        False,
        depth=1,
        console_kwargs={"width": 150},
    )
)
