import flax.linen as nn
import jax.numpy as jnp

from einops import repeat


class MLPElectronJastrow(nn.Module):
    """
    make the MLP-based electron-electron jastrow factor. This is the varaint of the positronic
    jastrow factor in the paper https://doi.org/10.1038/s41467-024-49290-1 "Neural network variational Monte
    Carlo for positronic chemistry" by G. Cassella, D. Pfau.
    """

    num_layers: int = 2
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        expecting input shape (batch, num_of_electrons, 4)
        where 4 = 3 (position) + 1 (spin)

        :param x: the input electron feature vectors.

        :return: the electron-electron jastrow factors with dimension (batch, 1).
        """

        electron_electron_features = repeat(x, "b n f -> b n m f", m=x.shape[1])

        for i in range(self.num_layers):
            electron_electron_features = nn.Dense(
                features=self.hidden_dim,
                use_bias=True,
                bias_init=nn.initializers.zeros,
                kernel_init=nn.initializers.normal(),
            )(electron_electron_features)
            electron_electron_features = nn.tanh(electron_electron_features)
            electron_electron_features = nn.LayerNorm(
                epsilon=1e-5,
                use_bias=True,
                use_scale=True,
                bias_init=nn.initializers.zeros,
                scale_init=nn.initializers.ones,
            )(electron_electron_features)

        output = nn.Dense(features=1, use_bias=False)(electron_electron_features)

        return output.sum(axis=(1, 2, 3)).reshape(-1, 1)


# import jax
# print(MLPElectronJastrow().tabulate(jax.random.PRNGKey(0),
#                               jnp.ones((4096, 4, 512)),
#                               depth=1, console_kwargs={'width': 150}))
