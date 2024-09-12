import flax.linen as nn
import jax.numpy as jnp


class SimpleJastrow(nn.Module):
    """
    make the electron-electron jastrow factor which satisfies the electron-electron cusp condition. see the
    https://arxiv.org/abs/2211.13672 PsiFormer paper for more details.
    """

    def setup(self):
        self.alpha_parallel = self.param("alpha_parallel", nn.initializers.ones, (1, ))
        self.alpha_anti = self.param("alpha_anti", nn.initializers.ones, (1, ))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        expecting input shape (batch, num_of_electrons, 4)
        where 4 = 3 (position) + 1 (spin)

        :param x: the input electron feature vectors.

        :return: the electron-electron jastrow factors with dimension (batch, 1).
        """

        parallel_term = 0.
        anti_term = 0.

        for i in range(x.shape[1]):
            for j in range(i):
                parallel_term += jnp.where(
                    x[:, i, -1] == x[:, j, -1],
                    (
                        jnp.square(self.alpha_parallel)
                        / (
                            self.alpha_parallel
                            + jnp.linalg.norm(x[:, i, :3] - x[:, j, :3], axis=-1)
                        )
                    ),
                    0,
                )
                anti_term += jnp.where(
                    x[:, i, -1] == x[:, j, -1],
                    0,
                    (
                        jnp.square(self.alpha_anti)
                        / (
                            self.alpha_anti
                            + jnp.linalg.norm(x[:, i, :3] - x[:, j, :3], axis=-1)
                        )
                    ),
                )

        parallel_term *= -0.25
        anti_term *= -0.5

        return jnp.expand_dims(parallel_term + anti_term, -1)


# (SimpleJastrow().tabulate(jax.random.PRNGKey(0),
#                               jnp.ones((4096, 4, 512)),
#                               depth=1, console_kwargs={'width': 150}))
