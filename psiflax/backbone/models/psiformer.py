import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import repeat

from psiflax.backbone.blocks import PsiFormerBlock, SimpleJastrow, Envelop, MLPElectronJastrow
from psiflax.utils.logdet import signed_log_sum_exp

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

    spins: list
    nuc_positions: list
    scale_input: bool

    num_of_blocks: int
    num_heads: int
    qkv_size: int
    use_memory_efficient_attention: bool = False
    use_norm: bool = False
    group: None | int = None

    computation_dtype: jnp.dtype | str = "float32"
    param_dtype: jnp.dtype | str = "float32"

    def convert_to_input(
        self, coordinates: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        :param coordinates: the coordinates of the input coordinates, should have the shape
                            ``(batch, num_of_electrons, 3)``
        :return: electron-nuclear features ``(batch, num_of_electrons, num_of_nucleus, 5)``
                            , single-electron features ``(batch, num_of_electrons, 4)``
        """

        if coordinates.ndim == 2:
            coordinates = coordinates[None, ...]

        assert coordinates.ndim == 3

        batch = coordinates.shape[0]
        electron_nuclear_features = jnp.ones(
            (batch, self.num_of_electrons, len(self.nuc_positions), 4),
            dtype=self.computation_dtype,
        )

        spins_reshaped = repeat(self.spins, f"e -> {coordinates.shape[0]} e 1")
        single_electron_features = coordinates
        single_electron_features = jnp.concatenate(
            [single_electron_features, spins_reshaped], axis=-1
        )

        for i in range(self.num_of_electrons):
            for j in range(len(self.nuc_positions)):
                electron_nuclear_features = electron_nuclear_features.at[
                    :, i, j, :3
                ].set(coordinates[:, i, :] - self.nuc_positions[j, :])
                electron_nuclear_features = electron_nuclear_features.at[
                    :, i, j, 3
                ].set(
                    jnp.linalg.norm(
                        coordinates[:, i, :] - self.nuc_positions[j, :], axis=-1
                    )
                )

        spins_reshaped = repeat(
            self.spins, f"e -> {coordinates.shape[0]} e {len(self.nuc_positions)} 1"
        )

        electron_nuclear_features = jnp.concatenate(
            [electron_nuclear_features, spins_reshaped], axis=-1
        )

        if self.scale_input:
            electron_nuclear_features = electron_nuclear_features.at[:, :, :, :4].set(
                electron_nuclear_features[:, :, :, :4]
                * jnp.expand_dims(
                    (
                        jnp.log(1. + electron_nuclear_features[..., 3])
                        / electron_nuclear_features[..., 3]
                    ),
                    3,
                )
            )

        return electron_nuclear_features, single_electron_features

    @nn.compact
    def __call__(
        self,
        coordinates: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        :param coordinates: the electronic nuclear features tensor, should have the shape
                                           (batch, num_of_electrons, 3)
        :return: wavefunction values with shape (batch, 1)
        """

        electron_nuclear_features, single_electron_features = self.convert_to_input(
            coordinates
        )

        x = repeat(electron_nuclear_features, "b n c f -> b n (c f)")
        x = nn.Dense(
            features=self.num_heads * self.qkv_size,
            use_bias=False,
            dtype=self.computation_dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(),
        )(x)

        for _ in range(self.num_of_blocks):
            x = PsiFormerBlock(
                num_heads=self.num_heads,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                use_norm=self.use_norm,
                group=self.group,
                param_dtype=self.param_dtype,
                computation_dtype=self.computation_dtype,
                kernel_init=nn.initializers.normal(),
                bias_init=nn.initializers.normal(),
            )(x)

        psiformer_pre_det = nn.Dense(
            features=self.num_of_electrons * self.num_of_determinants,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            dtype=self.computation_dtype,
            param_dtype=self.param_dtype,
        )(x)

        wavefunction = Envelop(
            num_of_determinants=self.num_of_determinants,
            num_of_electrons=self.num_of_electrons,
            num_of_nucleus=self.num_of_nucleus,
            param_dtype=self.param_dtype,
            computation_dtype=self.computation_dtype,
        )(jnp.expand_dims(electron_nuclear_features[..., 3], -1), psiformer_pre_det)

        jastrow_factor = SimpleJastrow()(single_electron_features)

        wavefunction *= jnp.exp(jastrow_factor)

        return jnp.log(jnp.abs(wavefunction))


"""
import jax

print(PsiFormer(num_of_determinants=6,
                num_of_electrons=2,
                num_of_nucleus=2,
                num_of_blocks=5,
                num_heads=8,
                qkv_size=64,
                scale_input=True,
                spins=jnp.array([-1, 1]),
                nuc_positions=jnp.array([[1, 0, 0], [0, 0, 0]])).tabulate(jax.random.PRNGKey(0),
                                       jnp.ones((512, 2, 3)),
                                       depth=1, console_kwargs={'width': 150}))
#"""
