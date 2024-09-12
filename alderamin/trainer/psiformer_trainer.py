import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
from jax import random
from jax.lib import xla_bridge
from functools import partial
from einops import repeat
import jax
import logging
import folx
import optax
from optax_shampoo import shampoo

from alderamin.backbone.models import PsiFormer
from alderamin.config import PsiFormerConfig
from alderamin.sampler import MetropolisHastingSampler


class PsiFormerTrainer:
    def __init__(self, config: PsiFormerConfig):
        self.config = config

        # log environment information
        logging.info(f"JAX backend: {xla_bridge.get_backend().platform}")

        logging.info(f"JAX process: {jax.process_index() + 1} / {jax.process_count()}")
        logging.info(f"JAX local devices: {jax.local_devices()}")

        # create sampler
        self.sampler = MetropolisHastingSampler(
            system=self.config.sampler_spec.system,
            batch_size=self.config.sampler_spec.batch_size,
            sampling_seed=self.config.sampler_spec.sampling_seed,
            target_acceptance=self.config.sampler_spec.target_acceptance,
            init_width=self.config.sampler_spec.init_width,
            sample_width=self.config.sampler_spec.sample_width,
            sample_width_adapt_freq=self.config.sampler_spec.sample_width_adapt_freq,
            log_epsilon=self.config.hyperparams.log_epsilon,
            computation_dtype=self.config.sampler_spec.computation_dtype,
            scale_input=self.config.hyperparams.scale_input,
        )

        # burn-in steps
        if self.config.hyperparams.burn_in_steps is not None:
            self.sampler.burn_in(self.config.hyperparams.burn_in_steps)

        # build neural network model
        self.psiformer = PsiFormer(
            num_of_determinants=self.config.nn_spec.num_of_determinants,
            num_of_electrons=self.config.nn_spec.num_of_electrons,
            num_of_nucleus=self.config.nn_spec.num_of_nucleus,
            num_of_blocks=self.config.nn_spec.num_of_blocks,
            num_heads=self.config.nn_spec.num_heads,
            qkv_size=self.config.nn_spec.qkv_size,
            use_memory_efficient_attention=self.config.nn_spec.use_memory_efficient_attention,
            group=self.config.nn_spec.group,
            computation_dtype=self.config.nn_spec.computation_dtype,
            param_dtype=self.config.nn_spec.param_dtype
        )

        # make some handy alias
        self.system = self.config.sampler_spec.system
        self.num_of_electrons = self.system.total_electrons
        self.num_of_nucleus = self.system.total_nucleus
        self.nuc_charges = jnp.array([nuc.charge for nuc in self.system.nucleus_list])
        self.nuc_positions = jnp.array([member.position for member in self.system.nucleus_list])
        self.spins = jnp.array([electron.spin for electron in self.system.electrons_list], dtype=jnp.float32)

        # initialise optimiser
        self.optimiser = optax.adam(self.config.hyperparams.learning_rate)
        #self.optimiser = shampoo(self.config.hyperparams.learning_rate, 128)
        self.optimiser = optax.chain(
            optax.clip_by_global_norm(self.config.hyperparams.gradient_clipping),
            self.optimiser,
        )

    @partial(jax.jit, static_argnums=0)
    def convert_to_psiformer_input(self, coordinates: jnp.ndarray) \
            -> (jnp.ndarray, jnp.ndarray):
        """

        :param coordinates: sampled cartesian coordinates, with shape (batch, N, 3)
        :return:
        """
        if coordinates.ndim == 2:
            coordinates = coordinates[None, ...]

        assert coordinates.ndim == 3

        batch = coordinates.shape[0]
        electron_nuclear_features = jnp.zeros((batch, self.num_of_electrons,
                                               len(self.nuc_positions), 4),
                                              dtype=self.config.sampler_spec.computation_dtype)

        spins_reshaped = repeat(self.spins, f"e -> {coordinates.shape[0]} e 1")
        single_electron_features = coordinates
        single_electron_features = jnp.concatenate([single_electron_features,
                                                    spins_reshaped], axis=-1)

        for i in range(self.num_of_electrons):
            for j in range(len(self.nuc_positions)):
                electron_nuclear_features = electron_nuclear_features.at[:, i, j, :3].set(
                    coordinates[:, i, :] - self.nuc_positions[j, :]
                )
                electron_nuclear_features = electron_nuclear_features.at[:, i, j, -1].set(
                    jnp.linalg.norm(coordinates[:, i, :] - self.nuc_positions[j, :], axis=-1)
                )

        spins_reshaped = repeat(self.spins, f"e -> {coordinates.shape[0]} "
                                            f"e {self.num_of_nucleus} 1")

        electron_nuclear_features = jnp.concatenate([electron_nuclear_features,
                                                     spins_reshaped], axis=-1)

        if self.config.hyperparams.scale_input:
            electron_nuclear_features = electron_nuclear_features.at[:, :, :, :4].set(
                electron_nuclear_features[:, :, :, :4] *
                jnp.expand_dims((jnp.log(1. + electron_nuclear_features[..., 3])
                                 / electron_nuclear_features[..., 3]), 3))

        return electron_nuclear_features, single_electron_features

    def _get_electric_hamiltonian(self, coordinates: jnp.ndarray) -> jnp.ndarray:
        elec_elec_term = jnp.zeros((self.config.hyperparams.batch_size, 1))
        elec_nuc_term = jnp.zeros((self.config.hyperparams.batch_size, 1))
        nuc_nuc_term = jnp.zeros((self.config.hyperparams.batch_size, 1))

        # TODO: double check after run
        for i in range(self.num_of_electrons):
            for j in range(i):
                elec_elec_term = elec_elec_term.at[:, 0].add(1. / jnp.linalg.norm(coordinates[:, i, :]
                                                                                  - coordinates[:, j, :],
                                                                                  axis=-1))

        for I in range(self.num_of_nucleus):
            for i in range(self.num_of_electrons):
                elec_nuc_term = elec_nuc_term.at[:, 0].add((self.nuc_charges[I] / jnp.linalg.norm(coordinates[:, i, :]
                                                                                                  - self.nuc_positions[
                                                                                                    I, :],
                                                                                                  axis=-1)
                                                            ))

        for I in range(self.num_of_nucleus):
            for J in range(I):
                nuc_nuc_term = nuc_nuc_term.at[:, 0].add(
                    (self.nuc_charges[I] * self.nuc_charges[J]) /
                    jnp.linalg.norm(self.nuc_positions[I, :] - self.nuc_positions[J, :],
                                    axis=-1)
                )

        return elec_elec_term - elec_nuc_term + nuc_nuc_term

    @partial(jax.jit, static_argnums=0)
    def _train_step(self, batch, state):

        def get_logabs_wavefunction(raw_batch):
            converted_batch = self.convert_to_psiformer_input(raw_batch)
            wavefunction = state.apply_fn(
                {"params": state.params},
                *converted_batch,
            )

            if wavefunction.shape == (1, 1):
                wavefunction = jnp.float32(wavefunction[0][0])

            return wavefunction

        def get_grad(input_tensor):
            return jax.grad(get_logabs_wavefunction)(input_tensor)

        def get_energy(params):
            electric_term = self._get_electric_hamiltonian(batch)

            #jacobian = jax.vmap(get_grad)(batch)
            #laplacian_op = jax.grad(lambda x: jnp.sum(get_grad(x)))
            #laplacian = jax.vmap(laplacian_op)(batch)
            #kinetic_term = -(jnp.square(jacobian).sum(axis=(-1, -2)) + laplacian.sum(axis=(-1, -2))) * 0.5

            laplacian_op = jax.jit(jax.vmap(folx.ForwardLaplacianOperator(0)(get_logabs_wavefunction)))
            laplacian, jacobian = laplacian_op(batch)
            #laplacian, jacobian = result.laplacian, result.jacobian.dense_array
            kinetic_term = -(jnp.square(jacobian) + laplacian).sum(axis=-1) * 0.5

            kinetic_term = kinetic_term.reshape(-1, 1)

            energy_batch = electric_term + kinetic_term

            mean_absolute_deviation = jnp.mean(jnp.abs(energy_batch - jnp.median(energy_batch)))
            n = self.config.hyperparams.mad_clipping_factor
            energy_batch = jnp.clip(energy_batch,
                                    jnp.median(energy_batch) - (n * mean_absolute_deviation),
                                    jnp.median(energy_batch) + (n * mean_absolute_deviation))

            return energy_batch.mean()

        energy, grads = jax.value_and_grad(get_energy)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, get_energy(state.params)

    def train(self):

        logging.info("initializing model.")
        init_elec_nuc = jnp.ones(
            (
                self.config.hyperparams.batch_size,
                self.config.nn_spec.num_of_electrons,
                self.config.nn_spec.num_of_nucleus,
                5,
            ),
            jnp.float32,
        )

        init_single_elec = jnp.ones(
            (
                self.config.hyperparams.batch_size,
                self.config.nn_spec.num_of_electrons,
                4,
            ),
            jnp.float32,
        )

        rngs = {"params": random.PRNGKey(self.config.hyperparams.training_seed)}
        params = self.psiformer.init(rngs, init_elec_nuc, init_single_elec)["params"]

        state = TrainState.create(
            apply_fn=self.psiformer.apply,
            params=params,
            tx=self.optimiser,
        )

        for step in range(self.config.hyperparams.step):
            batch = self.sampler.sample_psiformer(state,
                                                  self.config.hyperparams.sample_steps)

            state, energy = self._train_step(batch, state)
            logging.info(f"step: {step} "
                         f"energy: {energy}"
                         f"")
