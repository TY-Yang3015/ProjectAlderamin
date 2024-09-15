from flax.training.train_state import TrainState
import jax.numpy as jnp
from clu import metric_writers
from jax import random
from jax.lib import xla_bridge
from functools import partial
from omegaconf import DictConfig
import jax
import logging
import optax

import alderamin.folx as folx
from alderamin.shampoo.distributed_shampoo import distributed_shampoo as shampoo
from alderamin.backbone.models import PsiFormer
from alderamin.data import GlobalSystem
from alderamin.sampler import MetropolisHastingSampler


writer = metric_writers.create_default_writer()
writer.write_histograms()

class PsiFormerTrainer:
    def __init__(self, config: DictConfig, system: GlobalSystem):
        self.config = config

        # log environment information
        logging.info(f"JAX backend: {xla_bridge.get_backend().platform}")

        logging.info(f"JAX process: {jax.process_index() + 1} / {jax.process_count()}")
        logging.info(f"JAX local devices: {jax.local_devices()}")

        # create sampler
        self.sampler = MetropolisHastingSampler(
            system=system,
            batch_size=self.config.hyperparam.batch_size,
            sampling_seed=self.config.sampler.sampling_seed,
            target_acceptance=self.config.sampler.target_acceptance,
            init_width=self.config.sampler.init_width,
            sample_width=self.config.sampler.sample_width,
            sample_width_adapt_freq=self.config.sampler.sample_width_adapt_freq,
            log_epsilon=self.config.hyperparam.log_epsilon,
            computation_dtype=self.config.sampler.computation_dtype,
            scale_input=self.config.hyperparam.scale_input,
        )

        # burn-in steps
        if self.config.sampler.burn_in_steps is not None:
            self.sampler.burn_in(self.config.sampler.burn_in_steps)

        # make some handy alias
        self.system = system
        self.num_of_electrons = self.system.total_electrons
        self.num_of_nucleus = self.system.total_nucleus
        self.nuc_charges = jnp.array([nuc.charge for nuc in self.system.nucleus_list])
        self.nuc_positions = jnp.array(
            [member.position for member in self.system.nucleus_list]
        )
        self.spins = jnp.array(
            [electron.spin for electron in self.system.electrons_list],
            dtype=jnp.float32,
        )

        # build neural network model
        self.psiformer = PsiFormer(
            num_of_determinants=self.config.psiformer.num_of_determinants,
            num_of_electrons=self.system.total_electrons,
            num_of_nucleus=self.system.total_nucleus,
            num_of_blocks=self.config.psiformer.num_of_blocks,
            num_heads=self.config.psiformer.num_heads,
            qkv_size=self.config.psiformer.qkv_size,
            use_memory_efficient_attention=self.config.psiformer.use_memory_efficient_attention,
            group=self.config.psiformer.group,
            computation_dtype=self.config.psiformer.computation_dtype,
            param_dtype=self.config.psiformer.param_dtype,
            spins=self.spins,
            nuc_positions=self.nuc_positions,
            scale_input=self.config.hyperparam.scale_input,
        )

        # initialise optimiser
        lr = optax.exponential_decay(
            init_value=self.config.hyperparam.learning_rate,
            transition_steps=self.config.hyperparam.step,
            transition_begin=1000,
            decay_rate=0.95,
            end_value=self.config.hyperparam.learning_rate / 2.0,
        )
        #self.optimiser = optax.adam(lr)
        self.optimiser = shampoo(self.config.hyperparam.learning_rate,
                                 block_size=128,
                                 diagonal_epsilon=1e-12,
                                 matrix_epsilon=1e-12)
        self.optimiser = optax.chain(
            optax.clip_by_global_norm(self.config.hyperparam.gradient_clipping),
            self.optimiser,
            # optax.add_decayed_weights(weight_decay=1e-4),
            optax.ema(0.99),
        )

    @partial(jax.jit, static_argnums=0)
    def _train_step(self, batch, state):
        def get_electric_hamiltonian(coordinates: jnp.ndarray) -> jnp.ndarray:
            elec_elec_term = jnp.zeros((self.config.hyperparam.batch_size, 1))
            elec_nuc_term = jnp.zeros((self.config.hyperparam.batch_size, 1))
            nuc_nuc_term = jnp.zeros((self.config.hyperparam.batch_size, 1))

            for i in range(self.num_of_electrons):
                for j in range(i):
                    elec_elec_term = elec_elec_term.at[:, 0].add(
                        1.0
                        / (
                            jnp.linalg.norm(
                                coordinates[:, i, :] - coordinates[:, j, :], axis=-1
                            )
                        )
                    )

            for I in range(self.num_of_nucleus):
                for i in range(self.num_of_electrons):
                    elec_nuc_term = elec_nuc_term.at[:, 0].add(
                        (
                            self.nuc_charges[I]
                            / (
                                jnp.linalg.norm(
                                    coordinates[:, i, :] - self.nuc_positions[I, :],
                                    axis=-1,
                                )
                            )
                        )
                    )

            for I in range(self.num_of_nucleus):
                for J in range(I):
                    nuc_nuc_term = nuc_nuc_term.at[:, 0].add(
                        (self.nuc_charges[I] * self.nuc_charges[J])
                        / jnp.linalg.norm(
                            self.nuc_positions[I, :] - self.nuc_positions[J, :], axis=-1
                        )
                    )

            return elec_elec_term - elec_nuc_term + nuc_nuc_term

        def get_energy(params):
            def get_wavefunction(raw_batch):
                wavefunction = state.apply_fn(
                    {"params": params},
                    raw_batch,
                )

                if wavefunction.shape == (1, 1):
                    wavefunction = jnp.float32(wavefunction[0][0])

                return wavefunction

            electric_term = get_electric_hamiltonian(batch)

            # jacobian_op = jax.grad(get_wavefunction)
            # jacobian = jax.vmap(jacobian_op)(batch)

            # laplacian_op = jax.grad(lambda x: jnp.sum(jacobian_op(x)))
            # laplacian = jax.vmap(laplacian_op)(batch)

            laplacian_op = folx.forward_laplacian(get_wavefunction)
            result = jax.vmap(laplacian_op)(batch)
            laplacian, jacobian = result.laplacian, result.jacobian.dense_array

            # print(jacobian.shape)
            # print(laplacian.shape)

            kinetic_term = -laplacian * 0.5

            kinetic_term = kinetic_term.reshape(-1, 1) / get_wavefunction(batch)

            energy_batch = electric_term + kinetic_term
            energy_batch = jnp.clip(energy_batch, None, 1.0)

            mean_absolute_deviation = jnp.mean(
                jnp.abs(energy_batch - jnp.median(energy_batch))
            )
            n = self.config.hyperparam.mad_clipping_factor
            energy_batch = jnp.clip(
                energy_batch,
                jnp.median(energy_batch) - (n * mean_absolute_deviation),
                jnp.median(energy_batch) + (n * mean_absolute_deviation),
            )

            return energy_batch.mean()

        energy, grads = jax.value_and_grad(get_energy)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, energy

    def train(self):

        logging.info("initializing model.")
        init_elec = jnp.zeros(
            (
                self.config.hyperparam.batch_size,
                self.system.total_electrons,
                3,
            ),
            jnp.float32,
        )

        rngs = {"params": random.PRNGKey(self.config.hyperparam.training_seed)}
        params = jax.jit(self.psiformer.init)(rngs, init_elec)["params"]

        state = TrainState.create(
            apply_fn=self.psiformer.apply,
            params=params,
            tx=self.optimiser,
        )

        for step in range(self.config.hyperparam.step):
            batch = self.sampler.sample_psiformer(
                state, self.config.sampler.sample_steps
            )
            # batch = self.sampler.walker_state.positions
            state, energy = self._train_step(batch, state)
            # print(batch[0, 0, :])
            logging.info(f"step: {step} " f"energy: {energy}" f"")

        return state
