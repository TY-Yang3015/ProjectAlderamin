from flax.training.train_state import TrainState
import jax.numpy as jnp
from clu import metric_writers
from jax import random, tree_map
from jax.lib import xla_bridge
from functools import partial
from omegaconf import DictConfig, OmegaConf
import jax
import logging
import optax
import hydra
import os
import orbax.checkpoint as ocp

import psiflax.folx as folx
from psiflax.shampoo.distributed_shampoo import distributed_shampoo as shampoo
from psiflax.backbone.models import PsiFormer
from psiflax.data import GlobalSystem
from psiflax.sampler import MetropolisHastingSampler
from psiflax.utils import log_histograms
from psiflax.hamiltonian import VanillaHamiltonian


class PsiFormerTrainer:
    def __init__(self, config: DictConfig, system: GlobalSystem):
        self.config = config

        logger = logging.getLogger("env")
        logger.setLevel(logging.INFO)
        # log environment information
        logger.info(f"JAX backend: {xla_bridge.get_backend().platform}")

        logger.info(f"JAX process: {jax.process_index() + 1} / {jax.process_count()}")
        logger.info(f"JAX local devices: {jax.local_devices()}")

        # create sampler
        self.sampler = MetropolisHastingSampler(
            system=system,
            batch_size=self.config.hyperparam.batch_size,
            sampling_seed=self.config.sampler.sampling_seed,
            acceptance_range=self.config.sampler.acceptance_range,
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

        # build neural network model
        self.psiformer = PsiFormer(
            num_of_determinants=self.config.psiformer.num_of_determinants,
            num_of_electrons=self.system.total_electrons,
            num_of_nucleus=self.system.total_nucleus,
            num_of_blocks=self.config.psiformer.num_of_blocks,
            num_heads=self.config.psiformer.num_heads,
            qkv_size=self.config.psiformer.qkv_size,
            use_memory_efficient_attention=self.config.psiformer.use_memory_efficient_attention,
            use_norm=self.config.psiformer.use_norm,
            group=self.config.psiformer.group,
            computation_dtype=self.config.psiformer.computation_dtype,
            param_dtype=self.config.psiformer.param_dtype,
            spin_counts=system.spin_counts,
            nuc_positions=self.nuc_positions,
            scale_input=self.config.hyperparam.scale_input,
        )

        # initialise optimiser
        if self.config.optimiser.type.casefold() == "adam":

            def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
                return self.config.optimiser.adam.init_learning_rate * jnp.power(
                    (1.0 / (1.0 + (t_ / self.config.lr.delay))), self.config.lr.decay
                )

            self.optimiser = optax.adam(
                learning_rate=learning_rate_schedule,
                b1=self.config.optimiser.adam.b1,
                b2=self.config.optimiser.adam.b2,
            )
        elif self.config.optimiser.type.casefold() == "shampoo":

            def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
                return self.config.optimiser.shampoo.learning_rate * jnp.power(
                    (1.0 / (1.0 + (t_ / self.config.lr.delay))), self.config.lr.decay
                )

            self.optimiser = shampoo(
                learning_rate=learning_rate_schedule,
                beta1=self.config.optimiser.shampoo.beta1,
                beta2=self.config.optimiser.shampoo.beta2,
                block_size=self.config.optimiser.shampoo.block_size,
                diagonal_epsilon=self.config.optimiser.shampoo.diagonal_epsilon,
                matrix_epsilon=self.config.optimiser.shampoo.matrix_epsilon,
                weight_decay=self.config.optimiser.shampoo.weight_decay,
                start_preconditioning_step=self.config.optimiser.shampoo.start_preconditioning_step,
                preconditioning_compute_steps=self.config.optimiser.shampoo.preconditioning_compute_steps,
                statistics_compute_steps=self.config.optimiser.shampoo.statistics_compute_steps,
                best_effort_shape_interpretation=self.config.optimiser.shampoo.best_effort_shape_interpretation,
                nesterov=self.config.optimiser.shampoo.nesterov,
                exponent_override=self.config.optimiser.shampoo.exponent_override,
                shard_optimizer_states=self.config.optimiser.shampoo.shard_optimizer_states,
                best_effort_memory_usage_reduction=self.config.optimiser.shampoo.best_effort_memory_usage_reduction,
                inverse_failure_threshold=self.config.optimiser.shampoo.inverse_failure_threshold,
                moving_average_for_momentum=self.config.optimiser.shampoo.moving_average_for_momentum,
                skip_preconditioning_dim_size_gt=self.config.optimiser.shampoo.skip_preconditioning_dim_size_gt,
                decoupled_learning_rate=self.config.optimiser.shampoo.decoupled_learning_rate,
                decoupled_weight_decay=self.config.optimiser.shampoo.decoupled_weight_decay,
            )
        else:
            raise NotImplementedError(
                f"optimiser {self.config.optimiser.type} not available."
            )

        self.optimiser = optax.chain(
            optax.clip_by_global_norm(self.config.hyperparam.gradient_clipping),
            self.optimiser,
        )

        self.hamiltonian = VanillaHamiltonian(
            batch_size=self.config.hyperparam.batch_size,
            num_of_electrons=self.num_of_electrons,
            nuc_charges=self.nuc_charges,
            nuc_positions=self.nuc_positions,
        )

    def _init_savedir(self) -> str:
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_dir = str(os.path.join(save_dir, "results"))
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            logging.info("save directory initialised already.")
        return save_dir

    @partial(jax.jit, static_argnums=0, donate_argnums=2)
    def _train_step(self, batch, state):
        def get_energy_and_grad(params):
            energy_batch = self.hamiltonian.get_local_energy(state, batch)

            # mad clipping
            mean_absolute_deviation = jnp.mean(
                jnp.abs(energy_batch - jnp.median(energy_batch))
            )
            n = self.config.hyperparam.mad_clipping_factor
            energy_batch = jnp.clip(
                energy_batch,
                jnp.median(energy_batch) - (n * mean_absolute_deviation),
                jnp.median(energy_batch) + (n * mean_absolute_deviation),
            )

            output_energy = energy_batch.mean()
            output_var = energy_batch.var()

            def param_to_wavefunction(param_tree):
                wavefunction = state.apply_fn(
                    {"params": param_tree},
                    batch,
                )

                return wavefunction.squeeze(-1)

            mean_energy = energy_batch.mean()
            energy_batch -= mean_energy
            energy_batch = energy_batch.squeeze(-1)

            total_grad = jax.vjp(param_to_wavefunction, state.params)[1](energy_batch)[
                0
            ]
            mean_grad = tree_map(lambda g: g / energy_batch.shape[0], total_grad)

            return (output_energy, output_var), mean_grad

        e_and_v, grad = get_energy_and_grad(state.params)
        state = state.apply_gradients(grads=grad)

        return state, e_and_v, grad

    def train(self):
        logger = logging.getLogger("loop")
        logger.info("initializing model.")
        init_elec = jnp.ones(
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

        sharding = jax.sharding.NamedSharding(
            mesh=jax.sharding.Mesh(jax.devices(), axis_names="model"),
            spec=jax.sharding.PartitionSpec(),
        )

        create_sharded_array = lambda x: jax.device_put(x, sharding)
        state = jax.tree_util.tree_map(create_sharded_array, state)

        save_dir = self._init_savedir()

        if self.config.ckpt.save_ckpt:
            save_vae_path = ocp.test_utils.erase_and_create_empty(
                os.path.abspath(save_dir + "/ckpt")
            )

            save_options = ocp.CheckpointManagerOptions(
                max_to_keep=self.config.ckpt.save_num_ckpt,
                save_interval_steps=self.config.ckpt.ckpt_freq,
            )

            mngr = ocp.CheckpointManager(
                save_vae_path, options=save_options, item_names=("state", "config")
            )

        writer = metric_writers.SummaryWriter(logdir=save_dir)

        for step in range(self.config.hyperparam.step):
            batch, pmean = self.sampler.sample_psiformer(
                state, self.config.sampler.sample_steps
            )
            state, energy_and_var, grad = self._train_step(batch, state)

            if self.config.ckpt.save_ckpt:
                config_dict = OmegaConf.to_container(self.config, resolve=True)
                mngr.save(
                    step,
                    args=ocp.args.Composite(
                        state=ocp.args.StandardSave(state),
                        config=ocp.args.JsonSave(config_dict),
                    ),
                )

            if self.config.log.log_grad_and_params:
                log_histograms(writer, state.params, grad, step)

            writer.write_scalars(
                step, {"energy": energy_and_var[0], "var": energy_and_var[1]}
            )

            if self.config.log.log_pmean:
                writer.write_scalars(step, {"pmean": pmean})

            logger.info(
                f"step: {step} "
                f"energy: {energy_and_var[0]:.4f} "
                f"var: {energy_and_var[1]:.4f} "
                f"pmean: {pmean:.4f}"
            )

        writer.flush()

        return state
